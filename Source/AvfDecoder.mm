#include <span>
#include "AvfDecoder.h"
#include "SoyPixels.h"
#include "SoyAvf.h"
#include "SoyFourcc.h"
#include "SoyLib/src/magic_enum/include/magic_enum.hpp"
#include "json11.hpp"
#include "SoyTime.h"

#include <CoreMedia/CMBase.h>
#include <VideoToolbox/VTBase.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>
#include <CoreMedia/CMSampleBuffer.h>
#include <CoreMedia/CMFormatDescription.h>
#include <CoreMedia/CMTime.h>
#include <VideoToolbox/VTSession.h>
#include <VideoToolbox/VTCompressionProperties.h>
#include <VideoToolbox/VTCompressionSession.h>
#include <VideoToolbox/VTDecompressionSession.h>
#include <VideoToolbox/VTErrors.h>
#include "SoyH264.h"
#include "AvfPixelBuffer.h"

#include "PopH264.h"	//	param keys
#include "FileReader.hpp"
#include <span>

namespace Jpeg
{
	class Meta_t;
	static Meta_t		DecodeMeta(std::span<uint8_t> FileData);
}

class Jpeg::Meta_t
{
public:
	uint32_t	mWidth = 0;
	uint32_t	mHeight = 0;
};

Jpeg::Meta_t Jpeg::DecodeMeta(std::span<uint8_t> FileData)
{
	using namespace PopH264;
	FileReader_t Reader(FileData);
	
	//	check jpeg header
	{
		auto a = Reader.Read8();
		auto b = Reader.Read8();
		if ( a != 0xff || b != 0xd8 )
			throw std::runtime_error("Not a jpeg");
	}
	
	//	walk through blocks
	while ( Reader.RemainingBytes() > 0 )
	{
		auto BlockId = Reader.Read16Reverse();
		if ( (BlockId & 0xff00) != 0xff00 )
			throw std::runtime_error("Invalid block start");
		
		auto BlockLength = Reader.Read16Reverse();
		BlockLength -= 2;
	
		//	check JFIF header
		if ( BlockId == 0xffe0 )
		{
			Reader.ReadFourccReverse('JFIF');
			auto JfifTerm = Reader.Read8();
			if ( JfifTerm != 0 )
				throw std::runtime_error("Missing terminator after JFIF");
			//	read the rest of this block
			auto BlockData = Reader.ReadBytes( BlockLength - 5 );
		}
		else if ( BlockId == 0xffc0 )
		{
			auto Precision = Reader.Read8();
			Jpeg::Meta_t Meta;
			Meta.mHeight = Reader.Read16Reverse();
			Meta.mWidth = Reader.Read16Reverse();
			return Meta;
		}
		else
		{
			//	skip block
			auto BlockData = Reader.ReadBytes(BlockLength);
		}
	}

	throw std::runtime_error("Failed to extract meta from jpeg");
}



class Avf::TDecompressor
{
public:
	TDecompressor(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError);
	~TDecompressor();
	
	virtual void		Decode(PopH264::TInputNaluPacket& Packet)=0;
	void				Flush();

	//	public for delegate access
	void				OnDecodedFrame(OSStatus Status,CVImageBufferRef ImageBuffer,VTDecodeInfoFlags Flags,CMTime PresentationTimeStamp );
	void				OnDecodeError(std::string_view Error,CMTime PresentationTime);

protected:
	void				CreateDecoderSession(CFPtr<CMFormatDescriptionRef> InputFormat,H264::TSpsParams SpsParams);
	bool				HasSession();
	void				FreeSession();
	virtual void		DecodeSample(CFPtr<CMSampleBufferRef> FrameData,size_t FrameNumber);
	
	
	std::shared_ptr<AvfDecoderRenderer>	mDecoderRenderer;
	CFPtr<VTDecompressionSessionRef>	mSession;
	CFPtr<CMFormatDescriptionRef>		mInputFormat;
	std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)>		mOnFrame;
	std::function<void(const std::string&,PopH264::FrameNumber_t)>	mOnError;
	PopH264::TDecoderParams				mParams;
};


class Avf::TDecompressorH264 : public Avf::TDecompressor
{
public:
	TDecompressorH264(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError);
	
	virtual void			Decode(PopH264::TInputNaluPacket& Packet) override;

private:
	void					CreateSession();
	
	bool					mAllowSpsPpsToRecreateSession = true;	//	if true, then a new sps & pps appearing recreates session
	
	PopH264::FrameNumber_t	mLastFrameNumber = PopH264::FrameNumberInvalid;
	
	//	pending packets we need before we can create the session
	std::vector<uint8_t>	mNaluSps;
	std::vector<uint8_t>	mNaluPps;
	std::vector<uint8_t>	mNaluSei;
};


class Avf::TDecompressorJpeg : public Avf::TDecompressor
{
public:
	TDecompressorJpeg(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError);
	
	void					CreateDecoder(std::span<uint8_t> JpegData);
	virtual void			Decode(PopH264::TInputNaluPacket& Packet) override;
};


void OnDecompress(void* DecompressionContext,void* SourceContext,OSStatus Status,VTDecodeInfoFlags Flags,CVImageBufferRef ImageBuffer,CMTime PresentationTimeStamp,CMTime PresentationDuration)
{
	if ( !DecompressionContext )
	{
		std::Debug << "OnDecompress missing context" << std::endl;
		return;
	}
	
	auto& Decoder = *reinterpret_cast<Avf::TDecompressor*>( DecompressionContext );
	try
	{
		Decoder.OnDecodedFrame( Status, ImageBuffer, Flags, PresentationTimeStamp );
	}
	catch (std::exception& e)
	{
		Decoder.OnDecodeError( e.what(), PresentationTimeStamp );
	}
}

SoyPixelsMeta GetFormatDescriptionPixelMeta(CMFormatDescriptionRef Format)
{
	Boolean usePixelAspectRatio = false;
	Boolean useCleanAperture = false;
	auto Dim = CMVideoFormatDescriptionGetPresentationDimensions( Format, usePixelAspectRatio, useCleanAperture );
	
	return SoyPixelsMeta( Dim.width, Dim.height, SoyPixelsFormat::Invalid );
}


Avf::TDecompressor::TDecompressor(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError) :
	mDecoderRenderer	( new AvfDecoderRenderer() ),
	mOnFrame			( OnFrame ),
	mOnError			( OnError ),
	mParams				( Params )
{
}

Avf::TDecompressorH264::TDecompressorH264(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError) :
	TDecompressor	( Params, OnFrame, OnError )
{
	//	doesn't create a session until we've accumulate sps & pps
}


void Avf::TDecompressorH264::CreateSession()
{
	if ( mNaluSps.empty() )
		return;
	if ( mNaluPps.empty() )
		return;

	//	only create session if allowed
	if ( HasSession() )
	{
		if ( !mAllowSpsPpsToRecreateSession )
			return;
	}
	
	//	gr: don't let our bad sps decoding code stop decoding
	H264::TSpsParams SpsParams;
	try
	{
		//	gr: strip nalu prefix (ParseSps should do this!)
		auto SpsPrefixLength = H264::GetNaluLength( mNaluSps );
		auto SpsData = std::span(mNaluSps);
		SpsData = SpsData.subspan( SpsPrefixLength );
		SpsParams = H264::ParseSps( SpsData );
	}
	catch(std::exception& e)
	{
		std::Debug << "Warning: Failed to parse SPS before creating format; " << e.what() << std::endl;
	}
	
	
	auto InputFormat = Avf::GetFormatDescriptionH264( mNaluSps, mNaluPps, H264::NaluPrefix::ThirtyTwo, mParams.StripH264EmulationPrevention() );
	CreateDecoderSession( InputFormat, SpsParams );
	
	//	throw away the old sps/pps, so we can tell when we've got a new format
	mNaluSps.clear();
	mNaluPps.clear();
}

Avf::TDecompressorJpeg::TDecompressorJpeg(const PopH264::TDecoderParams& Params,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError) :
	TDecompressor	( Params, OnFrame, OnError )
{
	//	gr: we need the correct dimensions for the format or the decoder wont work
	//		so we unfortunetly need the first data before we can proceed
}


void Avf::TDecompressorJpeg::CreateDecoder(std::span<uint8_t> JpegData)
{
	CFAllocatorRef Allocator = nil;
	CMVideoCodecType Codec = kCMVideoCodecType_JPEG;
	
	//	width & height in the format NEEDS to be accurate or we'll get a decode error
	auto JpegMeta = Jpeg::DecodeMeta( JpegData );
	CFDictionaryRef Extensions = nullptr;
	//	refcounted on alloc, so we write to object
	CFPtr<CMFormatDescriptionRef> FormatDesc;
	auto Result = CMVideoFormatDescriptionCreate( Allocator, Codec, JpegMeta.mWidth, JpegMeta.mHeight, Extensions, &FormatDesc.mObject );
	Avf::IsOkay( Result, "CMVideoFormatDescriptionCreate jpeg");
	
	H264::TSpsParams ImageParams;
	ImageParams.mWidth = JpegMeta.mWidth;
	ImageParams.mHeight = JpegMeta.mHeight;

	CreateDecoderSession( FormatDesc, ImageParams );
}

bool Avf::TDecompressor::HasSession()
{
	return mSession.mObject != nullptr;
}

void Avf::TDecompressor::CreateDecoderSession(CFPtr<CMFormatDescriptionRef> InputFormat,H264::TSpsParams SpsParams)
{
	//	but skip if format is the same
	//	todo: may need to allow user to force this... or should inputformat be null'd when session is gone?
	static bool CheckFormatSame = true;
	
	if ( CheckFormatSame && mInputFormat )
	{
		//	gr: we can use VTDecompressionSessionCanAcceptFormatDescription() to retain the current session
		
		auto Same = CMFormatDescriptionEqual( mInputFormat.mObject, InputFormat.mObject );
		if ( Same )
		{
			if ( mParams.mVerboseDebug )
				std::Debug << "Skipping CreateDecoderSession() with duplicate input format" << std::endl;
			return;
		}
	}
		
	//	gr: we now allow recreation of sessions, so calling this function will always clear the old one
	//if ( HasSession() )
	//	return;
	FreeSession();
	
	std::cerr << "CreateDecoderSession with sps " << SpsParams.mWidth << "x" << SpsParams.mHeight << " profile=" << SpsParams.mProfile << " level=" << SpsParams.mLevel << std::endl;
	
	
	mInputFormat = InputFormat;
		
	CFAllocatorRef Allocator = nil;
	
	// Set the pixel attributes for the destination buffer
	CFMutableDictionaryRef destinationPixelBufferAttributes = CFDictionaryCreateMutable( Allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
	
	auto FormatPixelMeta = GetFormatDescriptionPixelMeta( mInputFormat.mObject );
	
	SInt32 Width = size_cast<SInt32>( FormatPixelMeta.GetWidth() );
	SInt32 Height = size_cast<SInt32>( FormatPixelMeta.GetHeight() );
	auto CroppedWidth = SpsParams.GetCroppedWidth();
	auto CroppedHeight = SpsParams.GetCroppedHeight();
	
	if ( Width != CroppedWidth || Height != CroppedHeight )
	{
		std::Debug << "Warning: Format width/height (" << Width << "x" << Height << ") doesn't match Sps cropped width/height " << CroppedWidth << "x" << CroppedHeight << "(uncropped " << SpsParams.mWidth << "x" << SpsParams.mHeight << ")" << std::endl;
	}
	//	gr: neither h264 nor jpeg seem to need this
	//CFDictionarySetValue(destinationPixelBufferAttributes,kCVPixelBufferWidthKey, CFNumberCreate(NULL, kCFNumberSInt32Type, &Width));
	//CFDictionarySetValue(destinationPixelBufferAttributes, kCVPixelBufferHeightKey, CFNumberCreate(NULL, kCFNumberSInt32Type, &Height));
	
	bool OpenglCompatible = false;
	auto ForceNonPlanarOutput = false;
	CFDictionarySetValue(destinationPixelBufferAttributes, kCVPixelBufferOpenGLCompatibilityKey, OpenglCompatible ? kCFBooleanTrue : kCFBooleanFalse );
	
	OSType destinationPixelType = 0;
	
	if ( ForceNonPlanarOutput )
	{
		destinationPixelType = kCVPixelFormatType_32BGRA;
	}
	else
	{
#if defined(TARGET_IOS)
		//	to get ios to use an opengl texture, we need to explicitly set the format to RGBA.
		//	None (auto) creates a non-iosurface compatible texture
		if ( OpenglCompatible )
		{
			//destinationPixelType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
			//destinationPixelType = kCVPixelFormatType_24RGB;
			destinationPixelType = kCVPixelFormatType_32BGRA;
		}
		else	//	for CPU copies we prefer bi-planar as it comes out faster and we merge in shader. though costs TWO texture uploads...
#endif
		{
			//	favour bi-plane so we can merge with shader rather than have OS do it in CPU
			destinationPixelType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
			//destinationPixelType = kCVPixelFormatType_24RGB;
		}
	}
	
	if ( destinationPixelType != 0 )
	{
		CFDictionarySetValue(destinationPixelBufferAttributes,kCVPixelBufferPixelFormatTypeKey, CFNumberCreate(NULL, kCFNumberSInt32Type, &destinationPixelType));
	}
	
	// Set the Decoder Parameters
	CFMutableDictionaryRef decoderParameters = CFDictionaryCreateMutable( Allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
	
	static bool AllowDroppedFrames = false;
	CFDictionarySetValue(decoderParameters,kVTDecompressionPropertyKey_RealTime, AllowDroppedFrames ? kCFBooleanTrue : kCFBooleanFalse );
	
	const VTDecompressionOutputCallbackRecord callback = { OnDecompress, this };
	
	auto Result = VTDecompressionSessionCreate(
											   Allocator,
											   mInputFormat.mObject,
											   decoderParameters,
											   destinationPixelBufferAttributes,
											   &callback,
											   &mSession.mObject );
	
	CFRelease(destinationPixelBufferAttributes);
	CFRelease(decoderParameters);
	
	Avf::IsOkay( Result, "TDecompressionSessionCreate" );
}

Avf::TDecompressor::~TDecompressor()
{
	try
	{
		FreeSession();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " FreeSession exception: " << e.what() << std::endl;
	}
	
}

void Avf::TDecompressor::FreeSession()
{
	if ( !mSession )
		return;
	
	try
	{
		Flush();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " Flush exception: " << e.what() << std::endl;
	}
	
	// End the session
	VTDecompressionSessionInvalidate( mSession.mObject );
	
	mSession.Release();
}

void Avf::TDecompressor::OnDecodedFrame(OSStatus Status,CVImageBufferRef ImageBuffer,VTDecodeInfoFlags Flags,CMTime PresentationTimeStamp)
{
	//	gr: the error -12349 is still undocumented, but we do continue to get frames after, seems to just be the first frame?
	IsOkay(Status,__PRETTY_FUNCTION__);
	
	//	gr: seem to need an extra retain. find out what's releaseing this twice despite retain below
	//auto RetainCount = CFGetRetainCount( ImageBuffer );
	//std::Debug << "On decoded frame, retain count=" << RetainCount << std::endl;
	//CFRetain( ImageBuffer );
	//RetainCount = CFGetRetainCount( ImageBuffer );
	
	//	todo: expand this to meta
	//	todo: this shouldn't be in CVPixelBuffer, and should be higherup as its part of the STREAM not the frame
	float3x3 Transform;

	//	gr: this is double deleting the cfptr somewhere
	bool Retain = true;
	std::shared_ptr<TPixelBuffer> PixelBuffer( new CVPixelBuffer(ImageBuffer, Retain, mDecoderRenderer, Transform ) );
	
	auto Time = Soy::Platform::GetTime(PresentationTimeStamp);
	PopH264::FrameNumber_t FrameNumber = size_cast<uint32_t>( Time.count() );
	mOnFrame( PixelBuffer, FrameNumber );
}

void Avf::TDecompressor::OnDecodeError(std::string_view Error,CMTime PresentationTimeStamp)
{
	if ( Error.empty() )
		Error = "<null>";
	std::Debug << __PRETTY_FUNCTION__ << Error << std::endl;
	std::string ErrorStr(Error);
	auto Time = Soy::Platform::GetTime(PresentationTimeStamp);
	PopH264::FrameNumber_t FrameNumber = size_cast<uint32_t>(Time.count());
	mOnError( ErrorStr, FrameNumber );
}

H264::NaluPrefix::Type GetNaluPrefixType(CMFormatDescriptionRef Format)
{
	//	need length-byte-size to get proper h264 format
	int nal_size_field_bytes = 0;
	auto Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( Format, 0, nullptr, nullptr, nullptr, &nal_size_field_bytes );
	Avf::IsOkay( Result, "Get H264 param NAL size");
	if ( nal_size_field_bytes < 0 )
		nal_size_field_bytes = 0;
	auto Type = static_cast<H264::NaluPrefix::Type>(nal_size_field_bytes);
	return Type;
}


CFPtr<CMSampleBufferRef> CreateSampleBuffer(std::span<uint8_t> DataArray,std::chrono::milliseconds PresentationTime,std::chrono::milliseconds DecodeTime,std::chrono::milliseconds DurationMs,CMFormatDescriptionRef Format)
{
	//	create buffer from packet
	CFAllocatorRef Allocator = nil;
	CFPtr<CMSampleBufferRef> SampleBuffer;
	
	uint32_t SubBlockSize = 0;
	CMBlockBufferFlags Flags = 0;
	CFPtr<CMBlockBufferRef> BlockBuffer;
	auto Result = CMBlockBufferCreateEmpty( Allocator, SubBlockSize, Flags, &BlockBuffer.mObject );
	Avf::IsOkay( Result, "CMBlockBufferCreateEmpty" );
		
	//	gr: when you pass memory to a block buffer, it only bloody frees it. make sure kCFAllocatorNull is the "allocator" for the data
	//		also means of course, for async decoding the data could go out of scope. May explain the wierd MACH__O error that came from the decoder?
	void* Data = DataArray.data();
	auto DataSize = DataArray.size();
	size_t Offset = 0;
	/*
	CMBlockBufferRef CM_NONNULL theBuffer,
	void * CM_NULLABLE memoryBlock,
	size_t blockLength,
	CFAllocatorRef CM_NULLABLE blockAllocator,
	const CMBlockBufferCustomBlockSource * CM_NULLABLE customBlockSource,
	size_t offsetToData,
	size_t dataLength,
	CMBlockBufferFlags flags)
	*/
	Result = CMBlockBufferAppendMemoryBlock( BlockBuffer.mObject,
											Data,
											DataSize,
											kCFAllocatorNull,
											nullptr,
											Offset,
											DataSize-Offset,
											Flags );
	Avf::IsOkay( Result, "CMBlockBufferAppendMemoryBlock" );
		
	
	/*
	//CMFormatDescriptionRef Format = GetFormatDescription( Packet.mMeta );
	//CMFormatDescriptionRef Format = Packet.mFormat->mDesc;
	auto Format = Packet.mFormat ? Packet.mFormat->mDesc : mFormatDesc->mDesc;
	if ( !VTDecompressionSessionCanAcceptFormatDescription( mSession->mSession, Format ) )
	{
		std::Debug << "VTDecompressionSessionCanAcceptFormatDescription failed" << std::endl;
		//	gr: maybe re-create session here with... new format? (save the packet's format to mFormatDesc?)
		//bool Dummy;
		//mOnStreamChanged.OnTriggered( Dummy );
	}
	*/
		
	int NumSamples = 1;
	BufferArray<size_t,1> SampleSizes;
	SampleSizes.PushBack( DataSize );
	BufferArray<CMSampleTimingInfo,1> SampleTimings;
	auto& FrameTiming = SampleTimings.PushBack();
	FrameTiming.duration = Soy::Platform::GetTime( DurationMs );
	FrameTiming.presentationTimeStamp = Soy::Platform::GetTime( PresentationTime );
	FrameTiming.decodeTimeStamp = Soy::Platform::GetTime( DecodeTime );
		
	Result = CMSampleBufferCreate(	Allocator,
								  BlockBuffer.mObject,
								  true,
								  nullptr,	//	callback
								  nullptr,	//	callback context
								  Format,
								  NumSamples,
								  SampleTimings.GetSize(),
								  SampleTimings.GetArray(),
								  SampleSizes.GetSize(),
								  SampleSizes.GetArray(),
								  &SampleBuffer.mObject );
	Avf::IsOkay( Result, "CMSampleBufferCreate" );

	
	//	sample buffer now has a reference to the block, so we dont want it
	//	gr: should now auto release
	//CFRelease( BlockBuffer );
	
	return SampleBuffer;
}


void Avf::TDecompressorJpeg::Decode(PopH264::TInputNaluPacket& Packet)
{
	//	ignore EOF packet
	if ( Packet.mContentType == ContentType::EndOfFile )
		return;
	
	auto JpegData = std::span( Packet.mData.data(), Packet.mData.size() );

	//	setup session once we have jpeg data
	CreateDecoder( JpegData );

	//	gr: all jpeg samples need to have frame number 0
	//uint64_t FrameNumber = 99;
	uint64_t FrameNumber = Packet.mFrameNumber;
	
	std::chrono::milliseconds PresentationTime(FrameNumber);
	std::chrono::milliseconds DecodeTime(FrameNumber);
	std::chrono::milliseconds Duration(16ull);
	auto SampleBuffer = CreateSampleBuffer( JpegData, PresentationTime, DecodeTime, Duration, mInputFormat.mObject );
	
	DecodeSample( SampleBuffer, FrameNumber );
}


void Avf::TDecompressorH264::Decode(PopH264::TInputNaluPacket& Packet)
{
	bool EndOfStream = false;
	if ( Packet.mContentType == ContentType::EndOfFile )
		EndOfStream = true;

	H264NaluContent::Type H264PacketType = H264NaluContent::Invalid;
	{
		auto Nalu = Packet.GetData();
		if ( Nalu.size() > 0 )
		{
			H264PacketType = H264::GetPacketType(Nalu);
		}
	}
	if ( H264PacketType == H264NaluContent::EndOfStream )
		EndOfStream = true;
	
	if ( EndOfStream )
	{
		//	synchronous flush
		Flush();
		return;
	}

	if ( mParams.mVerboseDebug )
		std::Debug << "Popped Nalu " << H264PacketType << " x" << Packet.GetData().size() << "bytes" << std::endl;

	//	some packets the avf decoder will error on, never decode them.
	bool DecodePacket = true;
	
	//	do not push SPS, PPS or SEI packets to decoder
	if ( H264PacketType == H264NaluContent::SequenceParameterSet )
	{
		mNaluSps = Packet.mData;
		DecodePacket = false;
	}
	else if ( H264PacketType == H264NaluContent::PictureParameterSet )
	{
		mNaluPps = Packet.mData;
		DecodePacket = false;
	}
	else if ( H264PacketType == H264NaluContent::SupplimentalEnhancementInformation )
	{
		//	SEI gives -12349 error
		if ( !mParams.mDecodeSei )
		{
			mNaluSei = Packet.mData;
			DecodePacket = false;
			return;
		}
	}

	//	try and create the session in case we've got the SPS & PPS we need
	CreateSession();

	//	dont need the warning below if we were going to drop it anyway
	if ( !DecodePacket )
		return;
	
	//	no decompression session yet, drop packet
	//	could be packet before SPS/PPS and we ignore it
	if ( !HasSession() )
	{
		if ( mParams.mVerboseDebug )
			std::Debug << "Dropping H264 frame (" << magic_enum::enum_name(H264PacketType) << ") as decompressor isn't ready (waiting for sps/pps)" << std::endl;
		return;
	}

	auto RequiredNaluFormat = Avf::GetFormatInputNaluPrefix(mInputFormat.mObject);
	//auto NaluSize = GetFormatNaluPrefixType();
	H264::ConvertNaluPrefix( Packet.mData, RequiredNaluFormat );
	
	uint64_t FrameNumber = Packet.mFrameNumber;
	
	static bool AllowDuplicateFrameNumbers = true;
	
	if ( !AllowDuplicateFrameNumbers )
	{
		//	if the frame number is the same, we're going to have trouble resolving which frame is which
		//	auto-increment duplicates, then error if user passes an old frame
		//	gr: move this to generic frame handling
		if ( FrameNumber == mLastFrameNumber )
		{
			FrameNumber = mLastFrameNumber+1;
			std::Debug << "Warning: duplicate frame number input " << mLastFrameNumber << ", auto-incrementing to " << FrameNumber << std::endl;
		}
	}
	//	gr: does this need to fail?
	if ( FrameNumber < mLastFrameNumber )
	{
		std::Debug << "Warning: next frame number (" << FrameNumber << ") in the past, last frame number=" << mLastFrameNumber << std::endl;
		//FrameNumber = mLastFrameNumber+1;
	}

	std::chrono::milliseconds PresentationTime(FrameNumber);
	std::chrono::milliseconds DecodeTime(FrameNumber);
	std::chrono::milliseconds Duration(16ull);
	auto PacketData = Packet.GetData();
	auto SampleBuffer = CreateSampleBuffer( PacketData, PresentationTime, DecodeTime, Duration, mInputFormat.mObject );
	
	std::Debug << "Decode " << magic_enum::enum_name(H264PacketType) << " frame=" << FrameNumber << "..." << std::endl;
	DecodeSample(SampleBuffer, FrameNumber);
	
	mLastFrameNumber = FrameNumber;
}


void Avf::TDecompressor::DecodeSample(CFPtr<CMSampleBufferRef> SampleBuffer,size_t FrameNumber)
{
	VTDecodeFrameFlags Flags = 0;
	VTDecodeInfoFlags FlagsOut = 0;
	
	//	gr: temporal means frames (may?) will be output in display order, OS will hold onto decoded frames
	bool OutputFramesInOrder = mParams.mAllowBuffering;
	if ( OutputFramesInOrder )
		Flags |= kVTDecodeFrame_EnableTemporalProcessing;
	
	//	gr: async means frames may or may not be decoded in the background
	//	gr: also we may have issues with sample buffer lifetime in async
	bool AsyncDecompression = mParams.mAsyncDecompression;
	if ( AsyncDecompression )
		Flags |= kVTDecodeFrame_EnableAsynchronousDecompression;
	
	//	1x/low power mode means it WONT try and decode faster than 1x
	bool LowPowerDecoding = mParams.mLowPowerMode;
	if ( LowPowerDecoding )
		Flags |= kVTDecodeFrame_1xRealTimePlayback;
	
	SoyTime DecodeDuration;
	auto OnFinished = [&DecodeDuration](SoyTime Timer)
	{
		DecodeDuration = Timer;
	};
	
	bool RecreateStream = false;
	{
		//	gr: here we could detect wrong nalu input
		//	gr: this will fail for jpeg though, so should be somewhere else
		//auto RequiredNaluFormat = Avf::GetFormatInputNaluPrefix( mInputFormat.mObject );
		
		//std::Debug << "decompressing " << Packet.mTimecode << "..." << std::endl;
		//Soy::TScopeTimer Timer("VTDecompressionSessionDecodeFrame", 0, OnFinished, true );
		Soy::TScopeTimerPrint Timer("VTDecompressionSessionDecodeFrame", 10 );
		auto Result = VTDecompressionSessionDecodeFrame( mSession.mObject, SampleBuffer.mObject, Flags, nullptr, &FlagsOut );
		Timer.Stop();
		//std::Debug << "Decompress " << Packet.mTimecode << " took " << DecodeDuration << "; error=" << (int)Result << std::endl;
		Avf::IsOkay( Result, "VTDecompressionSessionDecodeFrame" );
		
		static int FakeInvalidateSessionCounter = 0;
		static int FakeInvalidateSessionOnCount = -1;
		if ( ++FakeInvalidateSessionCounter == FakeInvalidateSessionOnCount )
		{
			FakeInvalidateSessionCounter = 0;
			Result = kVTInvalidSessionErr;
		}
		
		switch ( Result )
		{
				//	no error
			case 0:
				break;
				
				//	gr: if we pause for ages without eating a frame, we can get this...
				//		because for somereason the decoder thread is still trying to decode stuff??
			case MACH_RCV_TIMED_OUT:
				std::Debug << "Decompression MACH_RCV_TIMED_OUT..." << std::endl;
				break;
				
				//	gr: restoring iphone app sometimes gives us malfunction, sometimes invalid session.
				//		guessing invalid session is if it's been put to sleep properly or OS has removed some resources
			case kVTInvalidSessionErr:
			case kVTVideoDecoderMalfunctionErr:
			{
				//  gr: need to re-create session. Session dies when app sleeps and restores
				std::stringstream Error;
				Error << "Lost decompression session; " << Avf::GetString(Result);
				//OnDecodeError( Error.str(), Packet.mTimecode );
				//	make errors visible for debugging
				//std::this_thread::sleep_for( std::chrono::milliseconds(1000));
				RecreateStream = true;
			}
				break;
				
				
			default:
			{
				static bool RecreateOnDecompressionError = false;
				std::Debug << "some decompression error; " << Avf::GetString(Result) << std::endl;
				if ( RecreateOnDecompressionError )
					RecreateStream = true;
			}
				break;
		}
		
		//	gr: do we NEED to make sure all referecnes are GONE here? as the data in block buffer is still in use if >1?
		//auto SampleCount = CFGetRetainCount( SampleBuffer );
		//CFRelease( SampleBuffer );
		
		
		//	gr: hanging on destruction waiting for async frames, see if this makes it go away.
		if ( bool_cast(Flags & kVTDecodeFrame_EnableAsynchronousDecompression) )
			VTDecompressionSessionWaitForAsynchronousFrames( mSession.mObject );
	}
	
	//	recover from errors
	if ( RecreateStream )
	{
		std::Debug << "recreate decoder!" << std::endl;
	}
}

void Avf::TDecompressor::Flush()
{
	if ( !mSession )
		return;
	
	auto Error = VTDecompressionSessionFinishDelayedFrames(mSession.mObject);
	IsOkay(Error,"VTDecompressionSessionFinishDelayedFrames");
	
	Error = VTDecompressionSessionWaitForAsynchronousFrames(mSession.mObject);
	IsOkay(Error,"VTDecompressionSessionWaitForAsynchronousFrames");
}

	
Avf::TDecoder::TDecoder(const PopH264::TDecoderParams& Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	PopH264::TDecoder	( Params, OnDecodedFrame, OnFrameError )
{
}

Avf::TDecoder::~TDecoder()
{
	mDecompressor.reset();
}

void Avf::TDecoder::OnDecodedFrame(TPixelBuffer& PixelBuffer,PopH264::FrameNumber_t FrameNumber,const json11::Json& Meta)
{
	BufferArray<SoyPixelsImpl*,4> Planes;
	float3x3 Transform;
	PixelBuffer.Lock( GetArrayBridge(Planes), Transform );
	try
	{
		if ( Planes.GetSize() == 0 )
		{
			throw Soy::AssertException("No planes from pixel buffer");
		}
		else if ( Planes.GetSize() == 1 )
		{
			OnDecodedFrame( *Planes[0], FrameNumber, Meta );
		}
		else
		{
			//	merge planes
			SoyPixels Merged( *Planes[0] );

			//	gr: we have a problem here where we can have an image like
			//	height = 1001
			//	luma height = 1001
			//	but chroma plane (half height) comes out as 501
			auto LumaHeight = Planes[0]->GetHeight();
			auto DoubleChromaHeight = Planes[1]->GetHeight() * 2;
			if ( LumaHeight < DoubleChromaHeight )
				Merged.ResizeClip( Merged.GetWidth(), DoubleChromaHeight );

			if ( Planes.GetSize() == 2 )
			{
				Merged.AppendPlane(*Planes[1]);
			}
			else
			{
				Merged.AppendPlane(*Planes[1],*Planes[2]);
			}
			OnDecodedFrame( Merged, FrameNumber, Meta );
		}
		
		PixelBuffer.Unlock();
	}
	catch(...)
	{
		PixelBuffer.Unlock();
		throw;
	}
}


bool Avf::TDecoder::DecodeNextPacket()
{
	auto pNextPacket = PopNextPacket();
	if ( !pNextPacket )
		return false;
	auto& NextPacket = *pNextPacket;
	
	//	need to create a decompressor
	if ( !mDecompressor )
	{
		auto OnFrame = [this](std::shared_ptr<TPixelBuffer> pPixelBuffer,PopH264::FrameNumber_t FrameNumber)
		{
			//std::Debug << "Decompressed pixel buffer " << PresentationTime << std::endl;
			json11::Json::object Meta;
			this->OnDecodedFrame( *pPixelBuffer, FrameNumber, Meta );
		};
		
		auto OnError = [this](const std::string& Error,PopH264::FrameNumber_t FrameNumber)
		{
			this->OnFrameError(Error,FrameNumber);
		};
		
		if ( NextPacket.mContentType == ContentType::Jpeg )
		{
			mDecompressor.reset( new TDecompressorJpeg( mParams, OnFrame, OnError ) );
		}
		else
		{
			mDecompressor.reset( new TDecompressorH264( mParams, OnFrame, OnError ) );
		}
	}
	
	mDecompressor->Decode( NextPacket );
	
	if ( NextPacket.mContentType == ContentType::EndOfFile )
	{
		OnDecodedEndOfStream();
	}

	return true;
}
