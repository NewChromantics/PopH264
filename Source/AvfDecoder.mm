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



class Avf::TDecompressor
{
public:
	TDecompressor(const PopH264::TDecoderParams& Params,const ArrayBridge<uint8_t>& Sps,const ArrayBridge<uint8_t>& Pps,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError);
	~TDecompressor();
	
	void								Decode(ArrayBridge<uint8_t>&& Nalu,size_t FrameNumber);
	void								Flush();
	H264::NaluPrefix::Type				GetFormatNaluPrefixType()	{	return H264::NaluPrefix::ThirtyTwo;	}
	
	void								OnDecodedFrame(OSStatus Status,CVImageBufferRef ImageBuffer,VTDecodeInfoFlags Flags,CMTime PresentationTimeStamp );
	void								OnDecodeError(const char* Error,CMTime PresentationTime);

	std::shared_ptr<AvfDecoderRenderer>	mDecoderRenderer;
	CFPtr<VTDecompressionSessionRef>	mSession;
	CFPtr<CMFormatDescriptionRef>		mInputFormat;
	std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)>		mOnFrame;
	std::function<void(const std::string&,PopH264::FrameNumber_t)>	mOnError;
	PopH264::TDecoderParams				mParams;
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


Avf::TDecompressor::TDecompressor(const PopH264::TDecoderParams& Params,const ArrayBridge<uint8_t>& Sps,const ArrayBridge<uint8_t>& Pps,std::function<void(std::shared_ptr<TPixelBuffer>,PopH264::FrameNumber_t)> OnFrame,std::function<void(const std::string&,PopH264::FrameNumber_t)> OnError) :
	mDecoderRenderer	( new AvfDecoderRenderer() ),
	mOnFrame			( OnFrame ),
	mOnError			( OnError ),
	mParams				( Params )
{
	mInputFormat = Avf::GetFormatDescriptionH264( Sps, Pps, GetFormatNaluPrefixType() );
		
	CFAllocatorRef Allocator = nil;
	
	// Set the pixel attributes for the destination buffer
	CFMutableDictionaryRef destinationPixelBufferAttributes = CFDictionaryCreateMutable( Allocator, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
	
	auto FormatPixelMeta = GetFormatDescriptionPixelMeta( mInputFormat.mObject );
	
	SInt32 Width = size_cast<SInt32>( FormatPixelMeta.GetWidth() );
	SInt32 Height = size_cast<SInt32>( FormatPixelMeta.GetHeight() );
	
	CFDictionarySetValue(destinationPixelBufferAttributes,kCVPixelBufferWidthKey, CFNumberCreate(NULL, kCFNumberSInt32Type, &Width));
	CFDictionarySetValue(destinationPixelBufferAttributes, kCVPixelBufferHeightKey, CFNumberCreate(NULL, kCFNumberSInt32Type, &Height));
	
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
		Flush();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " flush exception: " << e.what() << std::endl;
	}
	
	// End the session
	VTDecompressionSessionInvalidate( mSession.mObject );
	
	try
	{
		mSession.Release();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " mSession.Release exception: " << e.what() << std::endl;
	}
}

void Avf::TDecompressor::OnDecodedFrame(OSStatus Status,CVImageBufferRef ImageBuffer,VTDecodeInfoFlags Flags,CMTime PresentationTimeStamp)
{
	//	gr: the error -12349 is still undocumented, but we do continue to get frames after, seems to just be the first frame?
	IsOkay(Status,__PRETTY_FUNCTION__);
	
	//	gr: seem to need an extra retain. find out what's releaseing this twice despite retain below
	auto RetainCount = CFGetRetainCount( ImageBuffer );
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
	PopH264::FrameNumber_t FrameNumber = Time.mTime;
	mOnFrame( PixelBuffer, FrameNumber );
}

void Avf::TDecompressor::OnDecodeError(const char* Error,CMTime PresentationTimeStamp)
{
	if ( Error == nullptr )
		Error = "<null>";
	std::Debug << __PRETTY_FUNCTION__ << Error << std::endl;
	std::string ErrorStr(Error);
	auto Time = Soy::Platform::GetTime(PresentationTimeStamp);
	PopH264::FrameNumber_t FrameNumber = Time.mTime;
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


CFPtr<CMSampleBufferRef> CreateSampleBuffer(ArrayBridge<uint8_t>& DataArray,SoyTime PresentationTime,SoyTime DecodeTime,SoyTime DurationMs,CMFormatDescriptionRef Format)
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
	void* Data = DataArray.GetArray();
	auto DataSize = DataArray.GetDataSize();
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

void Avf::TDecompressor::Decode(ArrayBridge<uint8_t>&& Nalu, size_t FrameNumber)
{
	//	if we get an endofstream packet, do a flush
	{
		auto H264PacketType = H264::GetPacketType(GetArrayBridge(Nalu));
		if ( H264PacketType == H264NaluContent::EndOfStream )
		{
			//	synchronous flush
			Flush();
			return;
		}
	}
	
	auto NaluSize = GetFormatNaluPrefixType();
	H264::ConvertNaluPrefix( Nalu, NaluSize );
	
	SoyTime PresentationTime( static_cast<uint64_t>(FrameNumber) );
	SoyTime DecodeTime( static_cast<uint64_t>(FrameNumber) );
	SoyTime Duration(16ull);
	auto SampleBuffer = CreateSampleBuffer( Nalu, PresentationTime, DecodeTime, Duration, mInputFormat.mObject );
	
	
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
		//std::Debug << "decompressing " << Packet.mTimecode << "..." << std::endl;
		Soy::TScopeTimer Timer("VTDecompressionSessionDecodeFrame", 1, OnFinished, true );
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
			if ( Planes.GetSize() == 2 )
				Merged.AppendPlane(*Planes[1]);
			else
				Merged.AppendPlane(*Planes[1],*Planes[2]);
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


void Avf::TDecoder::AllocDecoder()
{
	auto OnPacket = [this](std::shared_ptr<TPixelBuffer> pPixelBuffer,PopH264::FrameNumber_t FrameNumber)
	{
		//std::Debug << "Decompressed pixel buffer " << PresentationTime << std::endl;
		json11::Json::object Meta;
		this->OnDecodedFrame( *pPixelBuffer, FrameNumber, Meta );
	};
	
	auto OnError = [this](const std::string& Error,PopH264::FrameNumber_t FrameNumber)
	{
		this->OnFrameError(Error,FrameNumber);
	};

	if ( mDecompressor )
		return;
	
	if ( mNaluSps.IsEmpty() || mNaluPps.IsEmpty() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " waiting for " << (mNaluSps.IsEmpty()?"SPS":"") << " " << (mNaluPps.IsEmpty()?"PPS":"") << std::endl;
		return;
	}
	
	//	gr: does decompressor need to wait for SPS&PPS?
	mDecompressor.reset( new TDecompressor( mParams, GetArrayBridge(mNaluSps), GetArrayBridge(mNaluPps), OnPacket, OnError ) );
}

bool Avf::TDecoder::DecodeNextPacket()
{
	Array<uint8_t> Nalu;
	PopH264::FrameNumber_t FrameNumber=0;
	if ( !PopNalu( GetArrayBridge(Nalu), FrameNumber ) )
		return false;

	//	store latest sps & pps, need to cache these so we can create decoder
	auto H264PacketType = H264::GetPacketType(GetArrayBridge(Nalu));
	if ( mParams.mVerboseDebug )
		std::Debug << "Popped Nalu " << H264PacketType << " x" << Nalu.GetDataSize() << "bytes" << std::endl;

	//	do not push SPS, PPS or SEI packets to decoder
	//	SEI gives -12349 error
	if ( H264PacketType == H264NaluContent::SequenceParameterSet )
	{
		mNaluSps = Nalu;
		return true;
	}
	else if ( H264PacketType == H264NaluContent::PictureParameterSet )
	{
		mNaluPps = Nalu;
		return true;
	}
	else if ( H264PacketType == H264NaluContent::SupplimentalEnhancementInformation )
	{
		if ( !mParams.mDecodeSei )
		{
			mNaluSei = Nalu;
			return true;
		}
	}

	//	make sure we have a decoder
	AllocDecoder();
	
	//	no decompressor yet, drop packet
	if ( !mDecompressor )
	{
		if ( mParams.mVerboseDebug )
			std::Debug << "Dropping H264 frame (" << magic_enum::enum_name(H264PacketType) << ") as decompressor isn't ready (waiting for sps/pps)" << std::endl;
		return true;
	}
	
	mDecompressor->Decode( GetArrayBridge(Nalu), FrameNumber );
	
	//	if this was an end of stream packet, the decompressor should have flushed, so now
	//	queue up an EndOfStream packet
	//	todo: see if avf is marking a last packet as EOS
	if ( H264PacketType == H264NaluContent::EndOfStream )
	{
		OnDecodedEndOfStream();
	}
	
	return true;
}
