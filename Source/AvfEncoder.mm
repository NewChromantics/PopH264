#include "AvfEncoder.h"
#include "SoyPixels.h"
#include "SoyAvf.h"
#include "SoyFourcc.h"
#include "MagicEnum/include/magic_enum.hpp"

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


class Avf::TCompressor
{
public:
	TCompressor(const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&&,size_t)> OnPacket);
	~TCompressor();
	
	void	OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer);
	void	Flush();

	void	Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber);
	
private:
	void	OnPacket(const ArrayBridge<uint8_t>&& Data,SoyTime PresentationTime);
	
private:
	std::function<void(const ArrayBridge<uint8_t>&&,size_t)>	mOnPacket;

	
	VTCompressionSessionRef EncodingSession = nil;
	dispatch_queue_t aQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
	CMFormatDescriptionRef  format;
	CMSampleTimingInfo * timingInfo;
	BOOL initialized;
	int  frameCount = 0;
	NSData *sps = nullptr;
	NSData *pps = nullptr;
};

void OnCompressedCallback(void *outputCallbackRefCon,void *sourceFrameRefCon, OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer)
{
	//	https://chromium.googlesource.com/external/webrtc/+/6c78307a21252c2dbd704f6d5e92a220fb722ed4/webrtc/modules/video_coding/codecs/h264/h264_video_toolbox_encoder.mm#588
	try
	{
		auto* This = static_cast<Avf::TCompressor*>(outputCallbackRefCon);
		This->OnCompressed(status,infoFlags,sampleBuffer);
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception with OnCompressed callback; " << e.what() << std::endl;
	}
}



Avf::TCompressor::TCompressor(const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&&,size_t)> OnPacket) :
	mOnPacket	( OnPacket )
{
	if ( !mOnPacket )
		throw Soy::AssertException("OnPacket callback missing in Avf::TCompressor");
	
	//h264Encoder = [H264HwEncoderImpl alloc];
	//	[h264Encoder initWithConfiguration];
	auto Lambda = ^
	{
		CFMutableDictionaryRef sessionAttributes = CFDictionaryCreateMutable( NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
		
		// bitrate 只有当压缩frame设置的时候才起作用，有时候不起作用，当不设置的时候大小根据视频的大小而定
		//        int fixedBitrate = 2000 * 1024; // 2000 * 1024 -> assume 2 Mbits/s
		//        CFNumberRef bitrateNum = CFNumberCreate(NULL, kCFNumberSInt32Type, &fixedBitrate);
		//        CFDictionarySetValue(sessionAttributes, kVTCompressionPropertyKey_AverageBitRate, bitrateNum);
		//        CFRelease(bitrateNum);
		
		// CMTime CMTimeMake(int64_t value,	 int32_t timescale)当timescale设置为1的时候更改这个参数就看不到效果了
		//        float fixedQuality = 1.0;
		//        CFNumberRef qualityNum = CFNumberCreate(NULL, kCFNumberFloat32Type, &fixedQuality);
		//        CFDictionarySetValue(sessionAttributes, kVTCompressionPropertyKey_Quality, qualityNum);
		//        CFRelease(qualityNum);
		
		//貌似没作用
		//        int DataRateLimits = 2;
		//        CFNumberRef DataRateLimitsNum = CFNumberCreate(NULL, kCFNumberSInt8Type, &DataRateLimits);
		//        CFDictionarySetValue(sessionAttributes, kVTCompressionPropertyKey_DataRateLimits, DataRateLimitsNum);
		//        CFRelease(DataRateLimitsNum);
		
		// 创建编码
		//(__bridge void*) CallbackParam = this;
		void* CallbackParam = this;
		auto Width = Meta.GetWidth();
		auto Height = Meta.GetHeight();
		OSStatus status = VTCompressionSessionCreate( NULL, Width, Height, kCMVideoCodecType_H264, sessionAttributes, NULL, NULL, OnCompressedCallback, CallbackParam, &EncodingSession );
		//std::Debug << "H264: VTCompressionSessionCreate " << status << std::endl;
		Avf::IsOkay(status,"VTCompressionSessionCreate");

		//	gr: kVTProfileLevel_H264_Baseline_3_0 always fails in compression callback with -12348
		auto Profile = kVTProfileLevel_H264_Baseline_3_1;
		//kVTProfileLevel_H264_High_5_2
		auto Realtime = kCFBooleanTrue;
		auto AllowFramesOutOfOrder = false;
		auto AutoOrderFrames = AllowFramesOutOfOrder ? kCFBooleanTrue : kCFBooleanFalse;
		status = VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_RealTime, Realtime);
		Avf::IsOkay(status,"kVTCompressionPropertyKey_RealTime");
		status = VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_ProfileLevel, Profile);
		Avf::IsOkay(status,"kVTCompressionPropertyKey_ProfileLevel");
		status = VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_AllowFrameReordering, AutoOrderFrames);
		Avf::IsOkay(status,"kVTCompressionPropertyKey_AllowFrameReordering");

		// 启动编码
		//	control quality
		auto AverageKbRate = 512;//1024 * 1;
		auto AverageBitRate = AverageKbRate * 8;
		CFNumberRef AverageBitRateNumber = CFNumberCreate(NULL, kCFNumberSInt32Type, &AverageBitRate);
		status = VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_AverageBitRate, AverageBitRateNumber);
		Avf::IsOkay(status,"kVTCompressionPropertyKey_AverageBitRate");
	
		status = VTCompressionSessionPrepareToEncodeFrames(EncodingSession);
		Avf::IsOkay(status,"VTCompressionSessionPrepareToEncodeFrames");
	};
	dispatch_sync(aQueue, Lambda);
}

Avf::TCompressor::~TCompressor()
{
	Flush();
	
	// End the session
	VTCompressionSessionInvalidate( EncodingSession );
	CFRelease( EncodingSession );
	
	//	wait for the queue to end
}

void Avf::TCompressor::Flush()
{
	VTCompressionSessionCompleteFrames( EncodingSession, kCMTimeInvalid );
}


void AnnexBToAnnexB(const ArrayBridge<uint8_t>& Data,std::function<void(const ArrayBridge<uint8_t>&&)> EnumPacket)
{
	//	gr: does this start with 0001 etc? if so, cut
	Soy_AssertTodo();
	EnumPacket( GetArrayBridge(Data) );
}

void NaluToAnnexB(const ArrayBridge<uint8_t>& Data,size_t LengthSize,std::function<void(const ArrayBridge<uint8_t>&&)>& EnumPacket)
{
	//	walk through data
	int i=0;
	while ( i <Data.GetSize() )
	{
		size_t ChunkLength = 0;
		auto* pData = &Data[i+0];
		
		if ( LengthSize == 1 )
		{
			ChunkLength |= Data[i+0];
		}
		else if ( LengthSize == 2 )
		{
			ChunkLength |= Data[i+0] << 8;
			ChunkLength |= Data[i+1] << 0;
		}
		else if ( LengthSize == 4 )
		{
			//	gr: we should be using CFSwapInt32BigToHost
			ChunkLength |= Data[i+0] << 24;
			ChunkLength |= Data[i+1] << 16;
			ChunkLength |= Data[i+2] << 8;
			ChunkLength |= Data[i+3] << 0;
		}
	
		auto* DataStart = &Data[i+LengthSize];
		auto PacketContent = GetRemoteArray( DataStart, ChunkLength );

		EnumPacket( GetArrayBridge(PacketContent) );
		
		i += LengthSize + ChunkLength;
	}
}


//	this could be multiple nals, and we need to cut the prefix, so enum
extern "C" void ExtractPackets(const ArrayBridge<uint8_t>&& Packets,CMFormatDescriptionRef FormatDescription,std::function<void(const ArrayBridge<uint8_t>&&)> EnumPacket)
{
	int nal_size_field_bytes = 0;
	//	SPS & PPS (&sei?) set count, maybe we should integrate that into this func
	size_t ParamSetCount = 0;
	auto Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( FormatDescription, 0, nullptr, nullptr, &ParamSetCount, &nal_size_field_bytes );
	Avf::IsOkay( Result, "Get H264 param NAL size");
	
	//	extract header packets
	//	SPS, then PPS
	H264NaluContent::Type NaluContentTypes[] = { H264NaluContent::SequenceParameterSet, H264NaluContent::PictureParameterSet };
	for ( auto i=0;	i<ParamSetCount;	i++ )
	{
		if ( i > 1 )
			throw Soy::AssertException("Got Packet header > SPS & PPS");
		Array<uint8_t> SpsData;
		Avf::GetFormatDescriptionData( GetArrayBridge(SpsData), FormatDescription, i );

		//	gr: this header is already here. lets debug it in EnumPacket though
		//	insert nalu header
		//auto Content = NaluContentTypes[i];
		//auto Priority = H264NaluPriority::Important;
		//auto NaluByte = H264::EncodeNaluByte(Content,Priority);
		EnumPacket( GetArrayBridge(SpsData) );
	}
	
	
	//	-1 is annexB
	if ( nal_size_field_bytes < 0 )
		nal_size_field_bytes = 0;
	
	switch ( nal_size_field_bytes )
	{
		case 0:
			AnnexBToAnnexB( Packets, EnumPacket );
			return;
			
		//case 1:
		//case 2:
		case 4:
			NaluToAnnexB( Packets, nal_size_field_bytes, EnumPacket );
			return;
	}
	
	std::stringstream Debug;
	Debug << "Unhandled nal_size_field_bytes " << nal_size_field_bytes;
	throw Soy::AssertException(Debug);
}


void Avf::TCompressor::OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef SampleBuffer)
{
	Avf::IsOkay( status, "OnCompressed status");
	
	//	if flags & dropped, report
	if ( status != 0 || infoFlags != 0 )
	{
		std::Debug << __PRETTY_FUNCTION__ << "( status=" << status << " infoFlags=" << infoFlags << ")" << std::endl;
	}
	
	CMFormatDescriptionRef FormatDescription = CMSampleBufferGetFormatDescription(SampleBuffer);

	auto DescFourcc = CFSwapInt32HostToBig( CMFormatDescriptionGetMediaSubType(FormatDescription) );
	Soy::TFourcc Fourcc( DescFourcc );

	//	get meta
	CMTime PresentationTimestamp = CMSampleBufferGetPresentationTimeStamp(SampleBuffer);
	CMTime DecodeTimestamp = CMSampleBufferGetDecodeTimeStamp(SampleBuffer);
	CMTime SampleDuration = CMSampleBufferGetDuration(SampleBuffer);
	auto PresentationTime = Soy::Platform::GetTime(PresentationTimestamp);
	auto DecodeTimecode = Soy::Platform::GetTime(DecodeTimestamp);
	auto Duration = Soy::Platform::GetTime(SampleDuration);
	
	//	look for SPS & PPS data if we have a keyframe
	CFDictionaryRef Dictionary = static_cast<CFDictionaryRef>( CFArrayGetValueAtIndex(CMSampleBufferGetSampleAttachmentsArray(SampleBuffer, true), 0) );
	bool IsKeyframe = !CFDictionaryContainsKey( Dictionary, kCMSampleAttachmentKey_NotSync);

	//	doing this check after getting meta to help debug
	if (!CMSampleBufferDataIsReady(SampleBuffer))
	{
		throw Soy::AssertException("Data sample not ready");
	}

	//	extract data
	{
		//	this data could be an image buffer or a block buffer (for h264, expecting block)
		CMBlockBufferRef BlockBuffer = CMSampleBufferGetDataBuffer( SampleBuffer );
		//CVImageBufferRef ImageBuffer = CMSampleBufferGetImageBuffer( SampleBuffer );
		
		if ( BlockBuffer )
		{
			//	copy bytes into our array
			//	CMBlockBufferGetDataPointer is also an option...
			//	...but we copy in case buffer isn't contiguous
			//if (!CMBlockBufferIsRangeContiguous(block_buffer, 0, 0)) {
			Array<uint8_t> PacketData;
			
			auto DataSize = CMBlockBufferGetDataLength( BlockBuffer );
			PacketData.SetSize( DataSize );
			auto Result = CMBlockBufferCopyDataBytes( BlockBuffer, 0, PacketData.GetDataSize(), PacketData.GetArray() );
			Avf::IsOkay( Result, "CMBlockBufferCopyDataBytes" );
			
			auto EnumPacket = [&](const ArrayBridge<uint8_t>&& PacketData)
			{
				OnPacket( GetArrayBridge(PacketData), PresentationTime );
			};
			//	this could be multiple nals, and we need to cut the prefix, so enum
			ExtractPackets( GetArrayBridge(PacketData), FormatDescription, EnumPacket );
		}
		else
		{
			throw Soy::AssertException("Expecting block buffer in h264 packet");
		}
	}
	/*
	{
		CMBlockBufferRef dataBuffer = CMSampleBufferGetDataBuffer(SampleBuffer);
		size_t length, totalLength;
		char *dataPointer;
		OSStatus statusCodeRet = CMBlockBufferGetDataPointer(dataBuffer, 0, &length, &totalLength, &dataPointer);
		if (statusCodeRet == noErr) {
		
		size_t bufferOffset = 0;
		static const int AVCCHeaderLength = 4;
		while (bufferOffset < totalLength - AVCCHeaderLength) {
			
			// Read the NAL unit length
			uint32_t NALUnitLength = 0;
			memcpy(&NALUnitLength, dataPointer + bufferOffset, AVCCHeaderLength);
			
			// Convert the length value from Big-endian to Little-endian
			NALUnitLength = CFSwapInt32BigToHost(NALUnitLength);
			
			NSData* data = [[NSData alloc] initWithBytes:(dataPointer + bufferOffset + AVCCHeaderLength) length:NALUnitLength];
			[encoder->_delegate gotEncodedData:data isKeyFrame:keyframe];
			
			// Move to the next NAL unit in the block buffer
			bufferOffset += AVCCHeaderLength + NALUnitLength;
	 }
	}
	*/
}

void Avf::TCompressor::OnPacket(const ArrayBridge<uint8_t>&& Data,SoyTime PresentationTime)
{
	//	fill output with nalu header
	Array<uint8_t> NaluPacket;
	NaluPacket.PushBack(0);
	NaluPacket.PushBack(0);
	NaluPacket.PushBack(0);
	NaluPacket.PushBack(1);

	static bool Debug = false;
	if ( Debug )
	{
		//	content type should already be here
		H264NaluContent::Type Content;
		H264NaluPriority::Type Priority;
		auto NaluByte = Data[0];
		H264::DecodeNaluByte( NaluByte, Content, Priority );
		std::Debug << __PRETTY_FUNCTION__ << " x" << Data.GetDataSize() << "bytes (pre-0001) " << magic_enum::enum_name(Content) << " " << magic_enum::enum_name(Priority) << std::endl;
	}
	
	NaluPacket.PushBackArray(Data);
	
	auto FrameNumber = PresentationTime.mTime / 1000;
	
	mOnPacket( GetArrayBridge(NaluPacket), FrameNumber );
}


void Avf::TCompressor::Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber)
{
	auto Lambda = ^
	{
		//	we're using this to pass a frame number, but really we should be giving a real time to aid the encoder
		CMTime presentationTimeStamp = CMTimeMake(FrameNumber, 1);
		VTEncodeInfoFlags OutputFlags = 0;
		
		//	specifying duration helps with bitrates and keyframing
		//kCMTimeInvalid
		auto Duration = Soy::Platform::GetTime( SoyTime(std::chrono::milliseconds(33)) );
		void* FrameMeta = nullptr;
		
		//	kVTEncodeFrameOptionKey_ForceKeyFrame
		CFDictionaryRef frameProperties = nullptr;
		void* FrameMeta = nullptr;
		
		// Pass it to the encoder
		auto Status = VTCompressionSessionEncodeFrame(EncodingSession,
													  PixelBuffer,
													  presentationTimeStamp,
													  Duration,
													  frameProperties, FrameMeta, &OutputFlags);
		Avf::IsOkay(Status,"VTCompressionSessionEncodeFrame");

		//	expecting this output to be async
		static bool Debug = false;
		if ( Debug )
		{
			auto FrameDropped = ( OutputFlags & kVTEncodeInfo_FrameDropped) != 0;
			auto EncodingAsync = ( OutputFlags & kVTEncodeInfo_Asynchronous) != 0;
			std::Debug << "VTCompressionSessionEncodeFrame returned FrameDropped=" << FrameDropped << " EncodingAsync=" << EncodingAsync << std::endl;
		}
	};
	dispatch_sync(aQueue,Lambda);
}
	

	
Avf::TEncoder::TEncoder(std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket )
{
}

Avf::TEncoder::~TEncoder()
{
	mCompressor.reset();
}


void Avf::TEncoder::AllocEncoder(const SoyPixelsMeta& Meta)
{
	//	todo: change PPS if content changes
	if (mPixelMeta.IsValid())
	{
		if (mPixelMeta == Meta)
			return;
		std::stringstream Error;
		Error << "H264 encoder pixel format changing from " << mPixelMeta << " to " << Meta << ", currently unsupported";
		throw Soy_AssertException(Error);
	}
	
	auto OnPacket = [this](const ArrayBridge<uint8_t>&& PacketData,size_t FrameNumber)
	{
		this->OnPacketCompressed( PacketData, FrameNumber );
	};

	mCompressor.reset( new TCompressor( Meta, OnPacket ) );
	mPixelMeta = Meta;
}

void Avf::TEncoder::Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta)
{
	//	this should be fast as it sends to encoder, but synchronous
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__, 2);
	
	//	todo: make a complete CVPixelBuffer
	/*
	{
		auto YuvFormat = SoyPixelsFormat::GetMergedFormat( Luma.GetFormat(), ChromaU.GetFormat(), ChromaV.GetFormat() );
		auto YuvWidth = Luma.GetWidth();
		auto YuvHeight = Luma.GetHeight();
		SoyPixelsMeta YuvMeta( YuvWidth, YuvHeight, YuvFormat );
		AllocEncoder(YuvMeta);
	}
	*/
	auto& EncodePixels = Luma;
	AllocEncoder( EncodePixels.GetMeta() );

	auto PixelBuffer = Avf::PixelsToPixelBuffer(EncodePixels);
	auto FrameNumber = PushFrameMeta(Meta);
		
	mCompressor->Encode( PixelBuffer, FrameNumber );
}



void Avf::TEncoder::FinishEncoding()
{
	mCompressor.reset();
}



size_t Avf::TEncoder::PushFrameMeta(const std::string& Meta)
{
	TFrameMeta FrameMeta;
	FrameMeta.mFrameNumber = mFrameCount;
	FrameMeta.mMeta = Meta;
	mFrameMetas.PushBack(FrameMeta);
	mFrameCount++;
	return FrameMeta.mFrameNumber;
}

std::string Avf::TEncoder::GetFrameMeta(size_t FrameNumber)
{
	for ( auto i=0;	i<mFrameMetas.GetSize();	i++ )
	{
		auto& FrameMeta = mFrameMetas[i];
		if ( FrameMeta.mFrameNumber != FrameNumber )
			continue;

		//	gr: for now, sometimes we get multiple packets for one frame, so we can't discard them all
		//auto Meta = mFrameMetas.PopAt(i);
		auto Meta = mFrameMetas[i];
		return Meta.mMeta;
	}
	
	std::stringstream Error;
	Error << "No frame meta matching frame number " << FrameNumber;
	throw Soy::AssertException(Error);
}
	
void Avf::TEncoder::OnPacketCompressed(const ArrayBridge<uint8_t>& Data,size_t FrameNumber)
{
	Soy::TScopeTimerPrint Timer("OnNalPacket",2);
	//auto DecodeOrderNumber = mPicture.i_dts;
	
	//std::Debug << "OnNalPacket( pts=" << FrameNumber << ", dts=" << DecodeOrderNumber << ")" << std::endl;
	auto FrameMeta = GetFrameMeta(FrameNumber);
			
	PopH264::TPacket OutputPacket;
	OutputPacket.mData.reset(new Array<uint8_t>());
	OutputPacket.mInputMeta = FrameMeta;
	OutputPacket.mData->PushBackArray(Data);
	OnOutputPacket(OutputPacket);
}

