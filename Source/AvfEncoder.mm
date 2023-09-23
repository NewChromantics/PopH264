#include "AvfEncoder.h"
#include "SoyPixels.h"
#include "SoyAvf.h"
#include "SoyFourcc.h"
#include "SoyLib/src/magic_enum/include/magic_enum.hpp"
#include "json11.hpp"

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

#include "PopH264.h"	//	param keys
#include <span>

//#define EXECUTE_ON_DISPATCH_QUEUE

#include <CoreMedia/CMTime.h>

//	we were using framenumber=secs but didn't produce many non-keyframes (each frame was 1sec apart)
//	using a unit helps that; all frames still seem to be Slice_NonIDRPicture, but many much smaller ones
//	(bit confusing)
const auto FrameNumberToTimeUnit = 30;

CMTime FrameNumberToTime(size_t FrameNumber)
{
	//	1 unit / 30 per sec
	CMTime Timestamp = CMTimeMake( FrameNumber, FrameNumberToTimeUnit );
	return Timestamp;
}

size_t TimeToFrameNumber(CMTime Time)
{
	if ( CMTIME_IS_INVALID( Time ) )
		throw Soy::AssertException("Invalid timestamp");

	//	missing CMTimeGetSeconds ? link to the CoreMedia framework
	auto Value = Time.value;		//	frame number
	auto Scale = Time.timescale;	//	scale
	
	if ( Scale != FrameNumberToTimeUnit )
	{
		auto Seconds = CMTimeGetSeconds(Time);
		std::Debug << "TimeToFrameNumber( value=" << Value << " Scale=" << Scale <<") Seconds=" << Seconds << std::endl;
	}
	
	return Value;
}


Avf::TEncoderParams::TEncoderParams(json11::Json& Options)
{
	auto SetInt = [&](const char* Name,size_t& ValueUnsigned)
	{
		auto& Handle = Options[Name];
		if ( !Handle.is_number() )
			return false;
		auto Value = Handle.int_value();
		if ( Value < 0 )
		{
			std::stringstream Error;
			Error << "Value for " << Name << " is " << Value << ", not expecting negative";
			throw Soy::AssertException(Error);
		}
		ValueUnsigned = Value;
		return true;
	};
	auto SetBool = [&](const char* Name,bool& Value)
	{
		auto& Handle = Options[Name];
		if ( !Handle.is_bool() )
			return false;
		Value = Handle.bool_value();
		return true;
	};
	SetBool( POPH264_ENCODER_KEY_REALTIME, mRealtime );
	SetInt( POPH264_ENCODER_KEY_AVERAGEKBPS, mAverageKbps );
	SetInt( POPH264_ENCODER_KEY_MAXKBPS, mMaxKbps );
	SetInt( POPH264_ENCODER_KEY_MAXFRAMEBUFFERS, mMaxFrameBuffers );
	SetInt( POPH264_ENCODER_KEY_MAXSLICEBYTES, mMaxSliceBytes );
	SetBool( POPH264_ENCODER_KEY_MAXIMISEPOWEREFFICIENCY, mMaximisePowerEfficiency );
	SetInt( POPH264_ENCODER_KEY_PROFILELEVEL, mProfileLevel );
	SetInt( POPH264_ENCODER_KEY_KEYFRAMEFREQUENCY, mKeyFrameFrequency );
}
	
	

class Avf::TCompressor
{
public:
	TCompressor(TEncoderParams& Params,const SoyPixelsMeta& Meta,std::function<void(std::span<uint8_t>,size_t)> OnPacket,std::function<void(std::string_view Error)> OnFinished);
	~TCompressor();
	
	void	OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer);
	void	OnCompressionFinished();
	void	OnError(std::string_view Error);
	void	Flush();

	void	Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber,bool Keyframe);
	
private:
	void	OnPacket(std::span<uint8_t> Data,size_t FrameNumber);
	
private:
	std::function<void(std::span<uint8_t>,size_t)>	mOnPacket;
	std::function<void(std::string_view Error)>		mOnFinished;

	VTCompressionSessionRef	mSession = nil;
	dispatch_queue_t		mQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
};

void OnCompressedCallback(void *outputCallbackRefCon,void *sourceFrameRefCon, OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer)
{
	//	https://chromium.googlesource.com/external/webrtc/+/6c78307a21252c2dbd704f6d5e92a220fb722ed4/webrtc/modules/video_coding/codecs/h264/h264_video_toolbox_encoder.mm#588
	if ( !outputCallbackRefCon )
	{
		std::Debug << "OnCompressedCallback missing this" << std::endl;
		return;
	}
	
	auto* This = static_cast<Avf::TCompressor*>(outputCallbackRefCon);
	try
	{
		This->OnCompressed(status,infoFlags,sampleBuffer);
		
		//	gr: I think if we have this callback, it's when everything is done
		//		not per-frame (that's a different callback!)
		This->OnCompressionFinished();
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception with OnCompressed callback; " << e.what() << std::endl;
		This->OnError(e.what());
	}
}



Avf::TCompressor::TCompressor(TEncoderParams& Params,const SoyPixelsMeta& Meta,std::function<void(std::span<uint8_t>,size_t)> OnPacket,std::function<void(std::string_view Error)> OnFinished) :
	mOnPacket	( OnPacket ),
	mOnFinished	( OnFinished )
{
	if ( !mOnPacket )
		throw Soy::AssertException("OnPacket callback missing in Avf::TCompressor");
	if ( !mOnFinished )
		throw Soy::AssertException("OnFinished callback missing in Avf::TCompressor");

	//h264Encoder = [H264HwEncoderImpl alloc];
	//	[h264Encoder initWithConfiguration];
#if defined(EXECUTE_ON_DISPATCH_QUEUE)
	auto Lambda = ^
#endif
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

		//	https://en.wikipedia.org/w/index.php?title=Advanced_Video_Coding
		//	gr: so iphone5s will send resolution above what's supproted, but SE (or ios13) won't
		//		lets pre-empt this
		//	problem: the auto baseline the SE chooses(4) won't decode
		//	gr: using 30fps values
		std::map<size_t,std::tuple<size_t,size_t>> ProfileLevelMaxSize =
		{
			{	0,	{0,0}		},
			{	13,	{352,288}	},	//
			{	30,	{720,576}	},	//	720×576@25	720×480@30
			{	31,	{1280,720}	},	//	1,280×720@30.0 (5)
			{	32,	{1280,1024}	},	//	1,280×1,024@42.2 (4)
			{	40,	{2048,1024}	},	//	1,920×1,080@30.1	2,048×1,024@30.0
			{	41,	{2048,1024}	},	//
			{	42,	{2048,1080}	},	//
			{	50,	{2560,1920}	},	//
			{	51,	{4096,2048}	},	//
			{	52,	{4096,2304}	},	//
		};
		std::map<size_t,CFStringRef> ProfileLevelValue =
		{
			{	0,	kVTProfileLevel_H264_Baseline_AutoLevel	},
			{	13,	kVTProfileLevel_H264_Baseline_1_3	},
			{	30,	kVTProfileLevel_H264_Baseline_3_0	},
			{	31,	kVTProfileLevel_H264_Baseline_3_1	},
			{	32,	kVTProfileLevel_H264_Baseline_3_2	},
			{	40,	kVTProfileLevel_H264_Baseline_4_0	},
			{	41,	kVTProfileLevel_H264_Baseline_4_1	},
			{	42,	kVTProfileLevel_H264_Baseline_4_2	},
			{	50,	kVTProfileLevel_H264_Baseline_5_0	},
			{	51,	kVTProfileLevel_H264_Baseline_5_1	},
			{	52,	kVTProfileLevel_H264_Baseline_5_2	},
		};
		
		std::stringstream ProfileDebug;
		{
			auto ProfileNumber = Params.mProfileLevel;
			auto Profile = ProfileLevelValue[ProfileNumber];
			auto ProfileMaxSize = ProfileLevelMaxSize[ProfileNumber];
			ProfileDebug << Soy::CFStringToString(Profile) << "=" << ProfileNumber << ";";
			ProfileDebug << "ProfileMaxSize=" << std::get<0>(ProfileMaxSize) << "x" << std::get<1>(ProfileMaxSize) << ";";
		}
				
		{
			void* CallbackParam = this;
			auto Width = size_cast<int32_t>(Meta.GetWidth());
			auto Height = size_cast<int32_t>(Meta.GetHeight());
			ProfileDebug << Width << "x" << Height << ";";
			OSStatus status = VTCompressionSessionCreate( NULL, Width, Height, kCMVideoCodecType_H264, sessionAttributes, NULL, NULL, OnCompressedCallback, CallbackParam, &mSession );
			//std::Debug << "H264: VTCompressionSessionCreate " << status << std::endl;
			Avf::IsOkay(status, ProfileDebug.str()+"VTCompressionSessionCreate");
		}
		
		{
			auto ProfileNumber = Params.mProfileLevel;
			//	gr: kVTProfileLevel_H264_Baseline_3_0 always fails in compression callback with -12348 on osx
			//	32 not supported on iphonese/13 (VTSessionSetProperty fails)
			auto Profile = ProfileLevelValue[ProfileNumber];
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_ProfileLevel, Profile);
			Avf::IsOkay(status, ProfileDebug.str()+"kVTCompressionPropertyKey_ProfileLevel" );
		}
		
		{
			auto Realtime = Params.mRealtime ? kCFBooleanTrue : kCFBooleanFalse;
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_RealTime, Realtime);
			Avf::IsOkay(status,ProfileDebug.str()+"kVTCompressionPropertyKey_RealTime");
		}
		
		//	gr: this is the correct logic! (name sounds backwards to me)
		//		does this also enough more non-keyframes?
		{
			static auto OutputFramesInOrder = true;
			auto FrameReorder = OutputFramesInOrder ? kCFBooleanTrue : kCFBooleanFalse;
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_AllowFrameReordering, FrameReorder );
			Avf::IsOkay(status,ProfileDebug.str()+"kVTCompressionPropertyKey_AllowFrameReordering");
		}
		
		//	if this is false, it will force all frames to be keyframes
		//	kVTCompressionPropertyKey_AllowTemporalCompression
		
		//	control quality
		if ( Params.mAverageKbps > 0 )
		{
			//	this was giving about 25x too much, maybe im giving it the wrong values, but I dont think so
			int32_t AverageBitRate = size_cast<int32_t>(Params.mAverageKbps * 1024 * 8);
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &AverageBitRate );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_AverageBitRate, Number);
			Avf::IsOkay(status,ProfileDebug.str()+"kVTCompressionPropertyKey_AverageBitRate");
		}
		
		//	gr: setting this on my iphone 5s (ios 12) makes every frame drop
		//	gr: setting on iphone SE (ios 13) has little effect
		if ( Params.mMaxKbps > 0 )
		{
			int32_t Bytes = size_cast<int32_t>(Params.mMaxKbps * 1024);
			int32_t Secs = 1;
			CFNumberRef n1 = CFNumberCreate( kCFAllocatorDefault, kCFNumberSInt32Type, &Bytes );
			CFNumberRef n2 = CFNumberCreate( kCFAllocatorDefault, kCFNumberSInt32Type, &Secs );
			const void *values[] = {n1, n2};
			auto ValueCount = std::size(values);
			CFArrayRef dataRateLimits = CFArrayCreate(kCFAllocatorDefault,
													  (const void**)&values,
													  ValueCount,
													  nullptr);
			auto Status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_DataRateLimits, dataRateLimits);
			Avf::IsOkay(Status,ProfileDebug.str()+"kVTCompressionPropertyKey_DataRateLimits");
		}
		
		if ( Params.mMaxSliceBytes > 0 )
		{
			int32_t MaxSliceBytes = size_cast<int32_t>(Params.mMaxSliceBytes);
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &MaxSliceBytes );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaxH264SliceBytes, Number);
			Avf::IsOkay(status,ProfileDebug.str()+"kVTCompressionPropertyKey_MaxH264SliceBytes");
		}

		auto OsVersion = Platform::GetOsVersion();
#if defined(TARGET_IOS)
		auto MaxPowerSupported = true;
#else
		auto MaxPowerSupported = OsVersion.mMinor >= 14;
#endif
		if ( MaxPowerSupported )
		{
			auto MaximisePE = Params.mMaximisePowerEfficiency ? kCFBooleanTrue : kCFBooleanFalse;
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaximizePowerEfficiency, MaximisePE);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_MaximizePowerEfficiency");
		}
		else
		{
			std::Debug << "kVTCompressionPropertyKey_MaximizePowerEfficiency not supported on this OS version " << OsVersion << std::endl;
		}
		
		//	-1 is unlimited, and is the default
		if ( Params.mMaxFrameBuffers > 0 )
		{
			int32_t MaxFrameBuffers = size_cast<int32_t>(Params.mMaxFrameBuffers);
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &MaxFrameBuffers );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaxFrameDelayCount, Number);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_MaxFrameDelayCount");
		}
		
		if ( Params.mKeyFrameFrequency > 0 )
		{
			int32_t KeyframeFrequency = size_cast<int32_t>(Params.mKeyFrameFrequency);
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &KeyframeFrequency );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaxKeyFrameInterval, Number);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_MaxKeyFrameInterval");
		}
		
		auto status = VTCompressionSessionPrepareToEncodeFrames(mSession);
		Avf::IsOkay(status,ProfileDebug.str()+"VTCompressionSessionPrepareToEncodeFrames");
	};
#if defined(EXECUTE_ON_DISPATCH_QUEUE)
	dispatch_sync(mQueue, Lambda);
#endif
}

Avf::TCompressor::~TCompressor()
{
	Flush();
	
	// End the session
	VTCompressionSessionInvalidate( mSession );
	CFRelease( mSession );
	
	//	wait for the queue to end
}

void Avf::TCompressor::Flush()
{
	auto Error = VTCompressionSessionCompleteFrames( mSession, kCMTimeInvalid );
	IsOkay(Error,"VTCompressionSessionCompleteFrames");
}


void AnnexBToAnnexB(std::span<uint8_t> Data,std::function<void(std::span<uint8_t>)> EnumPacket)
{
	//	gr: does this start with 0001 etc? if so, cut
	Soy_AssertTodo();
	//EnumPacket( GetArrayBridge(Data) );
}

void NaluToAnnexB(std::span<uint8_t> Data,size_t LengthSize,std::function<void(std::span<uint8_t>)> EnumPacket)
{
	//	walk through data
	int i=0;
	while ( i <Data.size() )
	{
		size_t ChunkLength = 0;
		//auto* pData = &Data[i+0];
		
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
	
		auto PacketContent = Data.subspan( i+LengthSize, ChunkLength );

		EnumPacket( PacketContent );
		
		i += LengthSize + ChunkLength;
	}
}


//	this could be multiple nals, and we need to cut the prefix, so enum
//	gr: why is this extern C?
extern "C" void ExtractPackets(std::span<uint8_t> Packets,CMFormatDescriptionRef FormatDescription,std::function<void(std::span<uint8_t>)> EnumPacket)
{
	int nal_size_field_bytes = 0;
	//	SPS & PPS (&sei?) set count, maybe we should integrate that into this func
	size_t ParamSetCount = 0;
	auto Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( FormatDescription, 0, nullptr, nullptr, &ParamSetCount, &nal_size_field_bytes );
	Avf::IsOkay( Result, "Get H264 param NAL size");
	
	//	extract header packets
	//	SPS, then PPS
	//H264NaluContent::Type NaluContentTypes[] = { H264NaluContent::SequenceParameterSet, H264NaluContent::PictureParameterSet };
	for ( auto i=0;	i<ParamSetCount;	i++ )
	{
		if ( i > 1 )
			throw Soy::AssertException("Got Packet header > SPS & PPS");
		std::vector<uint8_t> SpsData;
		Avf::GetFormatDescriptionData( SpsData, FormatDescription, i );

		//	gr: this header is already here. lets debug it in EnumPacket though
		//	insert nalu header
		//auto Content = NaluContentTypes[i];
		//auto Priority = H264NaluPriority::Important;
		//auto NaluByte = H264::EncodeNaluByte(Content,Priority);
		EnumPacket( std::span(SpsData) );
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
	//CMTime DecodeTimestamp = CMSampleBufferGetDecodeTimeStamp(SampleBuffer);
	//CMTime SampleDuration = CMSampleBufferGetDuration(SampleBuffer);
	auto FrameNumber = TimeToFrameNumber(PresentationTimestamp);
	//	todo: deal with OOO packets with theirown decode time
	//auto DecodeTimecode = Soy::Platform::GetTime(DecodeTimestamp);
	//auto Duration = Soy::Platform::GetTime(SampleDuration);
	
	//	doing this check after getting meta to help debug
	if (!CMSampleBufferDataIsReady(SampleBuffer))
	{
		auto WasDropped = (infoFlags & kVTEncodeInfo_FrameDropped) ? "(Frame Dropped)" : "";
		throw Soy::AssertException( std::string("Data sample not ready") + WasDropped );
	}

	//	look for SPS & PPS data if we have a keyframe
	//	AFTER CMSampleBufferDataIsReady as SampleBuffer may be null
	//CFDictionaryRef Dictionary = static_cast<CFDictionaryRef>( CFArrayGetValueAtIndex(CMSampleBufferGetSampleAttachmentsArray(SampleBuffer, true), 0) );
	//bool IsKeyframe = !CFDictionaryContainsKey( Dictionary, kCMSampleAttachmentKey_NotSync);
	

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
			auto DataSize = CMBlockBufferGetDataLength( BlockBuffer );
			std::vector<uint8_t> PacketData( DataSize );
			auto Result = CMBlockBufferCopyDataBytes( BlockBuffer, 0, PacketData.size(), PacketData.data() );
			Avf::IsOkay( Result, "CMBlockBufferCopyDataBytes" );
			
			auto EnumPacket = [&](std::span<uint8_t> PacketData)
			{
				OnPacket( PacketData, FrameNumber );
			};
			//	this could be multiple nals, and we need to cut the prefix, so enum
			ExtractPackets( PacketData, FormatDescription, EnumPacket );
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


void Avf::TCompressor::OnCompressionFinished()
{
	mOnFinished( std::string_view() );
}

void Avf::TCompressor::OnError(std::string_view Error)
{
	mOnFinished(Error);
}

void Avf::TCompressor::OnPacket(std::span<uint8_t> Data,size_t FrameNumber)
{
	//	fill output with nalu header
	std::vector<uint8_t> NaluPacket{0,0,0,1};

	static bool Debug = true;
	if ( Debug )
	{
		//	content type should already be here
		H264NaluContent::Type Content;
		H264NaluPriority::Type Priority;
		auto NaluByte = Data[0];
		H264::DecodeNaluByte( NaluByte, Content, Priority );
		std::Debug << __PRETTY_FUNCTION__ << " x" << Data.size() << "bytes (pre-0001) " << magic_enum::enum_name(Content) << " " << magic_enum::enum_name(Priority) << std::endl;
	}
	
	std::copy( Data.begin(), Data.end(), std::back_inserter(NaluPacket) );

	mOnPacket( NaluPacket, FrameNumber );
}


void Avf::TCompressor::Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber,bool Keyframe)
{
	//	this throws with uncaught exceptions if in a dispatch queue,
	//	does it need to be? it was syncronous anyway
#if defined(EXECUTE_ON_DISPATCH_QUEUE)
	auto Lambda = ^
#endif
	{
		//	we're using this to pass a frame number, but really we should be giving a real time to aid the encoder
		CMTime presentationTimeStamp = FrameNumberToTime(FrameNumber);
		VTEncodeInfoFlags OutputFlags = 0;
		
		//	specifying duration helps with bitrates and keyframing
		//kCMTimeInvalid
		auto Duration = Soy::Platform::GetTime( std::chrono::milliseconds(33) );
		void* FrameMeta = nullptr;

		//	set keyframe
		CFMutableDictionaryRef FrameProperties = CFDictionaryCreateMutable( NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
		CFDictionarySetValue( FrameProperties, kVTEncodeFrameOptionKey_ForceKeyFrame, Keyframe ? kCFBooleanTrue : kCFBooleanFalse );
	
		// Pass it to the encoder
		auto Status = VTCompressionSessionEncodeFrame(mSession,
													  PixelBuffer,
													  presentationTimeStamp,
													  Duration,
													  FrameProperties, FrameMeta, &OutputFlags);
		Avf::IsOkay(Status,"VTCompressionSessionEncodeFrame");

		CFRelease(PixelBuffer);
		
		//	expecting this output to be async
		static bool Debug = true;
		if ( Debug )
		{
			auto FrameDropped = ( OutputFlags & kVTEncodeInfo_FrameDropped) != 0;
			auto EncodingAsync = ( OutputFlags & kVTEncodeInfo_Asynchronous) != 0;
			std::Debug << "VTCompressionSessionEncodeFrame returned FrameDropped=" << FrameDropped << " EncodingAsync=" << EncodingAsync << std::endl;
		}
	};
#if defined(EXECUTE_ON_DISPATCH_QUEUE)
	dispatch_sync(mQueue,Lambda);
#endif
}
	

	
Avf::TEncoder::TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket ),
	mParams				( Params )
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
	
	auto OnPacket = [this](std::span<uint8_t> PacketData,size_t FrameNumber)
	{
		this->OnPacketCompressed( PacketData, FrameNumber );
	};
	auto OnFinished = [this](std::string_view Error)
	{
		if ( !Error.empty() )
			this->OnError( Error );
		else
			this->OnFinished();
	};

	mCompressor.reset( new TCompressor( mParams, Meta, OnPacket, OnFinished ) );
	mPixelMeta = Meta;
}

void Avf::TEncoder::Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta,bool Keyframe)
{
	Soy_AssertTodo();
}


void Avf::TEncoder::Encode(const SoyPixelsImpl& Pixels,const std::string& Meta,bool Keyframe)
{
	//	this should be fast as it sends to encoder, but synchronous
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__, 13);

	AllocEncoder( Pixels.GetMeta() );

	if ( Pixels.GetFormat() == SoyPixelsFormat::Greyscale )
	{
		std::Debug << __PRETTY_FUNCTION__ << " Warning, encoding greyscale image which seems to come out pure grey regardless of data on ios (ipad 2020 ios14)" << std::endl;
	}

	auto PixelBuffer = Avf::PixelsToPixelBuffer(Pixels);
	auto FrameNumber = PushFrameMeta(Meta);
	
	mCompressor->Encode( PixelBuffer, FrameNumber, Keyframe );
}


void Avf::TEncoder::FinishEncoding()
{
	//	flush out frames
	if ( !mCompressor )
		return;
	
	mCompressor->Flush();
}


	
void Avf::TEncoder::OnPacketCompressed(std::span<uint8_t> Data,size_t FrameNumber)
{
	Soy::TScopeTimerPrint Timer("OnNalPacket",2);
	//auto DecodeOrderNumber = mPicture.i_dts;
	
	//std::Debug << "OnNalPacket( pts=" << FrameNumber << ", dts=" << DecodeOrderNumber << ")" << std::endl;
	auto FrameMeta = GetFrameMeta(FrameNumber);
			
	PopH264::TPacket OutputPacket;
	OutputPacket.mData.reset(new std::vector<uint8_t>());
	OutputPacket.mInputMeta = FrameMeta;
	std::copy( Data.begin(), Data.end(), std::back_inserter(*OutputPacket.mData) );
	OnOutputPacket(OutputPacket);
}

