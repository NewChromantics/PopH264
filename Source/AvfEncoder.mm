#include "AvfEncoder.h"
#include "SoyPixels.h"
#include "SoyAvf.h"
#include "SoyFourcc.h"
#include "MagicEnum/include/magic_enum.hpp"
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
}
	
	

class Avf::TCompressor
{
public:
	TCompressor(TEncoderParams& Params,const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&&,size_t)> OnPacket);
	~TCompressor();
	
	void	OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer);
	void	Flush();

	void	Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber,bool Keyframe);
	
private:
	void	OnPacket(const ArrayBridge<uint8_t>&& Data,SoyTime PresentationTime);
	
private:
	std::function<void(const ArrayBridge<uint8_t>&&,size_t)>	mOnPacket;
	
	VTCompressionSessionRef	mSession = nil;
	dispatch_queue_t		mQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
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



Avf::TCompressor::TCompressor(TEncoderParams& Params,const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&&,size_t)> OnPacket) :
	mOnPacket	( OnPacket )
{
	if ( !mOnPacket )
		throw Soy::AssertException("OnPacket callback missing in Avf::TCompressor");
	
	//h264Encoder = [H264HwEncoderImpl alloc];
	//	[h264Encoder initWithConfiguration];
	//auto Lambda = ^
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
		std::map<size_t,size_t> ProfileLevelMaxHeight =
		{
			{	30,	576	},	//	720×576@25	720×480@30
			{	31,	720	},	//	1,280×720@30.0 (5)
			{	32,	1024	},		//	1,280×1,024@42.2 (4)
			{	40,	1080	},	//	1,920×1,080@30.1	2,048×1,024@30.0
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
		
		{
			void* CallbackParam = this;
			auto Width = Meta.GetWidth();
			auto Height = Meta.GetHeight();
			OSStatus status = VTCompressionSessionCreate( NULL, Width, Height, kCMVideoCodecType_H264, sessionAttributes, NULL, NULL, OnCompressedCallback, CallbackParam, &mSession );
			//std::Debug << "H264: VTCompressionSessionCreate " << status << std::endl;
			Avf::IsOkay(status,"VTCompressionSessionCreate");
		}

		std::string ProfileDebug;
		{
			auto ProfileNumber = Params.mProfileLevel;
			auto Profile = ProfileLevelValue[ProfileNumber];
			std::stringstream Debug;
			Debug << Soy::CFStringToString( Profile) << "=" << ProfileNumber << ";";
			ProfileDebug = Debug.str();
		}

		
		{
			auto ProfileNumber = Params.mProfileLevel;
			//	gr: kVTProfileLevel_H264_Baseline_3_0 always fails in compression callback with -12348 on osx
			//	32 not supported on iphonese/13 (VTSessionSetProperty fails)
			auto Profile = ProfileLevelValue[ProfileNumber];
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_ProfileLevel, Profile);
			Avf::IsOkay(status, ProfileDebug + "kVTCompressionPropertyKey_ProfileLevel" );
			
			if ( ProfileNumber != 0 )
			{
				auto Height = Meta.GetHeight();
				auto MaxHeight = ProfileLevelMaxHeight[ProfileNumber];
				std::Debug << "H264 using profile " << ProfileNumber << " with height " << Height << ", max height in spec " << MaxHeight << std::endl;
			}
		}
		
		{
			auto Realtime = Params.mRealtime ? kCFBooleanTrue : kCFBooleanFalse;
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_RealTime, Realtime);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_RealTime");
		}
		
		//	gr: this is the correct logic! (name sounds backwards to me)
		//		does this also enough more non-keyframes?
		{
			static auto OutputFramesInOrder = true;
			auto FrameReorder = OutputFramesInOrder ? kCFBooleanTrue : kCFBooleanFalse;
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_AllowFrameReordering, FrameReorder );
			Avf::IsOkay(status,"kVTCompressionPropertyKey_AllowFrameReordering");
		}
		
		//	if this is false, it will force all frames to be keyframes
		//	kVTCompressionPropertyKey_AllowTemporalCompression
		
		//	control quality
		if ( Params.mAverageKbps > 0 )
		{
			//	this was giving about 25x too much, maybe im giving it the wrong values, but I dont think so
			int32_t AverageBitRate = Params.mAverageKbps * 1024 * 8;
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &AverageBitRate );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_AverageBitRate, Number);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_AverageBitRate");
		}
		
		//	gr: setting this on my iphone 5s (ios 12) makes every frame drop
		//	gr: setting on iphone SE (ios 13) has little effect
		if ( Params.mMaxKbps > 0 )
		{
			int32_t Bytes = Params.mMaxKbps * 1024;
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
			Avf::IsOkay(Status,"kVTCompressionPropertyKey_DataRateLimits");
		}
		
		if ( Params.mMaxSliceBytes > 0 )
		{
			int32_t MaxSliceBytes = Params.mMaxSliceBytes;
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &MaxSliceBytes );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaxH264SliceBytes, Number);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_MaxH264SliceBytes");
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
			int32_t MaxFrameBuffers = Params.mMaxFrameBuffers;
			CFNumberRef Number = CFNumberCreate(NULL, kCFNumberSInt32Type, &MaxFrameBuffers );
			auto status = VTSessionSetProperty(mSession, kVTCompressionPropertyKey_MaxFrameDelayCount, Number);
			Avf::IsOkay(status,"kVTCompressionPropertyKey_MaxFrameDelayCount");
		}
		
		auto status = VTCompressionSessionPrepareToEncodeFrames(mSession);
		Avf::IsOkay(status,ProfileDebug + "VTCompressionSessionPrepareToEncodeFrames");
	};
	//dispatch_sync(mQueue, Lambda);
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
	
	//	doing this check after getting meta to help debug
	if (!CMSampleBufferDataIsReady(SampleBuffer))
	{
		auto WasDropped = (infoFlags & kVTEncodeInfo_FrameDropped) ? "(Frame Dropped)" : "";
		throw Soy::AssertException( std::string("Data sample not ready") + WasDropped );
	}

	//	look for SPS & PPS data if we have a keyframe
	//	AFTER CMSampleBufferDataIsReady as SampleBuffer may be null
	CFDictionaryRef Dictionary = static_cast<CFDictionaryRef>( CFArrayGetValueAtIndex(CMSampleBufferGetSampleAttachmentsArray(SampleBuffer, true), 0) );
	bool IsKeyframe = !CFDictionaryContainsKey( Dictionary, kCMSampleAttachmentKey_NotSync);
	

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


void Avf::TCompressor::Encode(CVPixelBufferRef PixelBuffer,size_t FrameNumber,bool Keyframe)
{
	//	this throws with uncaught exceptions if in a dispatch queue,
	//	does it need to be? it was syncronous anyway
	//auto Lambda = ^
	{
		//	we're using this to pass a frame number, but really we should be giving a real time to aid the encoder
		CMTime presentationTimeStamp = CMTimeMake(FrameNumber, 1);
		VTEncodeInfoFlags OutputFlags = 0;
		
		//	specifying duration helps with bitrates and keyframing
		//kCMTimeInvalid
		auto Duration = Soy::Platform::GetTime( SoyTime(std::chrono::milliseconds(33)) );
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
		static bool Debug = false;
		if ( Debug )
		{
			auto FrameDropped = ( OutputFlags & kVTEncodeInfo_FrameDropped) != 0;
			auto EncodingAsync = ( OutputFlags & kVTEncodeInfo_Asynchronous) != 0;
			std::Debug << "VTCompressionSessionEncodeFrame returned FrameDropped=" << FrameDropped << " EncodingAsync=" << EncodingAsync << std::endl;
		}
	};
	//dispatch_sync(mQueue,Lambda);
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
	
	auto OnPacket = [this](const ArrayBridge<uint8_t>&& PacketData,size_t FrameNumber)
	{
		this->OnPacketCompressed( PacketData, FrameNumber );
	};

	mCompressor.reset( new TCompressor( mParams, Meta, OnPacket ) );
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

