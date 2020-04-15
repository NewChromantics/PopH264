#include "AppleEncoder.h"
#include "SoyPixels.h"
#include "SoyAvf.h"
#include "SoyFourcc.h"

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


class Avf::TCompressor
{
public:
	TCompressor(const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&,SoyTime)> OnPacket);
	~TCompressor();
	
	void	OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef sampleBuffer);
	void	Flush();

	void	Encode();
	
private:
	void	OnPacket(const ArrayBridge<uint8_t>&& Data,SoyTime PresentationTime);
	
private:
	std::function<void(const ArrayBridge<uint8_t>&,SoyTime)>	mOnPacket;

	
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



Avf::TCompressor::TCompressor(const SoyPixelsMeta& Meta,std::function<void(const ArrayBridge<uint8_t>&,SoyTime)> OnPacket) :
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
		std::Debug << "H264: VTCompressionSessionCreate " << status << std::endl;
		Avf::IsOkay(status,"VTCompressionSessionCreate");
		
		//设置properties（这些参数设置了也没用）
		auto Profile = kVTProfileLevel_H264_Baseline_5_2;
		//kVTProfileLevel_H264_High_5_2
		auto Realtime = kCFBooleanTrue;
		auto AllowFramesOutOfOrder = false;
		auto AutoOrderFrames = AllowFramesOutOfOrder ? kCFBooleanTrue : kCFBooleanFalse;
		VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_RealTime, Realtime);
		VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_ProfileLevel, Profile);
		VTSessionSetProperty(EncodingSession, kVTCompressionPropertyKey_AllowFrameReordering, AutoOrderFrames);
		
		// 启动编码
		VTCompressionSessionPrepareToEncodeFrames(EncodingSession);
	};
	dispatch_sync(aQueue, Lambda);
}

Avf::TCompressor::~TCompressor()
{
	Flush();
	
	// End the session
	VTCompressionSessionInvalidate( EncodingSession );
	CFRelease( EncodingSession );
}

void Avf::TCompressor::Flush()
{
	VTCompressionSessionCompleteFrames( EncodingSession, kCMTimeInvalid );
}

void AnnexBToAnnexB(ArrayBridge<uint8_t>&& Data)
{
	//	do nothing!
}

std::function<void(ArrayBridge<uint8_t>&&)> GetNaluConversionFunc(CMFormatDescriptionRef FormatDescription)
{
	int nal_size_field_bytes = 0;
	auto Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( FormatDescription, 0, nullptr, nullptr, nullptr, &nal_size_field_bytes );
	Avf::IsOkay( Result, "Get H264 param NAL size");
	
	//	-1 is annexB
	if ( nal_size_field_bytes < 0 )
		nal_size_field_bytes = 0;
	
	switch ( nal_size_field_bytes )
	{
		case 0:	return AnnexBToAnnexB;
	}
	
	std::stringstream Debug;
	Debug << "Unhandled nal_size_field_bytes " << nal_size_field_bytes;
	throw Soy::AssertException(Debug);
}


void Avf::TCompressor::OnCompressed(OSStatus status, VTEncodeInfoFlags infoFlags,CMSampleBufferRef SampleBuffer)
{
	std::Debug << __PRETTY_FUNCTION__ << "( status=" << status << " infoFlags=" << infoFlags << ")" << std::endl;
	Avf::IsOkay( status, "OnCompressed status");
	
	//H264HwEncoderImpl* encoder = (__bridge H264HwEncoderImpl*)outputCallbackRefCon;
	auto* encoder = this;
	
	CMFormatDescriptionRef FormatDescription = CMSampleBufferGetFormatDescription(SampleBuffer);

	auto DescFourcc = CFSwapInt32HostToBig( CMFormatDescriptionGetMediaSubType(FormatDescription) );
	Soy::TFourcc Fourcc( DescFourcc );
	
	//	get a function to convert nalu to what we want
	auto FixNaluData = GetNaluConversionFunc(FormatDescription);

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

	
	if ( IsKeyframe )
	{
		try
		{
			Array<uint8_t> SpsData;
			Avf::GetFormatDescriptionData( GetArrayBridge(SpsData), FormatDescription, 0 );
			FixNaluData( GetArrayBridge(SpsData) );
			OnPacket( GetArrayBridge(SpsData), PresentationTime );
		}
		catch(std::exception& e)
		{
			std::Debug << "Error getting SPS: " << e.what() << std::endl;
		}
		
		try
		{
			Array<uint8_t> PpsData;
			Avf::GetFormatDescriptionData( GetArrayBridge(PpsData), FormatDescription, 1 );
			FixNaluData( GetArrayBridge(PpsData) );
			OnPacket( GetArrayBridge(PpsData), PresentationTime );
		}
		catch(std::exception& e)
		{
			std::Debug << "Error getting PPS: " << e.what() << std::endl;
		}
	}

	//	extract data
	{
		//	this data could be an image buffer or a block buffer (for h264, expecting block)
		CMBlockBufferRef BlockBuffer = CMSampleBufferGetDataBuffer( SampleBuffer );
		//CVImageBufferRef ImageBuffer = CMSampleBufferGetImageBuffer( SampleBuffer );
		
		if ( BlockBuffer )
		{
			//	copy bytes into our array
			//	CMBlockBufferGetDataPointer is also an option
			Array<uint8_t> PacketData;
			
			auto DataSize = CMBlockBufferGetDataLength( BlockBuffer );
			PacketData.SetSize( DataSize );
			auto Result = CMBlockBufferCopyDataBytes( BlockBuffer, 0, PacketData.GetDataSize(), PacketData.GetArray() );
			Avf::IsOkay( Result, "CMBlockBufferCopyDataBytes" );
			
			//	note: this packet could have multiple NAL's
			//	we should adapt FixNaluData to spit out multiple packets
			FixNaluData( GetArrayBridge(PacketData) );
			OnPacket( GetArrayBridge(PacketData), PresentationTime );
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
	mOnPacket( Data, PresentationTime );
}

void Avf::TCompressor::Encode()
{
	CVPixelBufferRef PixelsToPixelBuffer(const SoyPixelsImpl& Pixels);

	CMSampleBufferRef
	encode:(CMSampleBufferRef )sampleBuffer // 频繁调用
		{
			dispatch_sync(aQueue, ^{
				
				frameCount++;
				// Get the CV Image buffer
				CVImageBufferRef imageBuffer = (CVImageBufferRef)CMSampleBufferGetImageBuffer(sampleBuffer);
				//            CVPixelBufferRef pixelBuffer = (CVPixelBufferRef)CMSampleBufferGetImageBuffer(sampleBuffer);
				
				// Create properties
				CMTime presentationTimeStamp = CMTimeMake(frameCount, 1); // 这个值越大画面越模糊
				//            CMTime duration = CMTimeMake(1, DURATION);
				VTEncodeInfoFlags flags;
				
				// Pass it to the encoder
				OSStatus statusCode = VTCompressionSessionEncodeFrame(EncodingSession,
																	  imageBuffer,
																	  presentationTimeStamp,
																	  kCMTimeInvalid,
																	  NULL, NULL, &flags);
				// Check for error
				if (statusCode != noErr) {
					NSLog(@"H264: VTCompressionSessionEncodeFrame failed with %d", (int)statusCode);
					error = @"H264: VTCompressionSessionEncodeFrame failed ";
					
					// End the session
					VTCompressionSessionInvalidate(EncodingSession);
					CFRelease(EncodingSession);
					EncodingSession = NULL;
					error = NULL;
					return;
				}
				NSLog(@"H264: VTCompressionSessionEncodeFrame Success");
			});
			
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
	
	mCompressor.reset( new TCompressor(Meta) );
	mPixelMeta = Meta;
}

void Avf::TEncoder::Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta)
{
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__, 2);
	{
		auto YuvFormat = SoyPixelsFormat::GetMergedFormat( Luma.GetFormat(), ChromaU.GetFormat(), ChromaV.GetFormat() );
		auto YuvWidth = Luma.GetWidth();
		auto YuvHeight = Luma.GetHeight();
		SoyPixelsMeta YuvMeta( YuvWidth, YuvHeight, YuvFormat );
		AllocEncoder(YuvMeta);
	}
	
	BufferArray<const SoyPixelsImpl*, 3> Planes;
	Planes.PushBack(&Luma);
	Planes.PushBack(&ChromaU);
	Planes.PushBack(&ChromaV);

	//	checks from example code https://github.com/jesselegg/x264/blob/master/example.c
	//	gr: look for proper validation funcs
	auto Width = Luma.GetWidth();
	auto Height = Luma.GetHeight();
	int LumaSize = Width * Height;
	int ChromaSize = LumaSize / 4;
	int ExpectedBufferSizes[] = { LumaSize, ChromaSize, ChromaSize };
	
	for (auto i = 0; i < Planes.GetSize(); i++)
	{
		auto* OutPlane = mPicture.img.plane[i];
		auto& InPlane = *Planes[i];
		auto& InPlaneArray = InPlane.GetPixelsArray();
		auto OutSize = ExpectedBufferSizes[i];
		auto InSize = InPlaneArray.GetDataSize();
		if (OutSize != InSize)
		{
			std::stringstream Error;
			Error << "Copying plane " << i << " for x264, but plane size mismatch " << InSize << " != " << OutSize;
			throw Soy_AssertException(Error);
		}
		memcpy(OutPlane, InPlaneArray.GetArray(), InSize );
	}
	
	mPicture.i_pts = PushFrameMeta(Meta);
	
	Encode(&mPicture);
	
	//	flush any other frames
	//	gr: this is supposed to only be called at the end of the stream...
	//		if DelayedFrameCount non zero, we may haveto call multiple times before nal size is >0
	//		so just keep calling until we get 0
	//	maybe add a safety iteration check
	//	gr: need this on OSX (latest x264) but on windows (old build) every subsequent frame fails
	//	gr: this was backwards? brew (old 2917) DID need to flush?
	if (X264_REV < 2969)
	{
		//	gr: flushing on OSX (X264_REV 2917) causing
		//	log: x264 [error]: lookahead thread is already stopped
#if !defined(TARGET_OSX)
		{
			//FlushFrames();
		}
#endif
	}
}



void Avf::TEncoder::FinishEncoding()
{
	//	when we're done with frames, we need to make the encoder flush out any more packets
	int Safety = 1000;
	while (--Safety > 0)
	{
		auto DelayedFrameCount = x264_encoder_delayed_frames(mHandle);
		if (DelayedFrameCount == 0)
			break;
		
		Encode(nullptr);
	}
}


void Avf::TEncoder::Encode(x264_picture_t* InputPicture)
{
	//	we're assuming here mPicture has been setup, or we're flushing
	
	//	gr: currently, decoder NEEDS to have nal packets split
	auto OnNalPacket = [&](FixedRemoteArray<uint8_t>& Data)
	{
		Soy::TScopeTimerPrint Timer("OnNalPacket",2);
		auto DecodeOrderNumber = mPicture.i_dts;
		auto FrameNumber = mPicture.i_pts;
		//std::Debug << "OnNalPacket( pts=" << FrameNumber << ", dts=" << DecodeOrderNumber << ")" << std::endl;
		auto FrameMeta = GetFrameMeta(FrameNumber);
		
		//	todo: either store these to make sure decode order (dts) is kept correct
		//		or send DTS order to TPacket for host to order
		//	todo: insert DTS into meta anyway!
		//	gr: DTS is 0 all of the time, I think there's a setting to allow out of order
		PopH264::TPacket OutputPacket;
		OutputPacket.mData.reset(new Array<uint8_t>());
		OutputPacket.mInputMeta = FrameMeta;
		OutputPacket.mData->PushBackArray(Data);
		OnOutputPacket(OutputPacket);
	};
	
	x264_picture_t OutputPicture;
	x264_nal_t* Nals = nullptr;
	int NalCount = 0;
	
	Soy::TScopeTimerPrint EncodeTimer("x264_encoder_encode",10);
	auto FrameSize = x264_encoder_encode(mHandle, &Nals, &NalCount, InputPicture, &OutputPicture);
	EncodeTimer.Stop();
	if (FrameSize < 0)
		throw Soy::AssertException("x264_encoder_encode error");
	
	//	processed, but no data output
	if (FrameSize == 0)
	{
		auto DelayedFrameCount = x264_encoder_delayed_frames(mHandle);
		std::Debug << "x264::Encode processed, but no output; DelayedFrameCount=" << DelayedFrameCount << std::endl;
		return;
	}
	
	//	process each nal
	auto TotalNalSize = 0;
	for (auto n = 0; n < NalCount; n++)
	{
		auto& Nal = Nals[n];
		auto NalSize = Nal.i_payload;
		auto PacketArray = GetRemoteArray(Nal.p_payload, NalSize);
		//	if this throws we lose a packet!
		OnNalPacket(PacketArray);
		TotalNalSize += NalSize;
	}
	if (TotalNalSize != FrameSize)
		throw Soy::AssertException("NALs output size doesn't match frame size");
}


size_t X264::TEncoder::PushFrameMeta(const std::string& Meta)
{
	TFrameMeta FrameMeta;
	FrameMeta.mFrameNumber = mFrameCount;
	FrameMeta.mMeta = Meta;
	mFrameMetas.PushBack(FrameMeta);
	mFrameCount++;
	return FrameMeta.mFrameNumber;
}

std::string X264::TEncoder::GetFrameMeta(size_t FrameNumber)
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
	
void Avf::TEncoder::OnPacketCompressed(const ArrayBridge<uint8_t>& Data,SoyTime PresentationTime)
{
	Soy::TScopeTimerPrint Timer("OnNalPacket",2);
	//auto DecodeOrderNumber = mPicture.i_dts;
	auto FrameNumber = PresentationTime.GetTime();
	
	//std::Debug << "OnNalPacket( pts=" << FrameNumber << ", dts=" << DecodeOrderNumber << ")" << std::endl;
	auto FrameMeta = GetFrameMeta(FrameNumber);
			
	PopH264::TPacket OutputPacket;
	OutputPacket.mData.reset(new Array<uint8_t>());
	OutputPacket.mInputMeta = FrameMeta;
	OutputPacket.mData->PushBackArray(Data);
	OnOutputPacket(OutputPacket);
}

