#include "AndroidDecoder.h"



namespace Android
{
	void			IsOkay(media_status_t Status,const char* Context);
	void			EnumCodecs(std::function<void(const std::string&)> Enum);
	
	constexpr int32_t		Mode_NvidiaSoftware = 1;
	constexpr int32_t		Mode_GoogleSoftware = 2;
	constexpr int32_t		Mode_NvidiaHardware = 3;
	constexpr int32_t		Mode_GoogleHardware = 4;

	constexpr auto*			NvidiaH264Codec = "OMX.Nvidia.h264.decode";	//	hardware
	constexpr auto*			GoogleH264Codec = "OMX.google.h264.decoder";	//	software according to https://forum.magicleap.com/hc/en-us/community/posts/360041748952-Follow-up-on-Multimedia-Decoder-API
	std::string				GetCodec(int32_t Mode,bool& HardwareSurface);

	//	got this mime from googling;
	//	http://hello-qd.blogspot.com/2013/05/choose-decoder-and-encoder-by-google.html
	//	http://twinkfed.homedns.org/Android/reference/android/media/MediaCodec.html#createDecoderByType(java.lang.String)
	const auto*			H264MimeType = "video/avc";
	
	//	CSD-0 (from android mediacodec api)
	//	not in the ML api
	//	https://forum.magicleap.com/hc/en-us/community/posts/360048067552-Setting-csd-0-byte-buffer-using-MLMediaFormatSetKeyByteBuffer
	//MLMediaFormatKey		MLMediaFormat_Key_CSD0 = "csd-0";
}

std::string GetStatusString(media_status_t Status)
{
	auto NameStr = magic_enum::enum_name(Status);
	if ( NameStr.length() )
		return std::string(NameStr);

#define CASE_ERROR(e)	case e: return #e
	switch(Status)
	{
	CASE_ERROR(AMEDIA_OK);
	CASE_ERROR(AMEDIACODEC_ERROR_INSUFFICIENT_RESOURCE);
	CASE_ERROR(AMEDIACODEC_ERROR_RECLAIMED);
	CASE_ERROR(AMEDIA_ERROR_UNKNOWN);
	CASE_ERROR(AMEDIA_ERROR_MALFORMED);
	CASE_ERROR(AMEDIA_ERROR_UNSUPPORTED);
	CASE_ERROR(AMEDIA_ERROR_INVALID_OBJECT);
	CASE_ERROR(AMEDIA_ERROR_INVALID_PARAMETER);
	CASE_ERROR(AMEDIA_ERROR_INVALID_OPERATION);
	CASE_ERROR(AMEDIA_ERROR_END_OF_STREAM);
	CASE_ERROR(AMEDIA_ERROR_IO);
	CASE_ERROR(AMEDIA_ERROR_WOULD_BLOCK);
	CASE_ERROR(AMEDIA_DRM_ERROR_BASE);
	CASE_ERROR(AMEDIA_DRM_NOT_PROVISIONED);
	CASE_ERROR(AMEDIA_DRM_RESOURCE_BUSY);
	CASE_ERROR(AMEDIA_DRM_DEVICE_REVOKED);
	CASE_ERROR(AMEDIA_DRM_SHORT_BUFFER);
	CASE_ERROR(AMEDIA_DRM_SESSION_NOT_OPENED);
	CASE_ERROR(AMEDIA_DRM_TAMPER_DETECTED);
	CASE_ERROR(AMEDIA_DRM_VERIFY_FAILED);
	CASE_ERROR(AMEDIA_DRM_NEED_KEY);
	CASE_ERROR(AMEDIA_DRM_LICENSE_EXPIRED);
	CASE_ERROR(AMEDIA_IMGREADER_ERROR_BASE);
	CASE_ERROR(AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE);
	//CASE_ERROR(AMEDIA_IMGREADER_ERROR_BASE);
	CASE_ERROR(AMEDIA_IMGREADER_MAX_IMAGES_ACQUIRED);
	CASE_ERROR(AMEDIA_IMGREADER_CANNOT_LOCK_IMAGE);
	CASE_ERROR(AMEDIA_IMGREADER_CANNOT_UNLOCK_IMAGE);
	CASE_ERROR(AMEDIA_IMGREADER_IMAGE_NOT_LOCKED);
	}
	
	return "[unknown media_status_t error]";
}

void Android::IsOkay(media_status_t Status,const char* Context)
{
	if ( Status == AMEDIA_OK )
		return;
		
	std::stringstream Error;
	Error << "AMedia error " << Status << "/" << GetStatusString(Status) << " in " << Context;
	throw Soy::AssertException(Error);
}


SoyPixelsMeta Android::GetPixelMeta(MLHandle Format)
{
	Soy_AssertTodo();
/*
	auto GetKey_integer = [&](MLMediaFormatKey Key)
	{
		//MLMediaFormatGetKeyValueInt32
		//MLMediaFormatGetKeyValueInt64
		//MLMediaFormatGetKeyValueFloat
		//	string
		//ML_API MLResult ML_CALL MLMediaFormatGetKeySize(MLHandle handle, MLMediaFormatKey name, size_t *out_size);
		//ML_API MLResult ML_CALL MLMediaFormatGetKeyString(MLHandle handle, MLMediaFormatKey name, char *out_string);
		
		//MLMediaFormatGetKeyByteBuffer
		int32_t Value = 0;
		auto Result = MLMediaFormatGetKeyValueInt32( Format, Key, &Value );
		IsOkay( Result, Key );
		return Value;
	};
	
	auto DebugKey = [&](MLMediaFormatKey Key)
	{
		try
		{
			auto Value = GetKey_integer(Key);
			std::Debug << "Format key " << Key << "=" << Value << std::endl;
		}
		catch(std::exception& e)
		{
			std::Debug << "Format key " << Key << " error " << e.what() << std::endl;
		}
	};
	
	DebugKey( MLMediaFormat_Key_Duration);
	auto Width = GetKey_integer(MLMediaFormat_Key_Width);
	auto Height = GetKey_integer(MLMediaFormat_Key_Height);
	DebugKey( MLMediaFormat_Key_Stride );
	//GetKey<string>(MLMediaFormat_Key_Mime
	DebugKey(MLMediaFormat_Key_Frame_Rate);
	DebugKey(MLMediaFormat_Key_Color_Format);
	auto ColourFormat = GetKey_integer(MLMediaFormat_Key_Color_Format);
	DebugKey(MLMediaFormat_Key_Crop_Left);
	DebugKey(MLMediaFormat_Key_Crop_Right);
	DebugKey(MLMediaFormat_Key_Crop_Bottom);
	DebugKey(MLMediaFormat_Key_Crop_Top);
	//GetKey_long(MLMediaFormat_Key_Repeat_Previous_Frame_After
	
	//	there's a colour format key, but totally undocumented
	//	SDK says for  MLMediaCodecAcquireNextAvailableFrame;
	//		Note: The returned buffer's color format is multi-planar YUV420. Since our
	//		underlying hardware interops do not support multiplanar formats, advanced
	auto PixelFormat = GetPixelFormat(ColourFormat);

	std::Debug << "Format; ";
	std::Debug << " Width=" << Width;
	std::Debug << " Height=" << Height;
	std::Debug << " PixelFormat=" << PixelFormat;
	std::Debug << std::endl;

	SoyPixelsMeta Meta( Width, Height, PixelFormat );
	return Meta;
	*/
}

/*
void MagicLeap::EnumCodecs(std::function<void(const std::string&)> EnumCodec)
{
	uint64_t Count = 0;
	auto Result = MLMediaCodecListCountCodecs(&Count);
	IsOkay( Result, "MLMediaCodecListCountCodecs" );
	
	for ( auto c=0;	c<Count;	c++ )
	{
		try
		{
			char NameBuffer[MAX_CODEC_NAME_LENGTH];
			auto Result = MLMediaCodecListGetCodecName( c, NameBuffer );
			IsOkay( Result, "MLMediaCodecListGetCodecName" );
			//	just in case
			NameBuffer[MAX_CODEC_NAME_LENGTH-1] = '\0';
			EnumCodec( NameBuffer );
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}
}

std::string MagicLeap::GetCodec(int32_t Mode,bool& HardwareSurface)
{
	switch ( Mode )
	{
		case Mode_NvidiaHardware:
		case Mode_GoogleHardware:
			HardwareSurface = true;
			break;
			
		default:
			HardwareSurface = false;
			break;
	}

	switch ( Mode )
	{
		case Mode_NvidiaSoftware:
		case Mode_NvidiaHardware:
			return NvidiaH264Codec;
			
		case Mode_GoogleSoftware:
		case Mode_GoogleHardware:
			return GoogleH264Codec;

		default:
			break;
	}
	std::stringstream Error;
	Error << "Unknown mode(" << Mode << ") for MagicLeap decoder";
	throw Soy::AssertException(Error);
}
*/

Android::TDecoder::TDecoder(std::function<void(const SoyPixelsImpl&,size_t)> OnDecodedFrame) :
	PopH264::TDecoder	( OnDecodedFrame ),
	mInputThread		( std::bind(&TDecoder::GetNextInputData, this, std::placeholders::_1 ), std::bind(&TDecoder::HasPendingData, this ) ),
	mOutputThread		( std::bind(&TDecoder::OnDecodedFrame, this, std::placeholders::_1, std::placeholders::_2 ) )
{
/*
	//	see main thread/same thread comments
	//	http://stackoverflow.com/questions/32772854/android-ndk-crash-in-androidmediacodec#
	//	after getting this working, appears we can start on one thread and iterate on another.
	//	keep this code here in case other phones/media players can't handle it...
	//	gr: we use the deffered alloc so we can wait for the surface to be created
	//		maybe we can mix creating stuff before the configure...
	static bool DefferedAlloc = true;
	bool Params_mDecoderUseHardwareBuffer = false;
	
	SoyPixelsMeta SurfaceMeta;
	if ( Params_mDecoderUseHardwareBuffer )
	{
		Soy_AssertTodo();
		SurfaceMeta = Params.mAssumedTargetTextureMeta;

		if ( !SurfaceMeta.IsValidDimensions() )
		{
			std::stringstream Error;
			Error << "Creating surface texture but params from VideoDecoderParams are invalid" << SurfaceMeta << std::endl;
			throw Soy::AssertException( Error.str() );
		}
	}
	
	if ( DefferedAlloc )
	{
		//	gr: create all this on the same thread as the buffer queue
		auto InvokeAlloc = [=](bool&)
		{
			try
			{
				Alloc( SurfaceMeta, Format, OpenglContext, Params.mAndroidSingleBufferMode );
			}
			catch(std::exception& e)
			{
				//	gr: get extended info here. Sometimes fails (and something then abort()'s the app) with
				//	I/Pop     (12487): pop: Failed to allocate encoder Java exception in void TJniObject::CallVoidMethod(const std::string &, TJniObject &, TJniObject &, TJniObject &, int)configure: android.media.MediaCodec$CodecException: Error 0xffffffea
				//	F/libc    (12487): Fatal signal 6 (SIGABRT), code -6 in tid 13250 (Thread-443)
				//	https://developer.android.com/reference/android/media/MediaCodec.CodecException.html
				std::Debug << "Failed to allocate encoder " << e.what() << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				WaitToFinish();
				std::Debug << "mCodec.reset()... " << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				mCodec.reset();
				std::Debug << "mCodec.reset()... finished "<< std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
			}
		};
		this->mOnStart = InvokeAlloc;
	}
	else
	{
		Alloc( SurfaceMeta, Format, OpenglContext, Params.mAndroidSingleBufferMode );
	}

	//	start thread
	Start();
	*/
}


//	return true if ready, return false if not ready (try again!). Exception on error
bool Android::TDecoder::CreateCodec()
{
	//	codec ready
	if ( mCodec )
		return true;
	
	//	need SPS & PPS 
	if ( mPendingSps.IsEmpty() || mPendingPps.IsEmpty() )
	{
		std::stringstream Error;
		Error << "CreateCodec still waiting for ";
		if ( mPendingSps.IsEmpty() )
			Error << "SPS ";
		if ( mPendingPps.IsEmpty() )
			Error << "PPS ";
		throw Soy::AssertException(Error);
		std::Debug << Error.str() << std::endl;
		return false;
	}	
	
	
	//	gr: magic leap has MLMediaCodecListGetCodecName
	//		but I can't see an android equivelent
	const auto* H264MimeType = "video/avc";
	auto MimeType = H264MimeType;
	
	mCodec = AMediaCodec_createDecoderByType(MimeType);
	if ( !mCodec )
		throw Soy::AssertException("AMediaCodec_createDecoderByType failed");
	
	
	AMediaCodecOnAsyncInputAvailable OnInputAvailible = [](AMediaCodec *codec,
		void *userdata,
		int32_t index)
	{
		std::Debug << "OnInputAvailible" << std::endl;
	};
	
	AMediaCodecOnAsyncOutputAvailable OnOutputAvailible = [](AMediaCodec *codec,
		void *userdata,
		int32_t index,
		AMediaCodecBufferInfo *bufferInfo)
	{
		std::Debug << "OnOutputAvailible" << std::endl;
	};

	AMediaCodecOnAsyncFormatChanged OnFormatChanged = [](AMediaCodec *codec,
		void *userdata,
		AMediaFormat *format)
	{
		std::Debug << "OnFormatChanged" << std::endl;
	};

	AMediaCodecOnAsyncError OnError = [](AMediaCodec *codec,
		void *userdata,
		media_status_t error,
		int32_t actionCode,
		const char *detail)
	{
		std::Debug << "OnError" << std::endl;
	};
	AMediaCodecOnAsyncNotifyCallback Callbacks = {0};
	Callbacks.onAsyncInputAvailable = OnInputAvailible;
	Callbacks.onAsyncOutputAvailable = OnOutputAvailible;
	Callbacks.onAsyncFormatChanged = OnFormatChanged;
	Callbacks.onAsyncError = OnError;
	
	media_status_t Status = AMEDIA_OK;
	#if __ANDROID_API__ >= 28
	Status = AMediaCodec_setAsyncNotifyCallback( mCodec, Callbacks, this );
	IsOkay(Status,"AMediaCodec_setAsyncNotifyCallback");
	#endif 


	//	create format
	//	https://android.googlesource.com/platform/cts/+/fb9023359a546eaa93d7753c0c1af37f8d859111/tests/tests/media/libmediandkjni/native-media-jni.cpp#525
	AMediaFormat* Format = AMediaFormat_new();
	AMediaFormat_setString( Format, AMEDIAFORMAT_KEY_MIME, MimeType );
	
	//AMediaFormat_setBuffer( Format, "CS0", SPS, sizeof(sps));


	//	configure with our new format
	ANativeWindow* Surface = nullptr;
	AMediaCrypto* Crypto = nullptr;
	uint32_t CodecFlags = 0;//AMEDIACODEC_CONFIGURE_FLAG_ENCODE;

/*
PopH264 : pop: int32_t PopH264_GetTestData(const char *, uint8_t *, int32_t) exception: No test data named TestData/Depth.h264
10-16 11:03:01.426 21828 21828 I PopH264 : pop: int32_t PopH264_GetTestData(const char *, uint8_t *, int32_t) exception: No test data named TestData/Depth.h264
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Decoded SequenceParameterSet result=H264SWDEC_STRM_PROCESSED
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Decoded PictureParameterSet result=H264SWDEC_STRM_PROCESSED
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Dropped SupplimentalEnhancementInformation x628
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Decoded Slice_CodedIDRPicture result=H264SWDEC_PIC_RDY
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Decoded picture 96x256^Yuv_8_8_8
10-16 11:03:01.427 21828 21828 I PopH264 : pop: Decoded EndOfStream result=H264SWDEC_STRM_PROCESSED
10-16 11:03:01.428 21828 21828 I PopH264 : pop: virtual bool Android::TDecoder::DecodeNextPacket() Creating codec...
10-16 11:03:01.429 21828 21830 I PopH264 : pop: Renamed thread from Thread--384847392 to AndroidOutputTh: 
10-16 11:03:01.429 21828 21830 I PopH264 : Renamed thread from Thread--383794720 to MagicLeapInputT: 
10-16 11:03:01.429 21828 21829 I PopH264 : pop: 
10-16 11:03:01.432 21828 21832 D CCodec  : allocate(c2.qti.avc.decoder)
10-16 11:03:01.437 21828 21832 I Codec2Client: Available Codec2 services: "default" "software"
10-16 11:03:01.440 21828 21832 I CCodec  : setting up 'default' as default (vendor) store
10-16 11:03:01.443   972  1327 I QC2Interface: Created Interface (c2.qti.avc.decoder)
10-16 11:03:01.446   972  1327 E QC2Prop : SK::kPropInputDelay:0
10-16 11:03:01.446   972  1327 E QC2Prop : SK::kPropOutputDelay:0
10-16 11:03:01.446   972  1327 I QC2Comp : Create: Allocated component[17] for name c2.qti.avc.decoder
10-16 11:03:01.446   972  1327 I QC2CompStore: Created component(c2.qti.avc.decoder) id(17)
10-16 11:03:01.449 21828 21832 I CCodec  : Created component [c2.qti.avc.decoder]
10-16 11:03:01.449 21828 21832 D CCodecConfig: read media type: video/avc
10-16 11:03:01.452 21828 21832 D ReflectedParamUpdater: extent() != 1 for single value type: output.buffers.pool-ids.values
10-16 11:03:01.463 21828 21832 D CCodecConfig: ignoring local param raw.size (0xd2001800) as it is already supported
10-16 11:03:01.463 21828 21832 D CCodecConfig: ignoring local param raw.color (0xd2001809) as it is already supported
10-16 11:03:01.463 21828 21832 D CCodecConfig: ignoring local param raw.hdr-static-info (0xd200180a) as it is already supported
10-16 11:03:01.467   972  1325 E QC2Prop : SK::kPropInputDelay:0
10-16 11:03:01.467   972  1325 E QC2Prop : SK::kPropOutputDelay:0
10-16 11:03:01.468 21828 21832 I CCodecConfig: query failed after returning 17 values (BAD_INDEX)
10-16 11:03:01.469 21828 21832 D CCodecConfig: c2 config diff is Dict {
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::i32 algo.priority.value = -1
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::float algo.rate.value = 4.2039e-44
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 algo.secure-mode.value = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::float coded.frame-rate.value = 30
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.pl.level = 20480
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.pl.profile = 20480
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.matrix = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.primaries = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.range = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.transfer = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 default.color.matrix = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 default.color.primaries = 3
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 default.color.range = 2
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 default.color.transfer = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 input.buffers.max-size.value = 13271040
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 input.delay.value = 4
10-16 11:03:01.469 21828 21832 D CCodecConfig:   string input.media-type.value = "video/avc"
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 output.delay.value = 18
10-16 11:03:01.469 21828 21832 D CCodecConfig:   string output.media-type.value = "video/raw"
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 raw.color.matrix = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 raw.color.primaries = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 raw.color.range = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::u32 raw.color.transfer = 0
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::float raw.hdr-static-info.mastering.blue.x = 1.4013e-45
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::float raw.hdr-static-info.mastering.blue.y = 1.4013e-45
10-16 11:03:01.469 21828 21832 D CCodecConfig:   c2::float raw.hdr-
10-16 11:03:01.472 21828 21832 W ColorUtils: expected specified color aspects (0:0:0:0)
10-16 11:03:01.477 21828 21832 D CCodec  : [c2.qti.avc.decoder] buffers are bound to CCodec for this session
10-16 11:03:01.477 21828 21832 D CCodec  : width is missing, which is required for image/video components.
10-16 11:03:01.478 21828 21831 E MediaCodec: Codec reported err 0xffffffea, actionCode 0, while in state 3
10-16 11:03:01.480 21828 21828 E MediaCodec: configure failed with err 0xffffffea, resetting...
10-16 11:03:01.481   972 21837 I QC2Comp : NOTE: handleReleaseCodec returning: 0 (OK=0)
10-16 11:03:01.482   972  1325 I QC2Comp : NOTE: Release returning: 0 (OK=0)
10-16 11:03:01.482 21828 21838 I hw-BpHwBinder: onLastStrongRef automatically unlinking death recipients
10-16 11:03:01.482   972  1325 I QC2CompStore: Deleting component(c2.qti.avc.decoder) id(17)
10-16 11:03:01.483   972  1325 I QC2Comp : [avcD_17] Deallocated component c2.qti.avc.decoder [id=17]
10-16 11:03:01.487 21828 21832 D CCodec  : allocate(c2.qti.avc.decoder)
10-16 11:03:01.491 21828 21832 I CCodec  : setting up 'default' as default (vendor) store
10-16 11:03:01.492   972  2068 I QC2Interface: Created Interface (c2.qti.avc.decoder)
10-16 11:03:01.494   972  2068 E QC2Prop : SK::kPropInputDelay:0
10-16 11:03:01.494   972  2068 E QC2Prop : SK::kPropOutputDelay:0
10-16 11:03:01.495   972  2068 I QC2Comp : Create: Allocated component[18] for name c2.qti.avc.decoder
10-16 11:03:01.495   972  2068 I QC2CompStore: Created component(c2.qti.avc.decoder) id(18)
10-16 11:03:01.497 21828 21832 I CCodec  : Created component [c2.qti.avc.decoder]
10-16 11:03:01.498 21828 21832 D CCodecConfig: read media type: video/avc
10-16 11:03:01.500 21828 21832 D ReflectedParamUpdater: extent() != 1 for single value type: output.buffers.pool-ids.values
10-16 11:03:01.521 21828 21832 D CCodecConfig: ignoring local param raw.size (0xd2001800) as it is already supported
10-16 11:03:01.521 21828 21832 D CCodecConfig: ignoring local param raw.color (0xd2001809) as it is already supported
10-16 11:03:01.521 21828 21832 D CCodecConfig: ignoring local param raw.hdr-static-info (0xd200180a) as it is already supported
10-16 11:03:01.524   972  1325 E QC2Prop : SK::kPropInputDelay:0
10-16 11:03:01.524   972  1325 E QC2Prop : SK::kPropOutputDelay:0
10-16 11:03:01.525 21828 21832 I CCodecConfig: query failed after returning 17 values (BAD_INDEX)
10-16 11:03:01.526 21828 21832 D CCodecConfig: c2 config diff is Dict {
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::i32 algo.priority.value = -1
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::float algo.rate.value = 4.2039e-44
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 algo.secure-mode.value = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::float coded.frame-rate.value = 30
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.pl.level = 20480
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.pl.profile = 20480
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.matrix = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.primaries = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.range = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 coded.vui.color.transfer = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 default.color.matrix = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 default.color.primaries = 3
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 default.color.range = 2
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 default.color.transfer = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 input.buffers.max-size.value = 13271040
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 input.delay.value = 4
10-16 11:03:01.526 21828 21832 D CCodecConfig:   string input.media-type.value = "video/avc"
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 output.delay.value = 18
10-16 11:03:01.526 21828 21832 D CCodecConfig:   string output.media-type.value = "video/raw"
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 raw.color.matrix = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 raw.color.primaries = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 raw.color.range = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::u32 raw.color.transfer = 0
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::float raw.hdr-static-info.mastering.blue.x = 1.4013e-45
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::float raw.hdr-static-info.mastering.blue.y = 1.4013e-45
10-16 11:03:01.526 21828 21832 D CCodecConfig:   c2::float raw.hdr-
10-16 11:03:01.528 21828 21832 W ColorUtils: expected specified color aspects (0:0:0:0)
10-16 11:03:01.531 21828 21828 E NdkMediaCodec: configure: err(-22), failed with format: AMessage(what = 0x00000000) = {
10-16 11:03:01.531 21828 21828 E NdkMediaCodec:   string mime = "video/avc"
10-16 11:03:01.531 21828 21828 E NdkMediaCodec:   int32_t flags = 0
10-16 11:03:01.531 21828 21828 E NdkMediaCodec: }
10-16 11:03:01.531 21828 21828 E NdkMediaCodec: sf error code: -22
10-16 11:03:01.531 21828 21828 I PopH264 : pop: DecodeNextPacket CreateCodec failed; AMedia error -10000/AMEDIA_ERROR_UNKNOWN in AMediaCodec_configure (assuming we need more data)
10-16 11:03:01.532 21828 21828 I PopH264 : pop: int32_t PopH264_PushData(int32_t, uint8_t *, int32_t, int32_t) exception: todo: void Android::TDecoder::DequeueInputBuffers()
10-16 11:03:01.532 21828 21828 I PopH264 : virtual bool Android::TInputThread::Iteration(std::function<void (std::chrono::milliseconds)>) No input buffers sleep(
10-16 11:03:01.532 21828 21829 I PopH264 : pop: 

*/
	Status = AMediaCodec_configure( mCodec, Format, Surface, Crypto, CodecFlags );
	IsOkay(Status,"AMediaCodec_configure");
/*
	//	configure
	MLHandle Format = ML_INVALID_HANDLE;
	Result = MLMediaFormatCreateVideo( H264MimeType, 1920, 1080, &Format );
	IsOkay( Result, "MLMediaFormatCreateVideo" );
*/
	//	configure with SPS & PPS
	//int8_t CSD_Buffer[]= { 0, 0, 0, 1, 103, 100, 0, 40, -84, 52, -59, 1, -32, 17, 31, 120, 11, 80, 16, 16, 31, 0, 0, 3, 3, -23, 0, 0, -22, 96, -108, 0, 0, 0, 1, 104, -18, 60, -128 };
	//auto* CSD_Bufferu8 = (uint8_t*)CSD_Buffer;
	//MLMediaFormatByteArray CSD_ByteArray {CSD_Bufferu8, sizeof(CSD_Buffer)};
	//MLMediaFormatSetKeyByteBuffer( Format, "csd-0", &CSD_ByteArray);
	/*
	MLMediaFormat_Key_CSD0
	// create media format with the given mime and resolution
	if (MLResult_Ok == MLMediaFormatCreateVideo(mime_type, 1280, 720, &format_handle)) {
		// Fill in other format info
		uint8_t bytebufferMpeg4[] = {0, 0, 1, 0, 0, 0, 1, 32........};
		MLMediaFormatByteArray csd_info {bytebufferMpeg4, sizeof(bytebufferMpeg4)};
		MLMediaFormatSetKeyByteBuffer(format_handle, "csd-0", &csd_info);
		MLMediaFormatSetKeyInt32(format_handle, MLMediaFormat_Key_Max_Input_Size, max_input_size);
		MLMediaFormatSetKeyInt64(format_handle, MLMediaFormat_Key_Duration, duration_us);
	MLMediaFormat_Key_CSD0
	*/
	
	/*
	//	force software mode
	if ( HardwareSurface )
	{
		Result = MLMediaCodecSetSurfaceHint( mHandle, MLMediaCodecSurfaceHint_Hardware );
		IsOkay( Result, "MLMediaCodecSetSurfaceHint(hardware)" );
	}
	else
	{
		Result = MLMediaCodecSetSurfaceHint( mHandle, MLMediaCodecSurfaceHint_Software );
		IsOkay( Result, "MLMediaCodecSetSurfaceHint(software)" );
	}
	
	//MLHandle Crypto = ML_INVALID_HANDLE;
	MLHandle Crypto = 0;	//	gr: INVALID_HANDLE doesnt work
	Result = MLMediaCodecConfigure( mHandle, Format, Crypto );
	IsOkay( Result, "MLMediaCodecConfigure" );
	
	Result = MLMediaCodecStart( mHandle );
	IsOkay( Result, "MLMediaCodecStart" );
	*/
	return true;
}



Android::TDecoder::~TDecoder()
{
/*
	try
	{
		auto Result = MLMediaCodecStop( mHandle );
		IsOkay( Result, "MLMediaCodecStop" );
		
		Result = MLMediaCodecFlush( mHandle );
		IsOkay( Result, "MLMediaCodecFlush" );
		
		Result = MLMediaCodecDestroy( mHandle );
		IsOkay( Result, "MLMediaCodecDestroy" );
	}
	catch(std::exception& e)
	{
		std::Debug << __func__ << e.what() << std::endl;
	}
	*/
}


void Android::TDecoder::DequeueInputBuffers()
{
	//		and stick in the queue
	//		this should trigger the input queue to start grabbing data
	while(true)
	{
		auto TimeoutImmediateReturn = 0;
		auto TimeoutBlock = -1;
		long TimeoutUs = TimeoutImmediateReturn;
			
			Soy_AssertTodo();
			/*
		//	this throws when not started... okay, but maybe cleaner not to. or as it's multithreaded...wait
		int InputBufferId = mCodec->CallIntMethod("dequeueInputBuffer", TimeoutUs );
		if ( InputBufferId < 0 )
		{
			//Java::IsOkay("Failed to get codec input buffer");
			//std::Debug << "Failed to get codec input buffer" << std::endl;
			//return false;
			break;
		}
		OnInputBufferAvailible(InputBufferId);
		*/
	}
}

void Android::TDecoder::DequeueOutputBuffers()
{
/*
	//		and stick in the queue
	//		this should trigger the input queue to start grabbing data
	while(true)
	{
		auto TimeoutImmediateReturn = 0;
		auto TimeoutBlock = -1;
		long TimeoutUs = TimeoutImmediateReturn;
			
		//	this throws when not started... okay, but maybe cleaner not to. or as it's multithreaded...wait
		int InputBufferId = mCodec->CallIntMethod("dequeueOutputBuffer", TimeoutUs );
		if ( InputBufferId < 0 )
		{
			//Java::IsOkay("Failed to get codec input buffer");
			//std::Debug << "Failed to get codec input buffer" << std::endl;
			//return false;
			break;
		}
		OnInputBufferAvailible(InputBufferId);
	}
	*/
}

//	returns true if more data to proccess
bool Android::TDecoder::DecodeNextPacket()
{
	//	we have to wait for input thread to want to pull data here, we can't force it
	mInputThread.Wake();


	//	if we have no codec yet, we won't have any input buffers, so we won't have any requests from the input thread
	//	we need to pull SPS & PPS and create codec
	if ( !mCodec )
	{
		std::Debug << __PRETTY_FUNCTION__<< " Creating codec..." << std::endl;
		try
		{
			//	gr: this will create codec if we're ready to
			//	gr: this will drop this packet if pre sps/pps		
			Array<uint8_t> NaluPacket;
			GetNextInputData( GetArrayBridge(NaluPacket) );
		}
		catch(std::exception& e)
		{
			std::Debug << "DecodeNextPacket CreateCodec failed; " << e.what() << " (assuming we need more data)" << std::endl;
			return false;
		}
	}
	
	//	gr: we don't currently have callbacks for buffers, so deque any that are availible
	DequeueInputBuffers();
	DequeueOutputBuffers();
	
	//	even if we didn't get a frame, try to decode again as we processed a packet
	return true;
}

void Android::TDecoder::GetNextInputData(ArrayBridge<uint8_t>&& PacketBuffer)
{
	//	gr: if no codec, fetch header packets
	//	gr:	make this neater 
	if ( !mCodec )
	{
		PeekHeaderNalus( GetArrayBridge(mPendingSps), GetArrayBridge(mPendingPps) );
		CreateCodec();
	}

	auto& Nalu = PacketBuffer;
	PopH264::FrameNumber_t FrameNumber=0;
	//	input thread wants some data to process
	if (!PopNalu(GetArrayBridge(Nalu),FrameNumber))
	{
		std::stringstream Error;
		Error << "GetNextInputData thread, no nalu ready ";//(" << GetPendingDataSize() <<" bytes ready pending)";
		throw Soy::AssertException(Error);
	}

	//	update header packets
	auto NaluType = H264::GetPacketType(GetArrayBridge(Nalu));
	if (NaluType == H264NaluContent::SequenceParameterSet)
	{
		mPendingSps = Nalu;
	}
	if (NaluType == H264NaluContent::PictureParameterSet)
	{
		mPendingPps = Nalu;
	}

	//	not got enough info (format data) to create codec, dropping packet
	CreateCodec();

	//	reached here without throwing, so data is okay
	//	gr: skip if sps?
}

void Android::TDecoder::OnDecodedFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber)
{
	PopH264::TDecoder::OnDecodedFrame(Pixels,FrameNumber);
}


void Android::TInputThread::PushInputBuffer(int64_t BufferIndex)
{
	Soy_AssertTodo();
	/*
	//	gr: we can grab a buffer without submitted it, so this is okay if we fail here
	std::Debug << "Pushing to input buffer #" << BufferIndex << std::endl;
		
	auto BufferHandle = static_cast<MLHandle>( BufferIndex );
	uint8_t* Buffer = nullptr;
	size_t BufferSize = 0;
	auto Result = MLMediaCodecGetInputBufferPointer( mHandle, BufferHandle, &Buffer, &BufferSize );
	IsOkay( Result, "MLMediaCodecGetInputBufferPointer" );
	if ( Buffer == nullptr )
	{
		std::stringstream Error;
		Error << "MLMediaCodecGetInputBufferPointer gave us null buffer (size=" << BufferSize << ")";
		throw Soy::AssertException(Error);
	}

	//	grab next packet
	//	gr: problem here, if the packet is bigger than the input buffer, we won't have anywhere to put it
	//		current system means this packet is dropped (or could unpop, but then we'll be stuck anyway)
	//	gr: as we can submit an offset, we could LOCK the pending data, submit, then unlock & delete and save a copy
	size_t BufferWrittenSize = 0;
	auto BufferArray = GetRemoteArray( Buffer, BufferSize, BufferWrittenSize );
	try
	{
		mPopPendingData( GetArrayBridge(BufferArray) );
	}

	//	process buffer
	int64_t DataOffset = 0;
	uint64_t PresentationTimeMicroSecs = mPacketCounter;
	mPacketCounter++;

	int Flags = 0;
	//Flags |= MLMediaCodecBufferFlag_KeyFrame;
	//Flags |= MLMediaCodecBufferFlag_CodecConfig;
	//Flags |= MLMediaCodecBufferFlag_EOS;
	Result = MLMediaCodecQueueInputBuffer( mHandle, BufferHandle, DataOffset, BufferWrittenSize, PresentationTimeMicroSecs, Flags );
	IsOkay( Result, "MLMediaCodecQueueInputBuffer" );
	
	OnInputSubmitted( PresentationTimeMicroSecs );

	std::Debug << "MLMediaCodecQueueInputBuffer( BufferIndex=" << BufferIndex << " DataSize=" << BufferWrittenSize << "/" << BufferSize << " presentationtime=" << PresentationTimeMicroSecs << ") success" << std::endl;
	*/
}


void Android::TInputThread::OnInputBufferAvailible(MLHandle CodecHandle,int64_t BufferIndex)
{
	{
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		mHandle = CodecHandle;
		mInputBuffers.PushBack(BufferIndex);
	}
	std::Debug << "OnInputBufferAvailible(" << BufferIndex << ") " << GetDebugState() << std::endl;
	Wake();
}

void Android::TDecoder::OnInputBufferAvailible(int64_t BufferIndex)
{
	MLHandle Handle;
	mInputThread.OnInputBufferAvailible( Handle, BufferIndex );
}

void Android::TDecoder::OnOutputBufferAvailible(int64_t BufferIndex,const MLMediaCodecBufferInfo& BufferMeta)
{
	TOutputBufferMeta Meta;
	Meta.mPixelMeta = mOutputPixelMeta;
	Meta.mBufferIndex = BufferIndex;
	Meta.mMeta = BufferMeta;
	mOutputThread.OnOutputBufferAvailible( BufferIndex, Meta );
	std::Debug << "OnOutputBufferAvailible(" << BufferIndex << ") " << GetDebugState() << mOutputThread.GetDebugState() << std::endl;
}

void Android::TDecoder::OnOutputTextureAvailible()
{
	mOutputThread.OnOutputTextureAvailible();
}

void Android::TDecoder::OnOutputTextureWritten(int64_t PresentationTime)
{
	mOutputThread.OnOutputTextureWritten(PresentationTime);
}

void Android::TDecoder::OnOutputFormatChanged(MLHandle NewFormat)
{
	//	gr: we should do this like a queue for the output thread
	//		for streams that can change format mid-way
	//	todo: make test streams that change format!
	std::Debug << "Getting output format" << std::endl;
	try
	{
		mOutputPixelMeta = GetPixelMeta( NewFormat );
		std::Debug << "New output format is " << mOutputPixelMeta << std::endl;
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " Exception " << e.what() << std::endl;
	}
}




Android::TOutputThread::TOutputThread(std::function<void(const SoyPixelsImpl& Pixels,size_t FrameNumber)> OnDecodedFrame) :
	SoyWorkerThread	("AndroidOutputThread", SoyWorkerWaitMode::Wake ),
	mOnDecodedFrame	( OnDecodedFrame )
{
	Start();
}


void Android::TOutputThread::OnInputSubmitted(int32_t PresentationTime)
{
	//	std::lock_guard<std::mutex> Lock(mPendingPresentationTimesLock);
	//	mPendingPresentationTimes.PushBack( PresentationTime );
}


void Android::TOutputThread::OnOutputBufferAvailible(MLHandle CodecHandle,const TOutputBufferMeta& BufferMeta)
{
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	mOutputBuffers.PushBack( BufferMeta );
	mCodecHandle = CodecHandle;
	Wake();
}

void Android::TOutputThread::OnOutputTextureAvailible()
{
	//	gr: from the forums, there's a suggestion that no calls should be done during a callback
	//		so queue up a grab-a-texture request
	mOutputTexturesAvailible++;
	Wake();
}

void Android::TOutputThread::OnOutputTextureWritten(int64_t PresentationTime)
{
	//	mark texture ready to be output by giving it a time
	//	this func/callback kinda suggests, there can only be one at a time?
	//	because we don't know which texture handle this refers to
	std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
	for ( auto t=0;	t<mOutputTextures.GetSize();	t++ )
	{
		auto& OutputTexture = mOutputTextures[t];
		//	already used
		if ( OutputTexture.mPresentationTime >= 0 )
			continue;
		//	shouldn't occur
		if ( OutputTexture.mPushed )
			continue;
		
		//	mark texture as ready and thread should process it
		OutputTexture.mPresentationTime = PresentationTime;
		std::Debug << "Marked texture " << t << " with time " << PresentationTime << " ready to be pushed" << std::endl;
		Wake();
		return;
	}

	std::stringstream Debug;
	Debug << "OnOutputTextureWritten(" << PresentationTime << ") but 0/" << mOutputTextures.GetSize() << " textures availible to mark written";
	throw Soy::AssertException(Debug);
}


bool Android::TOutputThread::CanSleep()
{
	//	buffers to get
	if ( !mOutputBuffers.IsEmpty() )
		return false;
	
	//	textures to get
	if ( mOutputTexturesAvailible > 0 )
		return false;
	
	if ( IsAnyOutputTextureReady() )
		return false;
	
	return true;
}


bool Android::TInputThread::CanSleep()
{
	if ( mInputBuffers.IsEmpty() )
		return true;
	
	if ( !HasPendingData() )
		return true;
	
	return false;
}

void Android::TOutputThread::RequestOutputTexture()
{
Soy_AssertTodo();
/*
	TOutputTexture OutputTexture;
	{
		//	gr: is this blocking?
		Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__,0);
		auto Result = MLMediaCodecAcquireNextAvailableFrame( mCodecHandle, &OutputTexture.mTextureHandle );
		IsOkay( Result, "MLMediaCodecAcquireNextAvailableFrame");
		//	gr: we DONT get OnFrameRendered (maybe its after we release)
		//		so we DONT know which frame/time this texture is for
		//		but it's ready (i guess)
		//	returns MLResult_UnspecifiedFailure when we request a THIRD texture (so maybe double buffered)
		//	to make current system continue, give it a time
		OutputTexture.mPresentationTime = mOutputTextureCounter++;
	}
	
	{
		std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
		mOutputTextures.PushBack( OutputTexture );
		mOutputTexturesAvailible--;
	}
	std::Debug << "Got new texture 0x" << std::hex << OutputTexture.mTextureHandle << std::dec << "..." << GetDebugState() << std::endl;
	*/
}

vec4x<uint8_t> GetDebugColour(int Index)
{
	vec4x<uint8_t> Colours[] =
	{
		//vec4x<uint8_t>(0,0,0,255),
		vec4x<uint8_t>(255,0,0,255),
		vec4x<uint8_t>(255,255,0,255),
		vec4x<uint8_t>(0,255,0,255),
		vec4x<uint8_t>(0,255,255,255),
		vec4x<uint8_t>(0,0,255,255),
		vec4x<uint8_t>(255,0,255,255),
		//vec4x<uint8_t>(255,255,255,255)
	};
	Index = Index % std::size(Colours);
	return Colours[Index];
}

void Android::TOutputThread::PushOutputTexture(TOutputTexture& OutputTexture)
{
	//	for now, dummy texture, we need to push a handle (or make this readpixels on a gl thread)
	SoyPixels DummyPixels( SoyPixelsMeta( 1, 1, SoyPixelsFormat::RGBA) );
	DummyPixels.SetPixel( 0, 0, GetDebugColour(OutputTexture.mPresentationTime) );

	std::Debug << "PushOutputTexture(0x" << std::hex << OutputTexture.mTextureHandle << std::dec << " time=" << OutputTexture.mPresentationTime << ")" << std::endl;
	
	PushFrame( DummyPixels, OutputTexture.mPresentationTime );
	
	//	gr: temp, we've "delivered" this texture, so release it now
	std::Debug << "ReleaseOutputTexture(0x" << std::hex << OutputTexture.mTextureHandle << std::dec << " time=" << OutputTexture.mPresentationTime << ")" << std::endl;
	ReleaseOutputTexture( OutputTexture.mTextureHandle );
}

bool Android::TOutputThread::IsAnyOutputTextureReady()
{
	std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
	for ( auto t=0;	t<mOutputTextures.GetSize();	t++ )
	{
		auto& OutputTexture = mOutputTextures[t];
		if ( OutputTexture.IsReadyToBePushed() )
			return true;
	}
	return false;	
}

void Android::TOutputThread::PushOutputTextures()
{
	if ( mOutputTextures.GetSize() == 0 )
		return;
	
	
	std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
	for ( auto t=0;	t<mOutputTextures.GetSize();	t++ )
	{
		auto& OutputTexture = mOutputTextures[t];
		if ( !OutputTexture.IsReadyToBePushed() )
			continue;
		
		PushOutputTexture( OutputTexture );
		OutputTexture.mPushed = true;
	}
	
}

void Android::TOutputThread::ReleaseOutputTexture(MLHandle TextureHandle)
{
	//	remove it from the list
	std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
	if ( !mOutputTextures.Remove( TextureHandle ) )
	{
		std::Debug << "Warning: ReleaseOutputTexture(" << std::hex << "0x" << TextureHandle << std::dec << ") wasn't in our texture output list. MLMediaCodecReleaseFrame() NOT CALLED." << std::endl;
		return;
	}

Soy_AssertTodo();
/*
	//	release it
	auto Result = MLMediaCodecReleaseFrame( mCodecHandle, TextureHandle );
	IsOkay( Result, "MLMediaCodecReleaseFrame" );
	*/
}

void Android::TOutputThread::PopOutputBuffer(const TOutputBufferMeta& BufferMeta)
{
/*
	{
		std::lock_guard<std::mutex> Lock(mPushFunctionsLock);
		if ( mPushFunctions.IsEmpty() )
		{
			std::stringstream Error;
			Error << "PopOutputBuffer(" << BufferMeta.mBufferIndex << ") but no push funcs yet (probably race condition)";
			throw Soy::AssertException( Error );
		}
	}
	*/
	//	gr: hold onto pointer and don't release buffer until it's been read,to avoid a copy
	//		did this on android and it was a boost
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__,0);
	auto BufferHandle = static_cast<MLHandle>( BufferMeta.mBufferIndex );

	auto ReleaseBuffer = [&](bool Render=true)
	{
	Soy_AssertTodo();
	/*
		//	release back!
		auto Result = MLMediaCodecReleaseOutputBuffer( mCodecHandle, BufferHandle, Render );
		IsOkay( Result, "MLMediaCodecReleaseOutputBuffer");
	*/
	};

	const uint8_t* Data = nullptr;
	size_t DataSize = -1;
	/*
	MLMediaCodecBufferInfo BufferMeta;
	int16_t Timeout = 0;
	int64_t NewBufferIndex = -1;
	auto Result = MLMediaCodecDequeueOutputBuffer( mCodecHandle, &BufferMeta, Timeout, &NewBufferIndex );
	IsOkay( Result, "MLMediaCodecDequeueOutputBuffer");
	std::Debug << "MLMediaCodecDequeueOutputBuffer returned buffer index " << NewBufferIndex << " compared to " << OutputBufferIndex << " time=" << BufferMeta.presentation_time_us << std::endl;
	*/
	
	Soy_AssertTodo();
	/*
	auto Result = MLMediaCodecGetOutputBufferPointer( mCodecHandle, BufferHandle, &Data, &DataSize );
	
	//	gr: if we're in hardware mode, this DOES NOT return null/0 like docs say, but instead invalid operation
	//		we release it
	//	https://forum.magicleap.com/hc/en-us/community/posts/360055134771-Stagefright-assert-after-calling-MLMediaCodecDequeueInputBuffer?page=1#community_comment_360008514291
	//	gr: we should probably store that we're in hardware mode...
	if ( Result == MLMediaGenericResult_InvalidOperation )
	{
		std::Debug << "MLMediaCodecGetOutputBufferPointer returned MLMediaGenericResult_InvalidOperation, assuming hardware" << std::endl;
		//	flush this frame (render=true) to get a frame-availible callback
		ReleaseBuffer(true);
		return;
	}
	
	IsOkay( Result, "MLMediaCodecGetOutputBufferPointer");
	
	
	//	if data is null, then output is a surface
	if ( Data == nullptr || DataSize == 0 )
	{
		std::Debug << "Got Invalid OutputBuffer(" << BufferMeta.mBufferIndex << ") DataSize=" << DataSize << " DataPtr=0x" << std::hex << (size_t)(Data) << std::dec << std::endl;
		ReleaseBuffer();
		return;
	}
	
	std::Debug << "Got OutputBuffer(" << BufferMeta.mBufferIndex << ") DataSize=" << DataSize << " DataPtr=0x" << std::hex << (size_t)(Data) << std::dec << std::endl;
	try
	{
		//	output pixels!
		auto* DataMutable = const_cast<uint8_t*>( Data );
		SoyPixelsRemote NewPixels( DataMutable, DataSize, BufferMeta.mPixelMeta );
		PushFrame( NewPixels );
		ReleaseBuffer();
	}
	catch(std::exception& e)
	{
		//	rethrow but make sure we return the buffer first
		ReleaseBuffer();
		throw;
	}
	*/
}


void Android::TOutputThread::PushFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber)
{
	mOnDecodedFrame( Pixels, FrameNumber );
}

bool Android::TOutputThread::Iteration()
{
	//	flush any texture requests
	if ( mOutputTexturesAvailible > 0 )
	{
		try
		{
			RequestOutputTexture();
		}
		catch(std::exception& e)
		{
			std::Debug << "Exception requesting output texture x" << mOutputTexturesAvailible << "availible; " << e.what() << GetDebugState() << std::endl;
			std::this_thread::sleep_for( std::chrono::seconds(1) );
		}
	}
	
	//	push any textures that have been written
	if ( mOutputTextures.GetSize() > 0 )
	{
		PushOutputTextures();
	}
	
	if ( !mOutputBuffers.IsEmpty() )
	{
		//	read a buffer
		TOutputBufferMeta BufferMeta;
		{
			std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
			BufferMeta = mOutputBuffers.PopAt(0);
		}
		try
		{
			PopOutputBuffer( BufferMeta );
		}
		catch(std::exception& e)
		{
			std::Debug << "Exception getting output buffer " << BufferMeta.mBufferIndex << "; " << e.what() << GetDebugState() << std::endl;
			std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
			auto& ElementZero = *mOutputBuffers.InsertBlock(0,1);
			ElementZero = BufferMeta;
			std::this_thread::sleep_for( std::chrono::seconds(1) );
		}
	}
	
	return true;
}


Android::TInputThread::TInputThread(std::function<void(ArrayBridge<uint8_t>&&)> PopPendingData,std::function<bool()> HasPendingData) :
	mPopPendingData	( PopPendingData ),
	mHasPendingData	( HasPendingData ),
	SoyWorkerThread	("MagicLeapInputThread", SoyWorkerWaitMode::Wake )
{
	Start();
}

auto InputThreadNotReadySleep = 3000;//12;
auto InputThreadThrottle = 2;
auto InputThreadErrorThrottle = 1000;

bool Android::TInputThread::Iteration(std::function<void(std::chrono::milliseconds)> Sleep)
{
	if ( mInputBuffers.IsEmpty() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " No input buffers sleep(" << InputThreadNotReadySleep << ")" << std::endl;
		//Sleep( std::chrono::milliseconds(InputThreadNotReadySleep) );
		std::this_thread::sleep_for(std::chrono::milliseconds(InputThreadNotReadySleep) );
		return true;
	}

	if ( !HasPendingData() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " No pending data" << std::endl;
		Sleep( std::chrono::milliseconds(InputThreadNotReadySleep) );
		return true;
	}
	
	std::Debug << __PRETTY_FUNCTION__ << " Reading a buffer" << std::endl;

	//	read a buffer
	int64_t BufferIndex = -1;
	{
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		BufferIndex = mInputBuffers.PopAt(0);
	}
	try
	{
		PushInputBuffer( BufferIndex );
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " Exception pushing input buffer " << BufferIndex << "; " << e.what() << GetDebugState() << std::endl;
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		auto& ElementZero = *mInputBuffers.InsertBlock(0,1);
		ElementZero = BufferIndex;
		Sleep( std::chrono::milliseconds(InputThreadErrorThrottle) );
	}
	
	//	throttle the thread
	Sleep( std::chrono::milliseconds(InputThreadThrottle) );
	
	return true;
}



std::string Android::TDecoder::GetDebugState()
{
	std::stringstream Debug;
	Debug << mInputThread.GetDebugState() << mOutputThread.GetDebugState();
	return Debug.str();
}


std::string Android::TInputThread::GetDebugState()
{
	std::lock_guard<std::mutex> Lock(mInputBuffersLock);
	
	std::stringstream Debug;
	Debug << " mInputBuffers[";
	for ( auto i=0;	i<mInputBuffers.GetSize();	i++ )
		Debug << mInputBuffers[i] << ",";
	Debug << "] ";
	return Debug.str();
}

std::string Android::TOutputThread::GetDebugState()
{
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	std::lock_guard<std::recursive_mutex> Lock2(mOutputTexturesLock);

	std::stringstream Debug;
	Debug << " mOutputBuffers[";
	for ( auto i=0;	i<mOutputBuffers.GetSize();	i++ )
		Debug << mOutputBuffers[i].mBufferIndex << ",";
	Debug << "] ";
	Debug << "mOutputTexturesAvailible=" << mOutputTexturesAvailible << " ";
	Debug << "mOutputTextures[";
	for ( auto i=0;	i<mOutputTextures.GetSize();	i++ )
		Debug << std::hex << "0x" << mOutputTextures[i].mTextureHandle << std::dec << ",";
	Debug << "] ";

	return Debug.str();
}