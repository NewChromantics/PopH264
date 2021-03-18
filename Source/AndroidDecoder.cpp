#include "AndroidDecoder.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#pragma message "Building with android API level " STR(__ANDROID_API__)


#include "TDecoder.h"
#include "SoyPixels.h"
#include "SoyMedia.h"	//	TPixelBuffer
#include "media/NdkImage.h"

#include "json11.hpp"

enum AndroidColourFormat
{
	//	http://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities.html
	COLOR_Format12bitRGB444	= 3,
	COLOR_Format16bitARGB1555 = 5,
	COLOR_Format16bitARGB4444 = 4,
	COLOR_Format16bitBGR565 = 7,
	COLOR_Format16bitRGB565 = 6,
	COLOR_Format18BitBGR666 = 41,
	COLOR_Format18bitARGB1665 = 9,
	COLOR_Format18bitRGB666 = 8,
	COLOR_Format19bitARGB1666 = 10,
	COLOR_Format24BitABGR6666 = 43,
	COLOR_Format24BitARGB6666 = 42,
	COLOR_Format24bitARGB1887 = 13,
	COLOR_Format24bitBGR888 = 12,
	COLOR_Format24bitRGB888 = 11,
	COLOR_Format32bitABGR8888 = 0x7f00a000,
	COLOR_Format32bitARGB8888 = 16,
	COLOR_Format32bitBGRA8888 = 15,
	COLOR_Format8bitRGB332 = 2,
	COLOR_FormatCbYCrY = 27,
	COLOR_FormatCrYCbY = 28,
	COLOR_FormatL16 = 36,
	COLOR_FormatL2 = 33,
	COLOR_FormatL32 = 38,
	COLOR_FormatL4 = 34,
	COLOR_FormatL8 = 35,
	COLOR_FormatMonochrome = 1,
	COLOR_FormatRGBAFlexible = 0x7f36a888,
	COLOR_FormatRGBFlexible = 0x7f36b888,
	COLOR_FormatRawBayer10bit = 31,
	COLOR_FormatRawBayer8bit = 30,
	COLOR_FormatRawBayer8bitcompressed = 32,
	COLOR_FormatSurface = 0x7f000789,
	COLOR_FormatYCbYCr = 25,
	COLOR_FormatYCrYCb = 26,
	COLOR_FormatYUV411PackedPlanar = 18,
	COLOR_FormatYUV411Planar = 17,
	COLOR_FormatYUV420Flexible = 0x7f420888,
	COLOR_FormatYUV420PackedPlanar = 20,
	COLOR_FormatYUV420PackedSemiPlanar = 39,
	COLOR_FormatYUV420Planar = 19,
	COLOR_FormatYUV420SemiPlanar = 21,
	COLOR_FormatYUV422Flexible = 0x7f422888,
	COLOR_FormatYUV422PackedPlanar = 23,
	COLOR_FormatYUV422PackedSemiPlanar = 40,
	COLOR_FormatYUV422Planar = 22,
	COLOR_FormatYUV422SemiPlanar = 24,
	COLOR_FormatYUV444Flexible = 0x7f444888,
	COLOR_FormatYUV444Interleaved = 29,
	COLOR_QCOM_FormatYUV420SemiPlanar = 0x7fa30c00,
	COLOR_TI_FormatYUV420PackedSemiPlanar = 0x7f000100,
	
	//	gr: mystery format (when using surface texture)
	COLOR_FORMAT_UNKNOWN_MAYBE_SURFACE = 261,
	
	//	note4 is giving us a pixel format of this, even when we have a surface texture. Renders okay.
	//	gr; in non-opengl mode, the colour space is broken!
	//		it is not the same as Yuv_8_8_8 (but close). The name is a big hint of this.
	//		probably more like Yuv_8_88 but line by line.
	OMX_QCOM_COLOR_FormatYVU420SemiPlanarInterlace = 0x7FA30C04,	//	2141391876
};


auto InputThreadNotReadySleep = 1000;//12;
auto InputThreadThrottle = 2;
auto InputThreadErrorThrottle = 1000;
auto OutputThreadErrorSleepMs = 500;


namespace Android
{
	void					IsOkay(media_status_t Status,const char* Context);
	void					EnumCodecs(std::function<void(const std::string&)> Enum);
	SoyPixelsMeta			GetPixelMeta(MediaFormat_t Format,bool VerboseDebug,json11::Json::object& Meta);
	SoyPixelsFormat::Type	GetPixelFormat(int32_t ColourFormat);

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

    void                    ResolveSymbols(Soy::TRuntimeLibrary& Dll);
#if __ANDROID_API__ == 28
    const char*             AMEDIAFORMAT_KEY_CSD_AVC = AMEDIAFORMAT_KEY_CSD;
#elif __ANDROID_API__ < 28
    const char*             AMEDIAFORMAT_KEY_CSD_AVC = "csd-0";
    const char*             AMEDIAFORMAT_KEY_DISPLAY_WIDTH = "display-width";
    const char*             AMEDIAFORMAT_KEY_DISPLAY_HEIGHT = "display-height";
    const char*             AMEDIAFORMAT_KEY_ROTATION = "rotation-degrees";
    const char*             AMEDIAFORMAT_KEY_DISPLAY_CROP = "crop";
    const char*             AMEDIAFORMAT_KEY_COLOR_RANGE = "2";

    std::function<bool (AMediaFormat*, const char *name, int32_t *left, int32_t *top, int32_t *right, int32_t *bottom)> AMediaFormat_getRect =
    [](AMediaFormat*, const char *name, int32_t *left, int32_t *top, int32_t *right, int32_t *bottom)
    {
        std::Debug << "AMediaFormat_getRect missing on this platform " << std::endl;
        return false;
    };

    std::function<media_status_t (AMediaCodec*, AMediaCodecOnAsyncNotifyCallback callback, void *userdata)> AMediaCodec_setAsyncNotifyCallback =
    [](AMediaCodec*, AMediaCodecOnAsyncNotifyCallback callback, void *userdata)
    {
        std::Debug << "AMediaCodecOnAsyncNotifyCallback missing on this platform " << std::endl;
        return AMEDIA_ERROR_UNSUPPORTED;
    };
#endif

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

void Android::ResolveSymbols(Soy::TRuntimeLibrary& Dll)
{
    try {
        if ( !AMEDIAFORMAT_KEY_CSD_AVC )
            AMEDIAFORMAT_KEY_CSD_AVC = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_CSD");
        
        if ( !AMEDIAFORMAT_KEY_DISPLAY_WIDTH )
            AMEDIAFORMAT_KEY_DISPLAY_WIDTH = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_DISPLAY_WIDTH");
        
        if ( !AMEDIAFORMAT_KEY_DISPLAY_HEIGHT )
            AMEDIAFORMAT_KEY_DISPLAY_HEIGHT = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_DISPLAY_HEIGHT");
        
        if ( !AMEDIAFORMAT_KEY_ROTATION )
            AMEDIAFORMAT_KEY_ROTATION = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_ROTATION");
        
        if ( !AMEDIAFORMAT_KEY_DISPLAY_CROP )
            AMEDIAFORMAT_KEY_DISPLAY_CROP = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_DISPLAY_CROP");
        
        if ( !AMEDIAFORMAT_KEY_COLOR_RANGE )
            AMEDIAFORMAT_KEY_COLOR_RANGE = (const char*)Dll.GetSymbol("AMEDIAFORMAT_KEY_COLOR_RANGE");
        
#if __ANDROID_API__ < 28
        Dll.SetFunction(AMediaFormat_getRect,"AMediaFormat_getRect");
        Dll.SetFunction(AMediaCodec_setAsyncNotifyCallback, "AMediaCodec_setAsyncNotifyCallback");
#endif

    } catch (std::exception& e) {
        std::Debug << e.what() << std::endl;
    }
}

SoyPixelsFormat::Type Android::GetPixelFormat(int32_t ColourFormat)
{
	switch(ColourFormat)
	{
	case AIMAGE_FORMAT_RGBA_8888:	return SoyPixelsFormat::RGBA;
	case AIMAGE_FORMAT_RGBX_8888:	return SoyPixelsFormat::RGBA;
	case AIMAGE_FORMAT_RGB_888:		return SoyPixelsFormat::RGB;
	//AIMAGE_FORMAT_RGB_565
	//case AIMAGE_FORMAT_RGBA_FP16:	return SoyPixelsFormat::HalfFloat4;
	case COLOR_FormatYUV420Planar:	return SoyPixelsFormat::Yuv_8_8_8;
	case COLOR_FormatYUV420SemiPlanar:	return SoyPixelsFormat::Yuv_8_88;
	case AIMAGE_FORMAT_YUV_420_888:	return SoyPixelsFormat::Yuv_8_8_8;
	//AIMAGE_FORMAT_JPEG
	case AIMAGE_FORMAT_RAW16:		return SoyPixelsFormat::Depth16mm;
	//	AIMAGE_FORMAT_RAW_PRIVATE	arbritry format, 8bit?
	//	AIMAGE_FORMAT_RAW10
	//	AIMAGE_FORMAT_RAW12
	case AIMAGE_FORMAT_DEPTH16:		return SoyPixelsFormat::Depth16mm;
	case AIMAGE_FORMAT_DEPTH_POINT_CLOUD:	return SoyPixelsFormat::Float4;
	//	AIMAGE_FORMAT_PRIVATE
	case AIMAGE_FORMAT_Y8:			return SoyPixelsFormat::Greyscale;
	//	AIMAGE_FORMAT_HEIC
	//	AIMAGE_FORMAT_DEPTH_JPEG
	//case COLOR_FormatYUV420Flexible:	//	could be one of many 420s
	};
	
	auto AndColourFormat = (AndroidColourFormat)ColourFormat;
	std::stringstream Error;
	Error << "Unhandled colour format " << ColourFormat << "/" << magic_enum::enum_name(AndColourFormat);
	throw Soy::AssertException(Error);
}

SoyPixelsMeta Android::GetPixelMeta(MediaFormat_t Format,bool VerboseDebug,json11::Json::object& Meta)
{
	if ( VerboseDebug )
	{
		auto FormatDebugString = AMediaFormat_toString(Format);
		if ( !FormatDebugString )
			FormatDebugString = "<null>";
		std::Debug << __PRETTY_FUNCTION__ << " format debug from android: " << FormatDebugString << std::endl; 
	}

	typedef const char* MLMediaFormatKey;
	auto GetKey_integer = [&](MLMediaFormatKey Key)
	{
		int32_t Value = 0;
		auto Result = AMediaFormat_getInt32( Format, Key, &Value );
		if ( !Result )
			throw Soy::AssertException( std::string("Failed to get MediaFormat key ") + Key );
		//auto Result = MLMediaFormatGetKeyValueInt32( Format, Key, &Value );
		//IsOkay( Result, Key );
		return Value;
	};
	auto GetKey_rect = [&](MLMediaFormatKey Key)
	{
		int32_t Left,Top,Right,Bottom;
		auto Result = AMediaFormat_getRect( Format, Key, &Left, &Top, &Right, &Bottom );
		if ( !Result )
			throw Soy::AssertException( std::string("Failed to get MediaFormat key ") + Key );
		//auto Result = MLMediaFormatGetKeyValueInt32( Format, Key, &Value );
		//IsOkay( Result, Key );
		Soy::Rectx<int> Rect( Left, Top, Right-Left, Bottom-Top);
		return Rect;
	};
	/*
	auto GetKey_String = [&](MLMediaFormat Key)
	{
		if ( !AMediaFormat_getString( Foramt, 
	};
	*/
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
	
	auto DebugKey_Rect = [&](MLMediaFormatKey Key)
	{
		try
		{
			auto Value = GetKey_rect(Key);
			std::Debug << "Format key (rect) " << Key << "=" << Value << std::endl;
		}
		catch(std::exception& e)
		{
			std::Debug << "Format key (rect) " << Key << " error " << e.what() << std::endl;
		}
	};
	
	auto MLMediaFormat_Key_Duration = AMEDIAFORMAT_KEY_DURATION;
	auto MLMediaFormat_Key_Width = AMEDIAFORMAT_KEY_WIDTH;
	auto MLMediaFormat_Key_Height = AMEDIAFORMAT_KEY_HEIGHT;
	auto MLMediaFormat_Key_Stride = AMEDIAFORMAT_KEY_STRIDE;
	auto MLMediaFormat_Key_Mime = AMEDIAFORMAT_KEY_MIME;
	auto MLMediaFormat_Key_Frame_Rate = AMEDIAFORMAT_KEY_FRAME_RATE;
	auto MLMediaFormat_Key_Color_Format = AMEDIAFORMAT_KEY_COLOR_FORMAT;
	//auto MLMediaFormat_Key_Crop_Left = AMEDIAFORMAT_KEY_FRAME_RATE;
	//auto MLMediaFormat_Key_Crop_Right = AMEDIAFORMAT_KEY_FRAME_RATE;
	//auto MLMediaFormat_Key_Crop_Bottom = AMEDIAFORMAT_KEY_FRAME_RATE;
	//auto MLMediaFormat_Key_Crop_Top = AMEDIAFORMAT_KEY_FRAME_RATE;
	auto Key_ChannelCount = AMEDIAFORMAT_KEY_CHANNEL_COUNT;
	auto Key_ColourRange = AMEDIAFORMAT_KEY_COLOR_RANGE;
	auto Key_Rotation = AMEDIAFORMAT_KEY_ROTATION;
	
	auto Width = GetKey_integer(MLMediaFormat_Key_Width);
	auto Height = GetKey_integer(MLMediaFormat_Key_Height);
	auto ColourFormat = GetKey_integer(MLMediaFormat_Key_Color_Format);
	if ( VerboseDebug )
	{
        DebugKey( AMEDIAFORMAT_KEY_CSD_AVC );
		DebugKey( MLMediaFormat_Key_Duration);
		DebugKey( MLMediaFormat_Key_Stride );
		//GetKey<string>(MLMediaFormat_Key_Mime
		DebugKey(MLMediaFormat_Key_Frame_Rate);
		DebugKey(MLMediaFormat_Key_Color_Format);
		DebugKey(Key_ColourRange);
        DebugKey(Key_Rotation);
		DebugKey(MLMediaFormat_Key_Width);
		DebugKey(MLMediaFormat_Key_Height);
		DebugKey(MLMediaFormat_Key_Color_Format);
        DebugKey(AMEDIAFORMAT_KEY_DISPLAY_WIDTH);
        DebugKey(AMEDIAFORMAT_KEY_DISPLAY_HEIGHT);
        DebugKey(AMEDIAFORMAT_KEY_DISPLAY_CROP);
		//DebugKey(MLMediaFormat_Key_Crop_Left);
		//DebugKey(MLMediaFormat_Key_Crop_Right);
		//DebugKey(MLMediaFormat_Key_Crop_Bottom);
		//DebugKey(MLMediaFormat_Key_Crop_Top);
		//GetKey_long(MLMediaFormat_Key_Repeat_Previous_Frame_After
	}
	
	//	set metas
	//	try and set crop rect from display w/h first (older system than crop)
	try
	{
		auto DisplayWidth = GetKey_integer(AMEDIAFORMAT_KEY_DISPLAY_WIDTH);
		auto DisplayHeight = GetKey_integer(AMEDIAFORMAT_KEY_DISPLAY_HEIGHT);
		json11::Json::array RectArray;
		RectArray.push_back(0);
		RectArray.push_back(0);
		RectArray.push_back(DisplayWidth);
		RectArray.push_back(DisplayHeight);
		Meta["ImageRect"] = RectArray;
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception getting meta ImageRect (display w/h): " << e.what() << std::endl;
	}

	//	then try and get real crop rect and overwrite previous setting	
	try
	{
		auto Rect = GetKey_rect(AMEDIAFORMAT_KEY_DISPLAY_CROP);
		json11::Json::array RectArray;
		RectArray.push_back(Rect.x);
		RectArray.push_back(Rect.y);
		RectArray.push_back(Rect.w);
		RectArray.push_back(Rect.h);
		Meta["ImageRect"] = RectArray;
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception getting meta ImageRect (display crop): " << e.what() << std::endl;
	}
		
	//	there's a colour format key, but totally undocumented
	//	SDK says for  MLMediaCodecAcquireNextAvailableFrame;
	//		Note: The returned buffer's color format is multi-planar YUV420. Since our
	//		underlying hardware interops do not support multiplanar formats, advanced
	auto PixelFormat = GetPixelFormat(ColourFormat);

	if ( VerboseDebug )
	{	
		std::Debug << "Format; ";
		std::Debug << " Width=" << Width;	
		std::Debug << " Height=" << Height;
		std::Debug << " PixelFormat=" << PixelFormat;
		std::Debug << std::endl;
	}
	SoyPixelsMeta PixelMeta( Width, Height, PixelFormat );
	return PixelMeta;
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


Android::TDecoder::TDecoder(PopH264::TDecoderParams Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	PopH264::TDecoder	( OnDecodedFrame, OnFrameError ),
	mParams				( Params ),
	mInputThread		( std::bind(&TDecoder::GetNextInputData, this, std::placeholders::_1, std::placeholders::_2 ), std::bind(&TDecoder::HasPendingData, this ) ),
	mOutputThread		( OnDecodedFrame, OnFrameError )
{
    
#if __ANDROID_API__ < 28
    mMediaCodecDll.reset( new Soy::TRuntimeLibrary("libmediandk.so") );
    Android::ResolveSymbols(*mMediaCodecDll);
#endif

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
void Android::TDecoder::CreateCodec()
{
	//	codec ready
	if ( mCodec )
		return;

	//	fetch header packets	
	PeekHeaderNalus( GetArrayBridge(mPendingSps), GetArrayBridge(mPendingPps) );
	
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
		auto& This = *reinterpret_cast<TDecoder*>(userdata);
		//std::Debug << "OnInputAvailible(" << index << ")" << std::endl;
		This.OnInputBufferAvailible(index);
	};
	
	AMediaCodecOnAsyncOutputAvailable OnOutputAvailible = [](AMediaCodec *codec,
		void *userdata,
		int32_t index,
		AMediaCodecBufferInfo *bufferInfo)
	{
		if ( !userdata )
		{
			std::Debug << "OnOutputAvailible null this" << std::endl;
			return;
		}
		if ( !bufferInfo )
		{
			std::Debug << "OnOutputAvailible null bufferinfo (index=" << index << ")" << std::endl;
			return;
		}
		
		auto& This = *reinterpret_cast<TDecoder*>(userdata);
		try
		{
			if ( This.mParams.mVerboseDebug )
				std::Debug << "OnOutputAvailible callback(" << index << ")..." << std::endl;
			This.OnOutputBufferAvailible(index,*bufferInfo);
		}
		catch(std::exception& e)
		{
			std::Debug << "OnOutputAvailible callback(" << index << ") exception; " << e.what() << std::endl;
		}
	};

	AMediaCodecOnAsyncFormatChanged OnFormatChanged = [](AMediaCodec *codec,
		void *userdata,
		AMediaFormat *format)
	{
		if ( !userdata )
		{
			std::Debug << "OnFormatChanged null this" << std::endl;
			return;
		}
		if ( !format )
		{
			std::Debug << "OnFormatChanged null format" << std::endl;
			return;
		}
		auto& This = *reinterpret_cast<TDecoder*>(userdata);
		This.OnOutputFormatChanged(format);
	};

	AMediaCodecOnAsyncError OnError = [](AMediaCodec *codec,
		void *userdata,
		media_status_t error,
		int32_t actionCode,
		const char *detail)
	{
		std::stringstream Error;
		if ( !detail )
			detail = "<null>";
		Error << "Async Error( " << GetStatusString(error) << ", actionCode=" << actionCode << ", detail=" << detail << ")";
		std::Debug << Error.str() << std::endl;
		if ( userdata )
		{
			auto& This = *reinterpret_cast<TDecoder*>(userdata);
			This.OnDecoderError(Error.str());
		}
	};
	AMediaCodecOnAsyncNotifyCallback Callbacks = {0};
	Callbacks.onAsyncInputAvailable = OnInputAvailible;
	Callbacks.onAsyncOutputAvailable = OnOutputAvailible;
	Callbacks.onAsyncFormatChanged = OnFormatChanged;
	Callbacks.onAsyncError = OnError;


	media_status_t Status = AMEDIA_OK;

	Status = AMediaCodec_setAsyncNotifyCallback( mCodec, Callbacks, this );
	IsOkay(Status,"AMediaCodec_setAsyncNotifyCallback");
	mAsyncBuffers = true;

	auto Width = mParams.mWidthHint;
	auto Height = mParams.mHeightHint;
	auto InputSize = mParams.mInputSizeHint;
	
	//	create format
	//	https://android.googlesource.com/platform/cts/+/fb9023359a546eaa93d7753c0c1af37f8d859111/tests/tests/media/libmediandkjni/native-media-jni.cpp#525
	AMediaFormat* Format = AMediaFormat_new();
	AMediaFormat_setString( Format, AMEDIAFORMAT_KEY_MIME, MimeType );

	//	if no width/height hint provided, try and extract from SPS
	if ( !Width || !Height )
	{
		try
		{
			auto Sps = H264::ParseSps( GetArrayBridge(mPendingSps) );
			std::Debug << "Extracted size " << Sps.mWidth << "x" << Sps.mHeight << " from sps (frame_crop_left_offset=" << Sps.frame_crop_left_offset << " frame_crop_right_offset=" << Sps.frame_crop_right_offset << " frame_crop_top_offset=" << Sps.frame_crop_top_offset << " frame_crop_bottom_offset=" << Sps.frame_crop_bottom_offset << std::endl;
			Width = Sps.mWidth;
			Height = Sps.mHeight;
		}
		catch(std::exception& e)
		{
			std::Debug << "Exception extracting SPS width & height; " << e.what() << std::endl;
		}
	}
	
	std::Debug << "Setting MediaFormat hints; (0=skipped) Width=" << Width << " Height=" << Height << " InputSize=" << InputSize << std::endl;

	//	gr: made all these optional for testing bad cases, but by default w&h should be something (as per decode params)
	//	gr: if these are not set, the decoder doesnt work. (Need to get some logs, did this fail to decode, or fail to configure)
	if ( Width > 0 )
		AMediaFormat_setInt32( Format, AMEDIAFORMAT_KEY_WIDTH, Width );
	if ( Height > 0 )
		AMediaFormat_setInt32( Format, AMEDIAFORMAT_KEY_HEIGHT, Height );
		
	//	doesn't seem to do anything
	if ( InputSize > 0 )
		AMediaFormat_setInt32( Format, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, InputSize );
	
	
	//AMEDIAFORMAT_KEY_LEVEL
	//AMEDIAFORMAT_KEY_PROFILE
	//
	//	magic leap version has nalu seperated SPS & PPS in csd-0
	//	0001 sps 0001 pps
	{
		auto& Sps = mPendingSps;
		auto& Pps = mPendingPps;
		Array<uint8_t> SpsAndPps;
		SpsAndPps.PushBackArray(Sps);
		SpsAndPps.PushBackArray(Pps);
		/*
		AMEDIAFORMAT_KEY_CSD; # var introduced=28
    AMEDIAFORMAT_KEY_CSD_0; # var introduced=28
    AMEDIAFORMAT_KEY_CSD_1; # var introduced=28
    AMEDIAFORMAT_KEY_CSD_2; # var introduced=28
    AMEDIAFORMAT_KEY_CSD_AVC; # var introduced=29
    AMEDIAFORMAT_KEY_CSD_HEVC; # var introduced=29
    */
		//	gr: this isn't making a different (same profile, same level, same dimensions)
		AMediaFormat_setBuffer( Format, AMEDIAFORMAT_KEY_CSD_AVC, SpsAndPps.GetArray(), SpsAndPps.GetDataSize() );
	}


	//	configure with our new format
	ANativeWindow* Surface = nullptr;
	AMediaCrypto* Crypto = nullptr;
	uint32_t CodecFlags = 0;//AMEDIACODEC_CONFIGURE_FLAG_ENCODE;

	//	gr: debug with 
	//		adb logcat -s CCodec
	if ( mParams.mVerboseDebug )
		std::Debug << __PRETTY_FUNCTION__ << " AMediaCodec_configure..." << std::endl;
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
	if ( mParams.mVerboseDebug )
		std::Debug << __PRETTY_FUNCTION__ << " AMediaCodec_Start..." << std::endl;
	Status = AMediaCodec_start(mCodec);
	IsOkay(Status,"AMediaCodec_Start");
	
	if ( mParams.mVerboseDebug )
		std::Debug << __PRETTY_FUNCTION__ << " Codec created." << std::endl;
}



Android::TDecoder::~TDecoder()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	if ( mCodec )
	{
		try
		{
			auto Result = AMediaCodec_stop( mCodec );
			IsOkay( Result, "AMediaCodec_stop" );
		
			Result = AMediaCodec_flush( mCodec );
			IsOkay( Result, "AMediaCodec_flush" );
		}
		catch(std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << " AMediaCodec_stop/AMediaCodec_flush exception; " << e.what() << std::endl;
		}
		
		try
		{
			std::Debug << __PRETTY_FUNCTION__ << "stop input & output threads..." << std::endl;
			mInputThread.Stop();
			mOutputThread.Stop();
			std::Debug << __PRETTY_FUNCTION__ << "wait for input & output threads to finish..." << std::endl;
			mInputThread.WaitToFinish();
			mOutputThread.WaitToFinish();
		
			//	gr: I seem to recall it's safe to destroy the codec if the threads arent finished, 
			//		they will error internally need to make sure. Find some reference to thread safety in the docs/code!
			std::Debug << "AMediaCodec_destroy" << std::endl;
			auto Result = AMediaCodec_delete( mCodec );
			IsOkay( Result, "AMediaCodec_delete" );
		}
		catch(std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << " exception; " << e.what() << std::endl;
		}
		mCodec = nullptr;
	}
	
	//	make sure threads are stopped regardless
	std::Debug << __PRETTY_FUNCTION__ << "hail mary wait for input & output threads to finish..." << std::endl;
	mInputThread.WaitToFinish();
	mOutputThread.WaitToFinish();
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
	std::Debug << __PRETTY_FUNCTION__<< std::endl;

	//	we have to wait for input thread to want to pull data here, we can't force it
	mInputThread.Wake();


	//	if we have no codec yet, we won't have any input buffers, so we won't have any requests from the input thread
	//	we need to pull SPS & PPS and create codec
	if ( !mCodec )
	{
		std::Debug << __PRETTY_FUNCTION__<< " Creating codec..." << std::endl;
		try
		{
			CreateCodec();
		}
		catch(std::exception& e)
		{
			std::Debug << "DecodeNextPacket CreateCodec failed; " << e.what() << " (assuming we need more data)" << std::endl;
			return false;
		}
	}
	
	//	gr: we don't currently have callbacks for buffers, so deque any that are availible
	if ( !mAsyncBuffers )
	{
		DequeueInputBuffers();
		DequeueOutputBuffers();
	}
	
	//	even if we didn't get a frame, try to decode again as we processed a packet
	//return true;
	return false;
}

void Android::TDecoder::GetNextInputData(ArrayBridge<uint8_t>&& PacketBuffer,PopH264::FrameNumber_t& FrameNumber)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	

	auto& Nalu = PacketBuffer;

	//	input thread wants some data to process
	if (!PopNalu(GetArrayBridge(Nalu),FrameNumber))
	{
		std::stringstream Error;
		Error << "GetNextInputData, no nalu ready ";//(" << GetPendingDataSize() <<" bytes ready pending)";
		throw Soy::AssertException(Error);
	}
/*
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
	*/
}


void Android::TInputThread::PushInputBuffer(int64_t BufferIndex)
{
	//	gr: we can grab a buffer without submitted it, so this is okay if we fail here
	std::Debug << "Pushing to input buffer #" << BufferIndex << std::endl;

	//auto BufferHandle = static_cast<MLHandle>( BufferIndex );
	uint8_t* Buffer = nullptr;
	size_t BufferSize = 0;
	Buffer = AMediaCodec_getInputBuffer( mCodec, BufferIndex, &BufferSize );
	//auto Result = MLMediaCodecGetInputBufferPointer( mHandle, BufferHandle, &Buffer, &BufferSize );
	//IsOkay( Result, "MLMediaCodecGetInputBufferPointer" );
	if ( Buffer == nullptr )
	{
		std::stringstream Error;
		Error << "AMediaCodec_getInputBuffer null buffer (size=" << BufferSize << ")";
		throw Soy::AssertException(Error);
	}

	//	grab next packet
	//	gr: problem here, if the packet is bigger than the input buffer, we won't have anywhere to put it
	//		current system means this packet is dropped (or could unpop, but then we'll be stuck anyway)
	//	gr: as we can submit an offset, we could LOCK the pending data, submit, then unlock & delete and save a copy
	size_t BufferWrittenSize = 0;
	auto BufferArray = GetRemoteArray( Buffer, BufferSize, BufferWrittenSize );
	PopH264::FrameNumber_t PacketTime = 0;
	mPopPendingData( GetArrayBridge(BufferArray), PacketTime );

	//	process buffer
	int64_t DataOffset = 0;
	uint64_t PresentationTimeMicroSecs = PacketTime;

	int Flags = 0;
  	//	AMEDIACODEC_BUFFER_FLAG_PARTIAL_FRAME
	//Flags |= MLMediaCodecBufferFlag_KeyFrame;
	//Flags |= AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG;
	//Flags |= AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM;
	//Flags |= MLMediaCodecBufferFlag_KeyFrame;
	//Flags |= MLMediaCodecBufferFlag_CodecConfig;
	//Flags |= MLMediaCodecBufferFlag_EOS;
	auto Result = AMediaCodec_queueInputBuffer( mCodec, BufferIndex, DataOffset, BufferWrittenSize, PresentationTimeMicroSecs, Flags );
	IsOkay( Result, "AMediaCodec_queueInputBuffer" );
	
	OnInputSubmitted( PresentationTimeMicroSecs );

	std::Debug << "AMediaCodec_queueInputBuffer( BufferIndex=" << BufferIndex << " DataSize=" << BufferWrittenSize << "/" << BufferSize << " presentationtime=" << PresentationTimeMicroSecs << ") success" << std::endl;
}


void Android::TInputThread::OnInputBufferAvailible(MediaCodec_t Codec,bool AsyncBuffers,int64_t BufferIndex)
{
	{
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		mCodec = Codec;
		mAsyncBuffers = AsyncBuffers;
		mInputBuffers.PushBack(BufferIndex);
	}
	std::Debug << "OnInputBufferAvailible(" << BufferIndex << ") " << GetDebugState() << std::endl;
	Wake();
}

void Android::TDecoder::OnInputBufferAvailible(int64_t BufferIndex)
{
	mInputThread.OnInputBufferAvailible( mCodec, mAsyncBuffers, BufferIndex );
}

void Android::TDecoder::OnOutputBufferAvailible(int64_t BufferIndex,const MediaBufferInfo_t& BufferMeta)
{
	//	gr: in non-async mode we won't have the format, should fetch it here
	/*	gr: this blocks... in async mode? no output in logcat
	auto* pFormat = AMediaCodec_getOutputFormat(mCodec);
	if ( !pFormat )
	{
		std::Debug << __PRETTY_FUNCTION__ << " failed to get current output format from codec" << std::endl;
	}
	*/
	TOutputBufferMeta Meta;
	Meta.mPixelMeta = mOutputPixelMeta;
	Meta.mBufferIndex = BufferIndex;
	Meta.mMeta = BufferMeta;
	mOutputThread.OnOutputBufferAvailible( mCodec, mAsyncBuffers, Meta );
	std::Debug << "OnOutputBufferAvailible sent to output thread(" << BufferIndex << ") " << GetDebugState() << mOutputThread.GetDebugState() << std::endl;
}

/*
void Android::TDecoder::OnOutputTextureAvailible()
{
	mOutputThread.OnOutputTextureAvailible();
}

void Android::TDecoder::OnOutputTextureWritten(int64_t PresentationTime)
{
	mOutputThread.OnOutputTextureWritten(PresentationTime);
}
*/

void Android::TDecoder::OnOutputFormatChanged(MediaFormat_t NewFormat)
{
	//	gr: we should do this like a queue for the output thread
	//		for streams that can change format mid-way
	//	todo: make test streams that change format!
	if ( mParams.mVerboseDebug )
		std::Debug << "Got new output format..." << std::endl;
	try
	{
		mOutputPixelMeta = GetPixelMeta( NewFormat, mParams.mVerboseDebug, mOutputMeta );
		
		//	gr: update output thread meta. Might need to be careful about changing this at the right time
		//		if we have a stream that changes format. It would need sending with each OnOutputBuffer meta (but lots of redundant json copy!)
		mOutputThread.mOutputMeta = mOutputMeta;
		//if ( mParams.mVerboseDebug )
			std::Debug << "New output format is " << mOutputPixelMeta << std::endl;
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " Exception " << e.what() << std::endl;
	}
}



Android::TOutputThread::TOutputThread(PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	SoyWorkerThread	("AndroidOutputThread", SoyWorkerWaitMode::Wake ),
	mOnDecodedFrame	( OnDecodedFrame ),
	mOnFrameError	( OnFrameError )
{
	Start();
}


void Android::TOutputThread::OnInputSubmitted(int32_t PresentationTime)
{
	//	std::lock_guard<std::mutex> Lock(mPendingPresentationTimesLock);
	//	mPendingPresentationTimes.PushBack( PresentationTime );
}


void Android::TOutputThread::OnOutputBufferAvailible(MediaCodec_t Codec,bool AsyncBuffers,const TOutputBufferMeta& BufferMeta)
{
	std::Debug << __PRETTY_FUNCTION__ << " locking..." << std::endl;
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	std::Debug << __PRETTY_FUNCTION__ << " locked" << std::endl;
	mOutputBuffers.PushBack( BufferMeta );
	mCodec = Codec;
	mAsyncBuffers = AsyncBuffers;
	Wake();
}
/*
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
*/

bool Android::TOutputThread::CanSleep()
{
	//	buffers to get
	if ( !mOutputBuffers.IsEmpty() )
		return false;
	/*
	//	textures to get
	if ( mOutputTexturesAvailible > 0 )
		return false;
	
	if ( IsAnyOutputTextureReady() )
		return false;
	*/
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
/*
void Android::TOutputThread::RequestOutputTexture()
{
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
}
*/
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
/*
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

	//	release it
	auto Result = MLMediaCodecReleaseFrame( mCodecHandle, TextureHandle );
	IsOkay( Result, "MLMediaCodecReleaseFrame" );
}
*/
void Android::TOutputThread::PopOutputBuffer(const TOutputBufferMeta& BufferMeta)
{
	//	gr: hold onto pointer and don't release buffer until it's been read,to avoid a copy
	//		did this on android and it was a boost
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__,0);
	//auto BufferHandle = static_cast<MLHandle>( BufferMeta.mBufferIndex );

	auto BufferIndex = BufferMeta.mBufferIndex;
	bool Released = false;

	//	release buffer back as data has been used
	auto ReleaseBuffer = [&](bool Render=true)
	{
	/*10-15 06:49:49.856  9681  9684 I CCodecBufferChannel: [c2.qti.avc.decoder#184] cannot render buffer without surface
10-15 06:49:49.856  9681  9683 I PopH264 : pop: void Android::TOutputThread::PopOutputBuffer(const Android::TOutputBufferMeta &) took 1ms/0ms to execute
10-15 06:49:49.856  9681  9683 I PopH264 : pop: Exception getting output buffer 0; SoyPixelsRemote meta size(0) different to data size (36864) mOutputBuffers[] 
10-15 06:49:50.857  9681  9683 E MediaCodec: getBufferAndFormat - invalid operation (the index 0 is not owned by client)
10-15 06:49:50.857  9681  9683 I PopH264 : pop: Got Invalid OutputBuffer(0) BufferSize=4294967295 BufferData=0x0
10-15 06:49:50.858  9681  9683 E NdkMediaCodec: sf error code: -13
10-15 06:49:50.858  9681  9683 I PopH264 : pop: void Android::TOutputThread::PopOutputBuffer(const Android::TOutputBufferMeta &) took 1ms/0ms to execute
10-15 06:49:50.859  9681  9683 I PopH264 : pop: Exception getting output buffer 0; AMedia error -10000/AMEDIA_ERROR_UNKNOWN in MLMediaCodecReleaseOutputBuffer mOutputBuffers[] 
10-15 06:49:51.630   986   986 I QC2CompStore: Deleting component(c2.qti.avc.decoder) id(32)
10-15 06:49:51.632  1167  1167 E mediaserver: unlinkToDeath: removed reference to death recipient but */
		if ( Released )
			return;
			
		//	gr: catch release errors as prev errors will probably just error here
		//		but we do it just in case
		try
		{
			Soy::TScopeTimerPrint Timer("Android::TOutputThread::PopOutputBuffer AMediaCodec_releaseOutputBuffer",1);
			auto Result = AMediaCodec_releaseOutputBuffer( mCodec, BufferIndex, Render );
			IsOkay( Result, "MLMediaCodecReleaseOutputBuffer");
		}
		catch(std::exception& e)
		{
			std::Debug << "Caught ReleaseBuffer exception; " << e.what() << std::endl;
		}
		Released = true;
	};

	/*
	MLMediaCodecBufferInfo BufferMeta;
	int16_t Timeout = 0;
	int64_t NewBufferIndex = -1;
	auto Result = MLMediaCodecDequeueOutputBuffer( mCodecHandle, &BufferMeta, Timeout, &NewBufferIndex );
	IsOkay( Result, "MLMediaCodecDequeueOutputBuffer");
	std::Debug << "MLMediaCodecDequeueOutputBuffer returned buffer index " << NewBufferIndex << " compared to " << OutputBufferIndex << " time=" << BufferMeta.presentation_time_us << std::endl;
	*/

	size_t BufferSize = -1;
	Soy::TScopeTimerPrint Timer2("Android::TOutputThread::PopOutputBuffer AMediaCodec_getOutputBuffer",1);
	uint8_t* BufferData = AMediaCodec_getOutputBuffer( mCodec, BufferIndex, &BufferSize ); 
	Timer2.Stop();
	
	//	if data is null, then output is a surface
	if ( BufferData == nullptr || BufferSize == 0 )
	{
		std::Debug << "Got Invalid OutputBuffer(" << BufferMeta.mBufferIndex << ") BufferSize=" << BufferSize << " BufferData=0x" << std::hex << (size_t)(BufferData) << std::dec << std::endl;
		ReleaseBuffer();
		return;
	}
	
	auto FrameTime = BufferMeta.mMeta.presentationTimeUs;
	auto Flags = BufferMeta.mMeta.flags; 
	auto BufferDataOffset = BufferMeta.mMeta.offset;
	auto BufferDataSize = BufferMeta.mMeta.size;
	std::Debug << "Got OutputBuffer(" << BufferMeta.mBufferIndex << ") BufferSize=" << BufferSize << " BufferData=0x" << std::hex << (size_t)(BufferData) << std::dec << " FrameTime=" << FrameTime << " Flags=" << Flags << " offset=" << BufferDataOffset << " size=" << BufferDataSize << std::endl;
	try
	{
		//	erroring here on samsung s7 with 
		//	format= 1280x1616^Yuv_8_88
		//	expected size = 3102720
		//	real size = 3104768
		//	https://stackoverflow.com/a/20707645/355753
		//	suggests padding (luma)plane to 2kb boundry, or rather, chroma plane being on a boundary
		//	BUT both real size and expected size align. (2048*1515 & 2048*1516)
		//	just for no reason, an extra page.
		//	gr: then realised, we're not using the buffer info offsets
		//auto OutputBufferSize = BufferSize;
		auto OutputBufferSize = BufferDataSize;
		auto PixelFormatBufferSize = BufferMeta.mPixelMeta.GetDataSize();
		if ( OutputBufferSize > PixelFormatBufferSize )
		{
			std::Debug << "Clipping output pixel size from buffersize=" << BufferSize << " (meta buffer size=" << BufferDataSize <<" offset=" << BufferDataOffset <<") to " << PixelFormatBufferSize << " for " << BufferMeta.mPixelMeta << std::endl;
			OutputBufferSize = PixelFormatBufferSize;
		}
	
		//	output pixels!
		//	gr: use buffer meta offset for when buffer isn't neccessarily aligned
		//	gr: todo: be careful here and detect bad offsets going OOB
		auto* BufferDataMutable = const_cast<uint8_t*>( BufferData );
		BufferDataMutable += BufferDataOffset;
		SoyPixelsRemote NewPixels( BufferDataMutable, OutputBufferSize, BufferMeta.mPixelMeta );
		
		//	extra meta
		json11::Json::object Meta = mOutputMeta;
		Meta["AndroidBufferFlags"] = static_cast<int>(Flags);
	
		{	
			Soy::TScopeTimerPrint Timer3("Android::TOutputThread::PopOutputBuffer PushFrame",1);
			PushFrame( NewPixels, FrameTime, Meta );
		}
		ReleaseBuffer();
	}
	catch(std::exception& e)
	{
		//	rethrow but make sure we return the buffer first
		ReleaseBuffer();
		throw;
	}
}


void Android::TOutputThread::PushFrame(const SoyPixelsImpl& Pixels,PopH264::FrameNumber_t FrameNumber,const json11::Json& Meta)
{
	mOnDecodedFrame( Pixels, FrameNumber, Meta );
}



bool Android::TOutputThread::Iteration()
{
/*
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
	*/
	if ( !mOutputBuffers.IsEmpty() )
	{
		//	read a buffer
		TOutputBufferMeta BufferMeta;
		if ( mAsyncBuffers )
		{
			std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
			BufferMeta = mOutputBuffers.PopAt(0);
		}
		else
		{
			Soy_AssertTodo();
		}
		try
		{
			PopOutputBuffer( BufferMeta );
		}
		catch(std::exception& e)
		{
			bool TryAgain = false;
			std::Debug << "Exception getting output buffer " << BufferMeta.mBufferIndex << "; " << e.what() << GetDebugState() << std::endl;
			
			{
				PopH264::FrameNumber_t FrameTime = BufferMeta.mMeta.presentationTimeUs;
				std::stringstream Error;
				Error << "PopOutputBuffer error " << e.what();
				mOnFrameError( Error.str(), &FrameTime );
			}
			
			if ( TryAgain )
			{
				std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
				auto& ElementZero = *mOutputBuffers.InsertBlock(0,1);
				ElementZero = BufferMeta;
			}
			std::this_thread::sleep_for( std::chrono::milliseconds(OutputThreadErrorSleepMs) );
		}
	}
	
	return true;
}


Android::TInputThread::TInputThread(std::function<void(ArrayBridge<uint8_t>&&,PopH264::FrameNumber_t&)> PopPendingData,std::function<bool()> HasPendingData) :
	mPopPendingData	( PopPendingData ),
	mHasPendingData	( HasPendingData ),
	SoyWorkerThread	("Android::TInputThread", SoyWorkerWaitMode::Wake )
{
	Start();
}


bool Android::TInputThread::Iteration(std::function<void(std::chrono::milliseconds)> Sleep)
{
	if ( mInputBuffers.IsEmpty() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " No input buffers sleep(" << InputThreadNotReadySleep << ")" << std::endl;
		//Sleep( std::chrono::milliseconds(InputThreadNotReadySleep) );
		//std::this_thread::sleep_for(std::chrono::milliseconds(InputThreadNotReadySleep) );
		return true;
	}

	if ( !HasPendingData() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " No pending data" << std::endl;
		//Sleep( std::chrono::milliseconds(InputThreadNotReadySleep) );
		//std::this_thread::sleep_for(std::chrono::milliseconds(InputThreadNotReadySleep) );
		return true;
	}
	
	//if ( mParams.mVerboseDebug )
	{
		//std::Debug << __PRETTY_FUNCTION__ << " Reading a buffer" << std::endl;
	}
	bool DequeueNextIndex = !mAsyncBuffers;

	//	read a buffer
	int64_t BufferIndex = -1;
	if ( !DequeueNextIndex )
	{
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		BufferIndex = mInputBuffers.PopAt(0);
	}
	
	try
	{
		if ( DequeueNextIndex )
		{
			int Timeout = 0;
			//	gr: this returns -1000 in async mode (see logcat MediaCodec)
			BufferIndex = AMediaCodec_dequeueInputBuffer( mCodec, Timeout );
			if ( BufferIndex < 0 )
			{
				std::stringstream Error;
				Error << "AMediaCodec_dequeueInputBuffer returned buffer index " << BufferIndex;
				throw Soy::AssertException(Error);
			}
			std::lock_guard<std::mutex> Lock(mInputBuffersLock);
			mInputBuffers.Remove(BufferIndex);
		}
		PushInputBuffer( BufferIndex );
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " Exception pushing input buffer " << BufferIndex << "; " << e.what() << GetDebugState() << std::endl;
		if ( BufferIndex >= 0 )
		{
			std::lock_guard<std::mutex> Lock(mInputBuffersLock);
			auto& ElementZero = *mInputBuffers.InsertBlock(0,1);
			ElementZero = BufferIndex;
		}
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
	//std::lock_guard<std::recursive_mutex> Lock2(mOutputTexturesLock);

	std::stringstream Debug;
	Debug << " mOutputBuffers[";
	for ( auto i=0;	i<mOutputBuffers.GetSize();	i++ )
		Debug << mOutputBuffers[i].mBufferIndex << ",";
	Debug << "] ";
	/*
	Debug << "mOutputTexturesAvailible=" << mOutputTexturesAvailible << " ";
	Debug << "mOutputTextures[";
	for ( auto i=0;	i<mOutputTextures.GetSize();	i++ )
		Debug << std::hex << "0x" << mOutputTextures[i].mTextureHandle << std::dec << ",";
	Debug << "] ";
*/
	return Debug.str();
}
