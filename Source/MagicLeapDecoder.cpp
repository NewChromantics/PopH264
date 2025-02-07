#include "MagicLeapDecoder.h"

#include <ml_media_codec.h>
#include <ml_media_codeclist.h>
#include <ml_media_format.h>
#include <ml_media_error.h>



namespace MagicLeap
{
	void			IsOkay(MLResult Result,const char* Context);
	void			IsOkay(MLResult Result,std::stringstream& Context);
	const char*		GetErrorString(MLResult Result);
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
	const auto*			H264MimeType = "video/avc";
	
	//	CSD-0 (from android mediacodec api)
	//	not in the ML api
	//	https://forum.magicleap.com/hc/en-us/community/posts/360048067552-Setting-csd-0-byte-buffer-using-MLMediaFormatSetKeyByteBuffer
	MLMediaFormatKey		MLMediaFormat_Key_CSD0 = "csd-0";

	SoyPixelsFormat::Type	GetPixelFormat(int32_t ColourFormat);
	SoyPixelsMeta			GetPixelMeta(MLHandle Format);
}



void MagicLeap::IsOkay(MLResult Result,std::stringstream& Context)
{
	if ( Result == MLResult_Ok )
		return;
	
	auto Str = Context.str();
	IsOkay( Result, Str.c_str() );
}

const char* MagicLeap::GetErrorString(MLResult Result)
{
	//	specific media errors
	auto HasPrefix = [&](MLResult Prefix)
	{
		auto And = Result & Prefix;
		return And == Prefix;
	};

	if ( HasPrefix(MLResultAPIPrefix_MediaGeneric) ||
		HasPrefix(MLResultAPIPrefix_Media) ||
		HasPrefix(MLResultAPIPrefix_MediaDRM) ||
		HasPrefix(MLResultAPIPrefix_MediaOMX) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXExtensions) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXVendors) ||
		HasPrefix(MLResultAPIPrefix_MediaPlayer) )
	{
		return MLMediaResultGetString( Result );
	}

	return MLGetResultString(Result);
}

void MagicLeap::IsOkay(MLResult Result,const char* Context)
{
	if ( Result == MLResult_Ok )
		return;

	auto ResultString = GetErrorString(Result);
	
	//	gr: sometimes we get unknown so, put error nmber in
	std::stringstream Error;
	Error << "Error in " << Context << ": " << ResultString;
	throw Soy::AssertException( Error );
}


SoyPixelsFormat::Type MagicLeap::GetPixelFormat(int32_t ColourFormat)
{
	//	http://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities.html
	enum
	{
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
	
	//	magic leap + nvidia decoder
	COLOR_FORMAT_UNKNOWN_NVIDIA_SURFACE = 262,

	//	note4 is giving us a pixel format of this, even when we have a surface texture. Renders okay.
	//	gr; in non-opengl mode, the colour space is broken!
	//		it is not the same as Yuv_8_8_8_Full (but close). The name is a big hint of this.
	//		probably more like Yuv_8_88 but line by line.
	OMX_QCOM_COLOR_FormatYVU420SemiPlanarInterlace = 0x7FA30C04,	//	2141391876
	};
	
	//	nvidia decoder gives 262
	//	google decoder gives 19
	//	these match
	//	http://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities.html

	switch (ColourFormat)
	{
		case COLOR_FormatYUV420Planar:				return SoyPixelsFormat::Yuv_8_8_8_Full;
		case COLOR_FormatYUV420SemiPlanar:			return SoyPixelsFormat::Yuv_8_88_Full;
		case COLOR_FORMAT_UNKNOWN_NVIDIA_SURFACE:	return SoyPixelsFormat::Yuv_8_8_8_Ntsc;	//	not sure what this is yet, so for identification
		default:break;
	}
	
	std::stringstream Error;
	Error << "Unhandled colour format #" << ColourFormat << std::endl;
	throw Soy::AssertException(Error);
}

SoyPixelsMeta MagicLeap::GetPixelMeta(MLHandle Format)
{
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
}


void MagicLeap::EnumCodecs(std::function<void(const std::string&)> EnumCodec)
{
	/* gr: these are the codecs my device reported;
OMX.Nvidia.mp4.decode
OMX.Nvidia.h263.decode
OMX.Nvidia.h264.decode
OMX.Nvidia.h264.decode.secure
OMX.Nvidia.vp8.decode
OMX.Nvidia.vp9.decode
OMX.Nvidia.vp9.decode.secure
OMX.Nvidia.h265.decode
OMX.Nvidia.h265.decode.secure
OMX.Nvidia.mpeg2v.decode
OMX.Nvidia.mp3.decoder
OMX.Nvidia.mp2.decoder
OMX.Nvidia.wma.decoder
OMX.Nvidia.vc1.decode
OMX.Nvidia.mjpeg.decoder
OMX.Nvidia.h264.encoder
OMX.Nvidia.h265.encoder
OMX.Nvidia.vp8.encoder
OMX.Nvidia.vp9.encoder
OMX.google.mp3.decoder
OMX.google.amrnb.decoder
OMX.google.amrwb.decoder
OMX.google.aac.decoder
OMX.google.g711.alaw.decoder
OMX.google.g711.mlaw.decoder
OMX.google.vorbis.decoder
OMX.google.opus.decoder
OMX.google.raw.decoder
OMX.google.aac.encoder
OMX.google.amrnb.encoder
OMX.google.amrwb.encoder
OMX.google.flac.encoder
OMX.Nvidia.eaacp.decoder
OMX.google.mpeg4.decoder
OMX.google.h263.decoder
OMX.google.h264.decoder
OMX.google.hevc.decoder
OMX.google.vp8.decoder
OMX.google.vp9.decoder
OMX.google.h263.encoder
OMX.google.h264.encoder
OMX.google.mpeg4.encoder
OMX.google.vp8.encoder
	*/
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


MagicLeap::TDecoder::TDecoder(int32_t Mode) :
	mInputThread	( std::bind(&TDecoder::PopPendingData, this, std::placeholders::_1 ), std::bind(&TDecoder::HasPendingData, this ) )
{
	auto EnumCodec = [](const std::string& Name)
	{
		std::Debug << "Codec: " << Name << std::endl;
	};
	EnumCodecs( EnumCodec );

	bool HardwareSurface = false;
	auto CodecName = GetCodec( Mode, HardwareSurface );
	bool IsMime = Soy::StringBeginsWith(CodecName,"video/",false);
	auto CreateByType = IsMime ? MLMediaCodecCreation_ByType : MLMediaCodecCreation_ByName;

	//	got this mime from googling;
	//	http://hello-qd.blogspot.com/2013/05/choose-decoder-and-encoder-by-google.html
	
	//	OMX.google.h264.decoder
	//	ByName
	//	https://forum.magicleap.com/hc/en-us/community/posts/360041748952-Follow-up-on-Multimedia-Decoder-API
	
	//	gr: this example has a MIME with TYPE
	//	https://forum.magicleap.com/hc/en-us/community/posts/360041748952-Follow-up-on-Multimedia-Decoder-API
	//	our use of OMX.Nvidia.h264.decode and NAME, errors

	auto Result = MLMediaCodecCreateCodec( CreateByType, MLMediaCodecType_Decoder, CodecName.c_str(), &mCodec );
	{
		std::stringstream Error;
		Error << "MLMediaCodecCreateCodec(" << CodecName << ")";
		IsOkay( Result, Error );
	}
	
	auto OnInputBufferAvailible = [](MLHandle Codec,int64_t BufferIndex,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnInputBufferAvailible( Codec=" << Codec << " BufferIndex=" << BufferIndex << ")" << std::endl;
		This.OnInputBufferAvailible(BufferIndex);
	};
	
	auto OnOutputBufferAvailible = [](MLHandle Codec,int64_t BufferIndex,MLMediaCodecBufferInfo* BufferInfo,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnOutputBufferAvailible( Codec=" << Codec << " BufferIndex=" << BufferIndex << ")" << std::endl;
		This.OnOutputBufferAvailible( BufferIndex, *BufferInfo );
	};
	
	auto OnOutputFormatChanged = [](MLHandle Codec,MLHandle NewFormat,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnOutputFormatChanged" << std::endl;
		This.OnOutputFormatChanged( NewFormat );
	};
	
	auto OnError = [](MLHandle Codec,int ErrorCode,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnError(" << ErrorCode << ")" << std::endl;
	};
	
	auto OnFrameRendered = [](MLHandle Codec,int64_t PresentationTimeMicroSecs,int64_t SystemTimeNano,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnFrameRendered( pt=" << PresentationTimeMicroSecs << " systime=" << SystemTimeNano << ")" << std::endl;
		//This.OnOutputTextureWritten( PresentationTimeMicroSecs );
	};
	
	auto OnFrameAvailible = [](MLHandle Codec,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnFrameAvailible" << std::endl;
		This.OnOutputTextureAvailible();
	};
	

	//	gr: not sure if this can be a temporary or not, demo on forums has a static
	MLMediaCodecCallbacks Callbacks;
	Callbacks.on_input_buffer_available = OnInputBufferAvailible;
	Callbacks.on_output_buffer_available = OnOutputBufferAvailible;
	Callbacks.on_output_format_changed = OnOutputFormatChanged;
	Callbacks.on_error = OnError;
	Callbacks.on_frame_rendered = OnFrameRendered;
	Callbacks.on_frame_available = OnFrameAvailible;
	
	Result = MLMediaCodecSetCallbacks( mCodec, &Callbacks, this );
	IsOkay( Result, "MLMediaCodecSetCallbacks" );

	//	configure
	MLHandle Format = ML_INVALID_HANDLE;
	Result = MLMediaFormatCreateVideo( H264MimeType, 1920, 1080, &Format );
	IsOkay( Result, "MLMediaFormatCreateVideo" );

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
	
	
	//	force software mode
	if ( HardwareSurface )
	{
		Result = MLMediaCodecSetSurfaceHint( mCodec, MLMediaCodecSurfaceHint_Hardware );
		IsOkay( Result, "MLMediaCodecSetSurfaceHint(hardware)" );
	}
	else
	{
		Result = MLMediaCodecSetSurfaceHint( mCodec, MLMediaCodecSurfaceHint_Software );
		IsOkay( Result, "MLMediaCodecSetSurfaceHint(software)" );
	}
	
	//MLHandle Crypto = ML_INVALID_HANDLE;
	MLHandle Crypto = 0;	//	gr: INVALID_HANDLE doesnt work
	Result = MLMediaCodecConfigure( mCodec, Format, Crypto );
	IsOkay( Result, "MLMediaCodecConfigure" );
	
	Result = MLMediaCodecStart( mCodec );
	IsOkay( Result, "MLMediaCodecStart" );
	
	//	MLMediaCodecFlush makes all inputs invalid... flush on close?
	std::Debug << "Created TDecoder (" << CodecName << ")" << std::endl;
}

MagicLeap::TDecoder::~TDecoder()
{
	try
	{
		auto Result = MLMediaCodecStop( mCodec );
		IsOkay( Result, "MLMediaCodecStop" );
		
		Result = MLMediaCodecFlush( mCodec );
		IsOkay( Result, "MLMediaCodecFlush" );
		
		Result = MLMediaCodecDestroy( mCodec );
		IsOkay( Result, "MLMediaCodecDestroy" );
	}
	catch(std::exception& e)
	{
		std::Debug << __func__ << e.what() << std::endl;
	}
}


//	returns true if more data to proccess
bool MagicLeap::TDecoder::DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)
{
	std::Debug << "DecodeNextPacket (pendingdata x" << mPendingData.GetSize() << ") " << GetDebugState() << mOutputThread.GetDebugState() << std::endl;

	//	current setup currently treats each packet here as a new frame (mp4 chunked)
	//	so for every packet coming in, there should be one output
	//	gr: this doesn't ring true for PPS/SPS data though?
	mOutputThread.PushOnOutputFrameFunc( OnFrameDecoded );
	
	//	wake the input thread, as when this is called, the super class has just put some pending data in
	mInputThread.Wake();
	return false;
}


void MagicLeap::TInputThread::PushInputBuffer(int64_t BufferIndex)
{
	//	gr: we can grab a buffer without submitted it, so this is okay if we fail here
	std::Debug << "Pushing to input buffer #" << BufferIndex << std::endl;
		
	auto BufferHandle = static_cast<MLHandle>( BufferIndex );
	uint8_t* Buffer = nullptr;
	size_t BufferSize = 0;
	auto Result = MLMediaCodecGetInputBufferPointer( mCodec, BufferHandle, &Buffer, &BufferSize );
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
	mPopPendingData( GetArrayBridge(BufferArray) );

	//	process buffer
	int64_t DataOffset = 0;
	uint64_t PresentationTimeMicroSecs = mPacketCounter;
	mPacketCounter++;

	int Flags = 0;
	//Flags |= MLMediaCodecBufferFlag_KeyFrame;
	//Flags |= MLMediaCodecBufferFlag_CodecConfig;
	//Flags |= MLMediaCodecBufferFlag_EOS;
	Result = MLMediaCodecQueueInputBuffer( mCodec, BufferHandle, DataOffset, BufferWrittenSize, PresentationTimeMicroSecs, Flags );
	IsOkay( Result, "MLMediaCodecQueueInputBuffer" );
	
	OnInputSubmitted( PresentationTimeMicroSecs );

	std::Debug << "MLMediaCodecQueueInputBuffer( BufferIndex=" << BufferIndex << " DataSize=" << BufferWrittenSize << "/" << BufferSize << " presentationtime=" << PresentationTimeMicroSecs << ") success" << std::endl;
}


void MagicLeap::TInputThread::OnInputBufferAvailible(MLHandle CodecHandle,int64_t BufferIndex)
{
	{
		std::lock_guard<std::mutex> Lock(mInputBuffersLock);
		mCodec = CodecHandle;
		mInputBuffers.PushBack(BufferIndex);
	}
	std::Debug << "OnInputBufferAvailible(" << BufferIndex << ") " << GetDebugState() << std::endl;
	Wake();
}

void MagicLeap::TDecoder::OnInputBufferAvailible(int64_t BufferIndex)
{
	mInputThread.OnInputBufferAvailible( mCodec, BufferIndex );
}

void MagicLeap::TDecoder::OnOutputBufferAvailible(int64_t BufferIndex,const MLMediaCodecBufferInfo& BufferMeta)
{
	TOutputBufferMeta Meta;
	Meta.mPixelMeta = mOutputPixelMeta;
	Meta.mBufferIndex = BufferIndex;
	Meta.mMeta = BufferMeta;
	mOutputThread.OnOutputBufferAvailible( mCodec, Meta );
	std::Debug << "OnOutputBufferAvailible(" << BufferIndex << ") " << GetDebugState() << mOutputThread.GetDebugState() << std::endl;
}

void MagicLeap::TDecoder::OnOutputTextureAvailible()
{
	mOutputThread.OnOutputTextureAvailible();
}

void MagicLeap::TDecoder::OnOutputTextureWritten(int64_t PresentationTime)
{
	mOutputThread.OnOutputTextureWritten(PresentationTime);
}

void MagicLeap::TDecoder::OnOutputFormatChanged(MLHandle NewFormat)
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

MagicLeap::TOutputThread::TOutputThread() :
	SoyWorkerThread	("MagicLeapOutputThread", SoyWorkerWaitMode::Wake )
{
	Start();
}

void MagicLeap::TOutputThread::OnInputSubmitted(int32_t PresentationTime)
{
	//	std::lock_guard<std::mutex> Lock(mPendingPresentationTimesLock);
	//	mPendingPresentationTimes.PushBack( PresentationTime );
}

void MagicLeap::TOutputThread::PushOnOutputFrameFunc(std::function<void(const SoyPixelsImpl&,SoyTime)>& PushFrameFunc)
{
	std::lock_guard<std::mutex> Lock(mPushFunctionsLock);
	mPushFunctions.PushBack( PushFrameFunc );
}

void MagicLeap::TOutputThread::OnOutputBufferAvailible(MLHandle CodecHandle,const TOutputBufferMeta& BufferMeta)
{
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	mOutputBuffers.PushBack( BufferMeta );
	mCodecHandle = CodecHandle;
	Wake();
}

void MagicLeap::TOutputThread::OnOutputTextureAvailible()
{
	//	gr: from the forums, there's a suggestion that no calls should be done during a callback
	//		so queue up a grab-a-texture request
	mOutputTexturesAvailible++;
	Wake();
}

void MagicLeap::TOutputThread::OnOutputTextureWritten(int64_t PresentationTime)
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


bool MagicLeap::TOutputThread::CanSleep()
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


bool MagicLeap::TInputThread::CanSleep()
{
	if ( mInputBuffers.IsEmpty() )
		return true;
	
	if ( !HasPendingData() )
		return true;
	
	return false;
}

void MagicLeap::TOutputThread::RequestOutputTexture()
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

void MagicLeap::TOutputThread::PushOutputTexture(TOutputTexture& OutputTexture)
{
	//	for now, dummy texture, we need to push a handle (or make this readpixels on a gl thread)
	SoyPixels DummyPixels( SoyPixelsMeta( 1, 1, SoyPixelsFormat::RGBA) );
	DummyPixels.SetPixel( 0, 0, GetDebugColour(OutputTexture.mPresentationTime) );

	std::Debug << "PushOutputTexture(0x" << std::hex << OutputTexture.mTextureHandle << std::dec << " time=" << OutputTexture.mPresentationTime << ")" << std::endl;
	
	PushFrame( DummyPixels );
	
	//	gr: temp, we've "delivered" this texture, so release it now
	std::Debug << "ReleaseOutputTexture(0x" << std::hex << OutputTexture.mTextureHandle << std::dec << " time=" << OutputTexture.mPresentationTime << ")" << std::endl;
	ReleaseOutputTexture( OutputTexture.mTextureHandle );
}

bool MagicLeap::TOutputThread::IsAnyOutputTextureReady()
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

void MagicLeap::TOutputThread::PushOutputTextures()
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

void MagicLeap::TOutputThread::ReleaseOutputTexture(MLHandle TextureHandle)
{
	//	remove it from the list
	std::lock_guard<std::recursive_mutex> Lock(mOutputTexturesLock);
	if ( !mOutputTextures.Remove( TextureHandle ) )
	{
		std::Debug << "Warning: ReleaseOutputTexture(" << std::hex << "0x" << TextureHandle << std::dec << ") wasn't in our texture output list. MLMediaCodecReleaseFrame() NOT CALLED." << std::endl;
		return;
	}

	//	release it
	auto Result = MLMediaCodecReleaseFrame( mCodecHandle, TextureHandle );
	IsOkay( Result, "MLMediaCodecReleaseFrame" );
}

void MagicLeap::TOutputThread::PopOutputBuffer(const TOutputBufferMeta& BufferMeta)
{
	{
		std::lock_guard<std::mutex> Lock(mPushFunctionsLock);
		if ( mPushFunctions.IsEmpty() )
		{
			std::stringstream Error;
			Error << "PopOutputBuffer(" << BufferMeta.mBufferIndex << ") but no push funcs yet (probably race condition)";
			throw Soy::AssertException( Error );
		}
	}
	
	//	gr: hold onto pointer and don't release buffer until it's been read,to avoid a copy
	//		did this on android and it was a boost
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__,0);
	auto BufferHandle = static_cast<MLHandle>( BufferMeta.mBufferIndex );

	auto ReleaseBuffer = [&](bool Render=true)
	{
		//	release back!
		auto Result = MLMediaCodecReleaseOutputBuffer( mCodecHandle, BufferHandle, Render );
		IsOkay( Result, "MLMediaCodecReleaseOutputBuffer");
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
}


void MagicLeap::TOutputThread::PushFrame(const SoyPixelsImpl& Pixels)
{
	std::lock_guard<std::mutex> Lock(mPushFunctionsLock);
	if ( mPushFunctions.IsEmpty() )
	{
		std::stringstream Error;
		Error << "Got frame to output, but ran out of push functions. Should have been checked before";
		throw Soy::AssertException( Error );
	}
	
	auto PushFunc = mPushFunctions.PopAt(0);

	SoyTime DecodeDuration;
	PushFunc( Pixels, DecodeDuration );
}

bool MagicLeap::TOutputThread::Iteration()
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


MagicLeap::TInputThread::TInputThread(std::function<void(ArrayBridge<uint8_t>&&)> PopPendingData,std::function<bool()> HasPendingData) :
	mPopPendingData	( PopPendingData ),
	mHasPendingData	( HasPendingData ),
	SoyWorkerThread	("MagicLeapInputThread", SoyWorkerWaitMode::Wake )
{
	Start();
}

auto InputThreadNotReadySleep = 12;
auto InputThreadThrottle = 2;
auto InputThreadErrorThrottle = 1000;

bool MagicLeap::TInputThread::Iteration(std::function<void(std::chrono::milliseconds)> Sleep)
{
	if ( mInputBuffers.IsEmpty() )
	{
		std::Debug << __PRETTY_FUNCTION__ << " No input buffers" << std::endl;
		Sleep( std::chrono::milliseconds(InputThreadNotReadySleep) );
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



std::string MagicLeap::TDecoder::GetDebugState()
{
	std::stringstream Debug;
	Debug << mInputThread.GetDebugState() << mOutputThread.GetDebugState();
	return Debug.str();
}


std::string MagicLeap::TInputThread::GetDebugState()
{
	std::lock_guard<std::mutex> Lock(mInputBuffersLock);
	
	std::stringstream Debug;
	Debug << " mInputBuffers[";
	for ( auto i=0;	i<mInputBuffers.GetSize();	i++ )
		Debug << mInputBuffers[i] << ",";
	Debug << "] ";
	return Debug.str();
}

std::string MagicLeap::TOutputThread::GetDebugState()
{
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	std::lock_guard<std::recursive_mutex> Lock2(mOutputTexturesLock);

	std::stringstream Debug;
	Debug << " mOutputBuffers[";
	for ( auto i=0;	i<mOutputBuffers.GetSize();	i++ )
		Debug << mOutputBuffers[i].mBufferIndex << ",";
	Debug << "] ";
	Debug << "PushFuncs x" << mPushFunctions.GetSize() << " ";
	Debug << "mOutputTexturesAvailible=" << mOutputTexturesAvailible << " ";
	Debug << "mOutputTextures[";
	for ( auto i=0;	i<mOutputTextures.GetSize();	i++ )
		Debug << std::hex << "0x" << mOutputTextures[i].mTextureHandle << std::dec << ",";
	Debug << "] ";

	return Debug.str();
}
