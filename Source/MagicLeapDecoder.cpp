#include "MagicLeapDecoder.h"

#include <ml_media_codec.h>
#include <ml_media_codeclist.h>
#include <ml_media_format.h>
#include <ml_media_error.h>

namespace MagicLeap
{
	void	IsOkay(MLResult Result,const char* Context);
	void	IsOkay(MLResult Result,std::stringstream& Context);
	void	EnumCodecs(std::function<void(const std::string&)> Enum);
	
	//const auto*	DefaultH264Codec = "OMX.Nvidia.h264.decode";	//	hardware
	const auto*	DefaultH264Codec = "OMX.google.h264.decoder";	//	software according to https://forum.magicleap.com/hc/en-us/community/posts/360041748952-Follow-up-on-Multimedia-Decoder-API
	
	//	got this mime from googling;
	//	http://hello-qd.blogspot.com/2013/05/choose-decoder-and-encoder-by-google.html
	const auto* H264MimeType = "video/avc";
	
	//	CSD-0 (from android mediacodec api)
	//	not in the ML api
	//	https://forum.magicleap.com/hc/en-us/community/posts/360048067552-Setting-csd-0-byte-buffer-using-MLMediaFormatSetKeyByteBuffer
	MLMediaFormatKey	MLMediaFormat_Key_CSD0 = "csd-0";


	SoyPixelsMeta		GetPixelMeta(MLHandle Format);
}



void MagicLeap::IsOkay(MLResult Result,std::stringstream& Context)
{
	if ( Result == MLResult_Ok )
		return;
	
	auto Str = Context.str();
	IsOkay( Result, Str.c_str() );
}

void MagicLeap::IsOkay(MLResult Result,const char* Context)
{
	if ( Result == MLResult_Ok )
		return;

	//	specific media errors
	auto HasPrefix = [&](MLResult Prefix)
	{
		auto And = Result & Prefix;
		return And == Prefix;
	};

	const char* ResultString = nullptr;

	if ( HasPrefix(MLResultAPIPrefix_MediaGeneric) ||
		HasPrefix(MLResultAPIPrefix_Media) ||
		HasPrefix(MLResultAPIPrefix_MediaDRM) ||
		HasPrefix(MLResultAPIPrefix_MediaOMX) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXExtensions) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXVendors) ||
		HasPrefix(MLResultAPIPrefix_MediaPlayer) )
	{
		ResultString = MLMediaResultGetString( Result );
	}
	
	if ( !ResultString )
		ResultString = MLGetResultString(Result);
	
	auto Results16 = static_cast<int16_t>( static_cast<uint32_t>(Result) >> 16 );
	auto Resultu16 = static_cast<uint16_t>( static_cast<uint32_t>(Result) >> 16 );

	//	gr: sometimes we get unknown so, put error nmber in
	std::stringstream Error;
	Error << "Error in " << Context << ": " << ResultString << " (0x" << std::hex <<  static_cast<uint32_t>(Result) << std::dec << "/" << Results16 << "/" << Resultu16 << ")";
	throw Soy::AssertException( Error );
}

/*
SoyPixelsFormat::Type MagicLeap::GetPixelFormat(int32_t ColourFormat)
{

 enum
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
		case MLSurfaceFormat_Unknown:	throw Soy::AssertException("Format is MLSurfaceFormat_Unknown");
			COLOR_Format12bitRGB444
			case
	}{
		case <#constant#>:
			<#statements#>
			break;
			
		default:
			break;
	}
}
*/
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
	DebugKey(MLMediaFormat_Key_Crop_Left);
	DebugKey(MLMediaFormat_Key_Crop_Right);
	DebugKey(MLMediaFormat_Key_Crop_Bottom);
	DebugKey(MLMediaFormat_Key_Crop_Top);
	//GetKey_long(MLMediaFormat_Key_Repeat_Previous_Frame_After
	
	//	there's a colour format key, but totally undocumented
	//	SDK says for  MLMediaCodecAcquireNextAvailableFrame;
	//		Note: The returned buffer's color format is multi-planar YUV420. Since our
	//		underlying hardware interops do not support multiplanar formats, advanced
	auto PixelFormat = SoyPixelsFormat::Yuv_8_88_Ntsc;

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



MagicLeap::TDecoder::TDecoder()
{
	auto EnumCodec = [](const std::string& Name)
	{
		std::Debug << "Codec: " << Name << std::endl;
	};
	EnumCodecs( EnumCodec );
	
	
	std::string CodecName = DefaultH264Codec;
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

	auto Result = MLMediaCodecCreateCodec( CreateByType, MLMediaCodecType_Decoder, CodecName.c_str(), &mHandle );
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
		This.OnOutputBufferAvailible(BufferIndex);
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
	};
	
	auto OnFrameAvailible = [](MLHandle Codec,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnFrameAvailible" << std::endl;
	};
	

	//	gr: not sure if this can be a temporary or not, demo on forums has a static
	MLMediaCodecCallbacks Callbacks;
	Callbacks.on_input_buffer_available = OnInputBufferAvailible;
	Callbacks.on_output_buffer_available = OnOutputBufferAvailible;
	Callbacks.on_output_format_changed = OnOutputFormatChanged;
	Callbacks.on_error = OnError;
	Callbacks.on_frame_rendered = OnFrameRendered;
	Callbacks.on_frame_available = OnFrameAvailible;
	
	Result = MLMediaCodecSetCallbacks( mHandle, &Callbacks, this );
	IsOkay( Result, "MLMediaCodecSetCallbacks" );

	//	configure
	MLHandle Format = ML_INVALID_HANDLE;
	Result = MLMediaFormatCreateVideo( H264MimeType, 1920, 1080, &Format );
	IsOkay( Result, "MLMediaFormatCreateVideo" );
	std::Debug << "Got format: " << Format << std::endl;

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
	Result = MLMediaCodecSetSurfaceHint( mHandle, MLMediaCodecSurfaceHint_Software );
	IsOkay( Result, "MLMediaCodecSetSurfaceHint(software)" );

	
	//MLHandle Crypto = ML_INVALID_HANDLE;
	MLHandle Crypto = 0;	//	gr: INVALID_HANDLE doesnt work
	Result = MLMediaCodecConfigure( mHandle, Format, Crypto );
	IsOkay( Result, "MLMediaCodecConfigure" );
	
	Result = MLMediaCodecStart( mHandle );
	IsOkay( Result, "MLMediaCodecStart" );
	
	//	MLMediaCodecFlush makes all inputs invalid... flush on close?
	std::Debug << "Created TDecoder (" << CodecName << ")" << std::endl;
}

MagicLeap::TDecoder::~TDecoder()
{
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
}


//	returns true if more data to proccess
bool MagicLeap::TDecoder::DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)
{
	if ( mPendingData.IsEmpty() )
		return false;
	
	auto GetNextNalOffset = [this]
	{
		for ( int i=3;	i<mPendingData.GetDataSize();	i++ )
		{
			if ( mPendingData[i+0] != 0 )	continue;
			if ( mPendingData[i+1] != 0 )	continue;
			if ( mPendingData[i+2] != 0 )	continue;
			if ( mPendingData[i+3] != 1 )	continue;
			return i;
		}
		//	assume is complete...
		return (int)mPendingData.GetDataSize();
		//return 0;
	};

	//	https://forum.magicleap.com/hc/en-us/community/posts/360041748952-Follow-up-on-Multimedia-Decoder-API
	
	//	waiting for input buffers
	//	gr: this is maybe just a signal there IS some processing space...
	if ( mInputBuffers.IsEmpty() )
	{
		std::Debug << __func__ << " waiting for input buffers" << std::endl;
		return false;
	}
	
	int64_t BufferIndex;
	{
		std::lock_guard<std::mutex> Lock(mInputBufferLock);
		BufferIndex = mInputBuffers.PopAt(0);
	}
	/*	gr: dont dequeue in async mode.
	 //	https://forum.magicleap.com/hc/en-us/community/posts/360055134771-Stagefright-assert-after-calling-MLMediaCodecDequeueInputBuffer?page=1#community_comment_360008477771
	 //	This asserts on magic leap, on android it just carries on, but sholdnt be being used
	//auto Timeout = 0;	//	return immediately
	auto Timeout = -1;	//	block
	
	//	API says MLMediaCodecDequeueInputBuffer output is index
	//	but everything referring to it, says it's a handle (uint64!)
	int64_t BufferIndex = -1;
	std::Debug << __func__ << " MLMediaCodecDequeueInputBuffer( timeout=" << Timeout << ")" << std::endl;
	auto Result = MLMediaCodecDequeueInputBuffer( mHandle, Timeout, &BufferIndex );
	IsOkay( Result, "MLMediaCodecDequeueInputBuffer" );
	
	if ( BufferIndex == MLMediaCodec_TryAgainLater )
	{
		std::Debug << __func__ << " MLMediaCodec_TryAgainLater" << std::endl;
		return true;
	}

	if ( BufferIndex == MLMediaCodec_FormatChanged )
	{
		std::Debug << __func__ << " MLMediaCodec_FormatChanged" << std::endl;
		return true;
	}
	
	//	gr: need to reset all buffers?
	if ( BufferIndex == MLMediaCodec_OutputBuffersChanged )
	{
		std::Debug << __func__ << " MLMediaCodec_OutputBuffersChanged" << std::endl;
		return true;
	}

	if ( BufferIndex < 0 )
	{
		std::stringstream Error;
		Error << "MLMediaCodecDequeueInputBuffer dequeued buffer index " << BufferIndex;
		throw Soy::AssertException(Error);
	}
	*/
	try
	{
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

		size_t DataSize = 0;
		{
			std::lock_guard<std::mutex> Lock( mPendingDataLock );
			auto GetNextNalOffset = [this]
			{
				for ( int i=3;	i<mPendingData.GetDataSize();	i++ )
				{
					if ( mPendingData[i+0] != 0 )	continue;
					if ( mPendingData[i+1] != 0 )	continue;
					if ( mPendingData[i+2] != 0 )	continue;
					if ( mPendingData[i+3] != 1 )	continue;
					return i;
				}
				//	assume is complete...
				return (int)mPendingData.GetDataSize();
				//return 0;
			};
			
			DataSize = GetNextNalOffset();
			auto* Data = mPendingData.GetArray();
			if ( DataSize > BufferSize )
			{
				std::stringstream Error;
				Error << "MLMediaCodecGetInputBufferPointer buffer size(" << BufferSize << ") too small for pending nal packet size (" << DataSize << ")";
				throw Soy::AssertException(Error);
			}

			//	fill buffer
			memcpy( Buffer, Data, DataSize );
		}
		
		//	process buffer
		int64_t DataOffset = 0;
		uint64_t PresentationTimeMicroSecs = mPacketCounter;
		mPacketCounter++;
		int Flags = 0;
		//Flags |= MLMediaCodecBufferFlag_KeyFrame;
		//Flags |= MLMediaCodecBufferFlag_CodecConfig;
		//Flags |= MLMediaCodecBufferFlag_EOS;
		Result = MLMediaCodecQueueInputBuffer( mHandle, BufferHandle, DataOffset, DataSize, PresentationTimeMicroSecs, Flags );
		IsOkay( Result, "MLMediaCodecQueueInputBuffer" );
		
		std::Debug << "MLMediaCodecQueueInputBuffer( BufferIndex=" << BufferIndex << " DataSize=" << DataSize << " presentationtime=" << PresentationTimeMicroSecs << ") success" << std::endl;
		RemovePendingData( DataSize );
	}
	catch(std::exception& e)
	{
		//	gr: maybe MLMediaCodecFlush()
		std::Debug << "Exception processing input buffer: " << e.what() << ". Flush frames here?" << std::endl;
	}
	
	return true;
}

void MagicLeap::TDecoder::OnInputBufferAvailible(int64_t BufferIndex)
{
	std::Debug << "OnInputBufferAvailible(" << BufferIndex << ")" << std::endl;
	std::lock_guard<std::mutex> Lock(mInputBufferLock);
	mInputBuffers.PushBack(BufferIndex);
}

void MagicLeap::TDecoder::OnOutputBufferAvailible(int64_t BufferIndex)
{
	mOutputThread.OnOutputBufferAvailible( mHandle, mOutputPixelMeta, BufferIndex );
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


/*
void Broadway::TDecoder::OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded,SoyTime DecodeDuration)
{
	//		headers just say
	//	u32 *pOutputPicture;  	//	  Pointer to the picture, YUV format
	auto Format = SoyPixelsFormat::Yuv_8_8_8_Full;
	SoyPixelsMeta PixelMeta( Meta.picWidth, Meta.picHeight, Format );
	//std::Debug << "Decoded picture " << PixelMeta << std::endl;
	
	//	gr: wish we knew exactly how many bytes Picture.pOutputPicture pointed at!
	//		but demos all use this measurement
	//auto DataSize = PixelMeta.GetDataSize();
	auto DataSize = (3 * Meta.picWidth * Meta.picHeight)/2;

	auto* Pixels8 = reinterpret_cast<uint8_t*>(Picture.pOutputPicture);
	SoyPixelsRemote Pixels( Pixels8, DataSize, PixelMeta );
	if ( OnFrameDecoded )
		OnFrameDecoded( Pixels, DecodeDuration );
}

*/




/*
 
 
 
 #include <stdlib.h>
 
 #include <unistd.h>
 
 
 
 
 #include <ml_codec.h>
 
 #include <ml_media_extractor.h>
 
 #include <ml_media_format.h>
 
 #include <ml_media_crypto.h>
 
 #include <ml_media_error.h>
 
 #include <Logging.h>
 
 
 
 
 #define MIN(a,b) (((a)<(b))?(a):(b))
 
 
 
 
 #define CHECK_SUCCESS(func)                         \
 
 do {                                              \
 
 MLResult result_ = (func);                      \
 
 if (MLResult_Ok != result_) {                   \
 
 char str[512] = (#func);                      \
 
 const char delim[2] = "(";                    \
 
 char *funcName = strtok(str, delim);          \
 
 ML_LOG_ERROR("%s failed with %d", funcName, result_); \
 
 return result_;                                 \
 
 }                                               \
 
 } while(0)
 
 
 
 
 #define CHECK_FAIL(func)                   \
 
 do {                                     \
 
 if (false == (func)) {                 \
 
 char str[200] = (#func);             \
 
 const char delim[2] = "(";           \
 
 char *funcName = strtok(str, delim); \
 
 ML_LOG_ERROR("%s failed", funcName);         \
 
 return false;                        \
 
 }                                      \
 
 } while(0)
 
 
 
 
 // callbacks forward declaration
 
 static void onInputBufferAvailable(MLHandle codec, int64_t index, void* cbData);
 
 static void onOutputBufferAvailable(MLHandle codec, int64_t index, MLMediaCodecBufferInfo* bufferInfo, void* cbData);
 
 static void onOutputFormatChanged(MLHandle codec, MLHandle format, void* cbData);
 
 static void onError(MLHandle codec, int errorCode, void* cbData);
 
 static void onFrameRendered(MLHandle codec, int64_t ptsUs, int64_t systemTimeNs, void* cbData);
 
 static void onVideoFrameAvailable(MLHandle codec, void* cbData);
 
 
 
 
 class MediaCodecPlayer {
 
 
 
 
 public:
 
 static MediaCodecPlayer* createStreamingMediaPlayer();
 
 MLResult setSource(const char *fileName);
 
 MLResult start();
 
 
 
 
 inline uint64_t getDuration() const { return mDurationUs; };
 
 inline int getPosition() const { return mPosition; };
 
 
 
 
 // TODO: Add pause, seek, stop and other functions
 
 MLResult seek() {}
 
 MLResult pause() {}
 
 MLResult resume() {}
 
 MLResult stop() {}
 
 
 
 
 // callback handlers - I recommend handling all these in an another thread
 
 void onInputBufferAvailable(MLHandle codec, int64_t index);
 
 void onOutputBufferAvailable(MLHandle codec, int64_t index, MLMediaCodecBufferInfo* bufferInfo);
 
 void onOutputFormatChanged(MLHandle codec, MLHandle format);
 
 void onError(MLHandle codec, int errorCode);
 
 void onFrameRendered(MLHandle codec, int64_t ptsUs, int64_t systemTimeNs);
 
 void onVideoFrameAvailable(MLHandle codec);
 
 
 
 
 private:
 
 MediaCodecPlayer();
 
 ~MediaCodecPlayer();
 
 MediaCodecPlayer(const MediaDrmClient&) = delete;
 
 MediaCodecPlayer(MediaDrmClient&&) = delete;
 
 MediaCodecPlayer& operator=(const MediaDrmClient&) = delete;
 
 MediaCodecPlayer& operator=(MediaDrmClient&&) = delete;
 
 
 
 
 MLResult setupMediaExtractor(const char* fileName);
 
 void playbackLoop();
 
 MLResult queueSecureInputBuffer(MLHandle codec, int64_t inputBufIndex, int64_t presentationTimeUs);
 
 MLResult processInputSample(MLHandle* codec);
 
 MLResult processOutputSample(MLHandle codec);
 
 void handleDecodedFrame(MLHandle codec, const MLMediaCodecBufferInfo* info, int64_t bufidx);
 
 void cleanUp();
 
 
 
 
 private:
 
 MLHandle            mExtractor = ML_INVALID_HANDLE;
 
 MLHandle            mVideoCodec = ML_INVALID_HANDLE;
 
 MLHandle            mAudioCodec = ML_INVALID_HANDLE;
 
 MLHandle            mCrypto = ML_INVALID_HANDLE;
 
 MLHandle            mDrm = ML_INVALID_HANDLE;
 
 MLHandle            mNativeFrameBuffer = ML_INVALID_HANDLE;
 
 MLHandle            mAudioHandle = ML_INVALID_HANDLE;
 
 uint64_t            mLastPresentationTimeUs = 0L;
 
 uint64_t            mDurationUs = 0L;
 
 int64_t             mAudioTrakIndex = -1L;
 
 int64_t             mVideoTrakIndex = -1L;
 
 int                 mPosition = 0;
 
 bool                mSawInputEOS = false;
 
 bool                mSawOutputEOS = false;
 
 std::vector<int64_t> mVideoIndexList;
 
 std::vector<int64_t> mAudioIndexList;
 
 };
 
 
 
 
 MediaCodecPlayer::MediaCodecPlayer() { }
 
 
 
 
 MediaCodecPlayer::~MediaCodecPlayer() { cleanup(); }
 
 
 
 
 bool MediaCodecPlayer::construct() {
 
 // TODO: Spawn a thread which will wait on start/pause/seek/stop and handling of callback events
 
 return true;
 
 }
 
 
 
 
 static MLMediaCodecCallbacks callbacksSync = {
 
 nullptr,
 
 nullptr,
 
 nullptr,
 
 nullptr,
 
 nullptr,
 
 onVideoFrameAvailable,
 
 };
 
 
 
 
 static MLMediaCodecCallbacks callbacksAsync = {
 
 onInputBufferAvailable,
 
 onOutputBufferAvailable,
 
 onOutputFormatChanged,
 
 onError,
 
 onFrameRendered,
 
 onVideoFrameAvailable,
 
 };
 
 
 
 
 static const int64_t kTimeOut = 2000LL;
 
 
 
 
 static inline int64_t systemnanotime() {
 
 struct timespec now;
 
 clock_gettime(CLOCK_MONOTONIC, &now);
 
 return now.tv_sec * 1000000000LL + now.tv_nsec;
 
 }
 
 
 
 
 static void onInputBufferAvailable(MLHandle codec, int64_t index, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 onInputBufferAvailable(codec, index);
 
 }
 
 }
 
 
 
 
 void MediaCodecPlayer::onInputBufferAvailable(MLHandle codec, int64_t index) {
 
 // Maintain 2 list of indexes - one for audio and one for video
 
 // Push this index into corresponding queue
 
 if (codec == mVideoCodec) {
 
 mVideoIndexList.push_back(index);
 
 } else if (codec == mAudioCodec) {
 
 mAudioIndexList.push_back(index);
 
 }
 
 
 
 // Do this as long as there is an entry in either of the index list in a loop
 
 size_t readCount = mAudioIndexList.size() + mVideoIndexList.size();
 
 while (readCount != 0) {
 
 int64_t trackIndex = -1;
 
 if (MLResultOk == MLMediaExtractorGetSampleTrackIndex(mExtractor, &trackIndex)) {
 
 MLHandle codec = ML_INVALID_HANDLE;
 
 if (mAudioTrakIndex == trackIndex) {
 
 codec = mAudioCodec;
 
 index = mAudioIndexList.front();
 
 mAudioIndexList.pop_front();
 
 } else if (mVideoTrakIndex == trackIndex) {
 
 codec = mVideoCodec;
 
 index = mVideoIndexList.front();
 
 mVideoIndexList.pop_front();
 
 }
 
 readSampleAndFeedCodec(codec, index);
 
 }
 
 readCount--;
 
 }
 
 }
 
 
 
 
 static void onOutputBufferAvailable(MLHandle codec, int64_t index, MLMediaCodecBufferInfo* bufferInfo, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 self->onOutputBufferAvailable(codec, index, bufferInfo);
 
 }
 
 }
 
 
 
 
 void MediaCodecPlayer::onOutputBufferAvailable(MLHandle codec, int64_t index, MLMediaCodecBufferInfo* bufferInfo) {
 
 handleDecodedFrame(codec, bufferInfo, index);
 
 }
 
 
 
 
 static void onOutputFormatChanged(MLHandle codec, MLHandle format, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 self->onOutputFormatChanged(codec, format);
 
 }
 
 }
 
 
 
 
 void MediaCodecPlayer::onOutputFormatChanged(MLHandle codec, MLHandle format) {
 
 // This is where we get to know the actual format of the media
 
 // What we get through MLMediaExtractorGetTrackFormat is an gestimation
 
 // Use the format to create MLAudio handle
 
 }
 
 
 
 
 static void onError(MLHandle codec, int errorCode, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 self->onError(codec, errorCode);
 
 }
 
 }
 
 
 
 
 void MediaCodecPlayer::onError(MLHandle codec, int errorCode) {
 
 // TODO: Notify the user about the possible error
 
 }
 
 
 
 
 static void onFrameRendered(MLHandle codec, int64_t ptsUs, int64_t systemTimeNs, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 self->onFrameRendered(codec, ptsUs, systemTimeNs);
 
 }
 
 }
 
 
 
 
 void MediaCodecPlayer::onFrameRendered(MLHandle codec, int64_t ptsUs, int64_t systemTimeNs) {
 
 // TODO: Its just info only call - you can ignore it
 
 }
 
 
 
 
 static void onVideoFrameAvailable(MLHandle codec, void* cbData) {
 
 MediaCodecPlayer* self = (MediaCodecPlayer*)cbData;
 
 if (self != nullptr) {
 
 // TODO: Make it an async call by letting another thread handle it as noted earlier
 
 self->onVideoFrameAvailable(codec);
 
 }
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::onVideoFrameAvailable(MLHandle codec) {
 
 // codec has to be mVideoCodec  if (mVideoCodec != codec) { ERROR!! }
 
 CHECK_SUCCESS(MLMediaCodecAcquireNextAvailableFrame(mVideoCodec, &mNativeFrameBuffer));
 
 // TODO: Render the native buffer now ==> render(mNativeFrameBuffer);
 
 CHECK_SUCCESS(MLMediaCodecReleaseFrame(mVideoCodec, mNativeFrameBuffer));
 
 return MLResult_Ok;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::queueSecureInputBuffer(MLHandle codec, int64_t inputBufIndex, int64_t presentationTimeUs) {
 
 MLHandle info = 0;
 
 // Encrypted sample - get the crypto info
 
 CHECK_SUCCESS(MLMediaExtractorGetSampleCryptoInfo(mExtractor, &info));
 
 CHECK_SUCCESS(MLMediaCodecQueueSecureInputBuffer(codec,
 
 (MLHandle)inputBufIndex, 0, info, presentationTimeUs, 0));
 
 MLMediaExtractorReleaseCryptoInfo(mExtractor, &info);
 
 return MLResult_Ok;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::readSampleAndFeedCodec(MLHandle codec, int64_t bufidx) {
 
 size_t   bufsize      = 0;
 
 uint8_t* buf         = nullptr;
 
 int64_t  sampleSize     = -1;
 
 int64_t  presentationTimeUs = -1;
 
 
 
 
 CHECK_SUCCESS(MLMediaCodecGetInputBufferPointer(codec, bufidx, &buf, &bufsize));
 
 if (nullptr == buf) {
 
 ML_LOG_ERROR("Input buffer pointer is nullptr!");
 
 return MLResult_UnspecifiedFailure;
 
 }
 
 
 
 
 CHECK_SUCCESS(MLMediaExtractorReadSampleData(mExtractor, buf, bufsize, 0, &sampleSize));
 
 
 
 
 if (sampleSize < 0) {
 
 sampleSize = 0;
 
 mSawInputEOS = true;
 
 ML_LOG_INFO("Found (Source) Media Content EOS");
 
 }
 
 
 
 
 CHECK_SUCCESS(MLMediaExtractorGetSampleTime(mExtractor, &presentationTimeUs));
 
 int sample_flags = 0;
 
 CHECK_SUCCESS(MLMediaExtractorGetSampleFlags(mExtractor, &sample_flags));
 
 if (!mSawInputEOS && mCrypto &&
 
 (sample_flags & MLMediaExtractorSampleFlag_Encrypted)) {
 
 // TODO: NOTE that I didn't do MediaCrypto setup while configuring
 
 CHECK_SUCCESS(queueSecureInputBuffer(codec, bufidx, presentationTimeUs));
 
 } else {
 
 CHECK_SUCCESS(MLMediaCodecQueueInputBuffer(codec, (MLHandle)bufidx, 0,
 
 sampleSize, presentationTimeUs,
 
 mSawInputEOS ? MLMediaCodecBufferFlag_EOS : 0));
 
 }
 
 
 
 
 MLMediaExtractorAdvance(mExtractor);
 
 return MLResult_Ok;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::processInputSample(MLHandle* codec) {
 
 int64_t bufidx       = MLMediaCodec_TryAgainLater;
 
 size_t  bufsize      = 0;
 
 uint8_t *buf         = nullptr;
 
 int64_t sampleSize     = -1;
 
 int64_t presentationTimeUs = -1;
 
 int64_t track_index = -1;
 
 
 
 
 *codec = ML_INVALID_HANDLE;
 
 if (mSawInputEOS) {
 
 return MLResult_Ok;
 
 }
 
 
 
 
 CHECK_SUCCESS(MLMediaExtractorGetSampleTrackIndex(mExtractor, &track_index));
 
 if (track_index < 0) {
 
 ML_LOG_INFO("Encountered Media Input EOS!");
 
 mSawInputEOS = true;
 
 return MLResult_Ok;
 
 }
 
 
 
 
 if (track_index == mAudioTrakIndex) {
 
 *codec = mAudioCodec;
 
 } else if (track_index == mVideoTrakIndex) {
 
 *codec = mVideoCodec;
 
 }
 
 
 
 
 if (*codec == ML_INVALID_HANDLE) {
 
 ML_LOG_ERROR("We haven't setup Codec for this track[%ld]!", track_index);
 
 return MLResult_Ok;
 
 }
 
 
 
 
 do {
 
 // Wait until we have a free slot in codec
 
 CHECK_SUCCESS(MLMediaCodecDequeueInputBuffer(*codec, kTimeOut, &bufidx));
 
 if (bufidx == MLMediaCodec_TryAgainLater) {
 
 usleep(100);
 
 } else {
 
 break;
 
 }
 
 } while (true);
 
 
 
 
 return readSampleAndFeedCodec(*codec, bufidx);
 
 }
 
 
 
 
 
 
 
 void MediaCodecPlayer::handleDecodedFrame(MLHandle codec, const MLMediaCodecBufferInfo* info, int64_t bufidx) {
 
 if (0 <= bufidx) {
 
 if (info->flags & MLMediaCodecBufferFlag_EOS) {
 
 ML_LOG_INFO("Found Output EOS");
 
 mSawOutputEOS = true;
 
 }
 
 
 
 
 int64_t presentationNano = info->presentation_time_us * 1000;
 
 mPosition = info->presentation_time_us / 1000;
 
 
 
 
 // TODO: If you want to render this - device a AV sync mechanism
 
 // Something like delay the rendering if its way ahead of current PTS
 
 
 
 
 if (mSawInputEOS && !mSawOutputEOS) {
 
 ML_LOG_INFO("Current Position[%ld] Duration[%ld] LastPresentation[%ld]",
 
 info->presentation_time_us, mDurationUs, mLastPresentationTimeUs);
 
 if ((uint64_t)info->presentation_time_us >= mDurationUs) {
 
 ML_LOG_INFO("Current Position[%ld] crossed Duration[%ld] - Found Output EOS",
 
 info->presentation_time_us, mDurationUs);
 
 mSawOutputEOS = true;
 
 } else if (mLastPresentationTimeUs == (uint64_t)info->presentation_time_us) {
 
 ML_LOG_INFO("The Current Position hasn't moved since last decode - Found Output EOS");
 
 mSawOutputEOS = true;
 
 }
 
 }
 
 if (bIsVideoTrack) {
 
 // Video stream decoding: synchronous rendering and releasing
 
 // Note that, in this case we get this callback ==> onVideoFrameAvailable
 
 CHECK_SUCCESS(MLMediaCodecReleaseOutputBuffer(codec, (MLHandle)bufidx, info->size != 0));
 
 } else {
 
 // Audio stream decoding: Get the decoded sample and play the audio
 
 size_t bufSize = 0;
 
 const uint8_t* pBuf = nullptr;
 
 CHECK_SUCCESS(MLMediaCodecGetOutputBufferPointer(codec, (MLHandle)bufidx, &pBuf, &bufSize));
 
 if (pBuf) {
 
 // write the decoded audio sample into Audio output ==> TODO: use MLAudio APIs
 
 }
 
 // release the output buffer now
 
 CHECK_SUCCESS(MLMediaCodecReleaseOutputBuffer(codec, bufidx, false));
 
 }
 
 mLastPresentationTimeUs = info->presentation_time_us;
 
 }
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::processOutputSample(MLHandle codec) {
 
 bool bIsVideoTrack = false;
 
 int64_t bufidx = MLMediaCodec_TryAgainLater;
 
 MLMediaCodecBufferInfo info = {};
 
 
 
 
 if (codec == ML_INVALID_HANDLE) {
 
 return MLResult_UnspecifiedFailure;
 
 }
 
 
 
 
 if (mSawOutputEOS) {
 
 return MLResult_Ok;
 
 }
 
 
 
 
 if (codec == mVideoCodec) {
 
 bIsVideoTrack = true;
 
 }
 
 
 
 
 CHECK_SUCCESS(MLMediaCodecDequeueOutputBuffer(codec, &info, 0, &bufidx));
 
 if (0 <= bufidx) {
 
 handleDecodedFrame(codec, &info, bufidx);
 
 } else if (bufidx == MLMediaCodec_OutputBuffersChanged) {
 
 ML_LOG_INFO("Output buffers changed");
 
 } else if (bufidx == MLMediaCodec_FormatChanged) {
 
 MLHandle format = 0;
 
 CHECK_SUCCESS(MLMediaCodecGetOutputFormat(codec, &format));
 
 char* newFormat = (char*)malloc(MAX_FORMAT_STRING_SIZE);
 
 CHECK_SUCCESS(MLMediaFormatObjectToString(format, newFormat));
 
 ML_LOG_INFO("Format changed to: %s", newFormat);
 
 // If we picked an audio track, let's setup the audio output
 
 int32_t sampleRate   = 0;
 
 int32_t channelCount = 0;
 
 if (MLResult_Ok == MLMediaFormatGetKeyValueInt32(format, MLMediaFormat_Key_Sample_Rate, &sampleRate)) {
 
 if (MLResult_Ok == MLMediaFormatGetKeyValueInt32(format, MLMediaFormat_Key_Channel_Count, &channelCount)) {
 
 ML_LOG_INFO("Audio sampling rate[%i] channel count[%i]", sampleRate, channelCount);
 
 if (sampleRate != 0 && channelCount != 0) {
 
 createAudioStream(sampleRate, channelCount);
 
 }
 
 }
 
 }
 
 free(newFormat);
 
 } else if (bufidx == MLMediaCodec_TryAgainLater) {
 
 // If the input has been consumed already, signal end of output as well
 
 if (mSawInputEOS) {
 
 mSawOutputEOS = true;
 
 }
 
 } else {
 
 ML_LOG_ERROR("Unexpected info code: %zd", bufidx);
 
 return MLResult_UnspecifiedFailure;
 
 }
 
 
 
 
 return MLResult_Ok;
 
 }
 
 
 
 
 void MediaCodecPlayer::playbackLoop() {
 
 while (!sawInputEOS || !mSawOutputEOS) {
 
 MLHandle codec = ML_INVALID_HANDLE;
 
 auto res = processInputSample(&codec);
 
 if (res != MLResult_Ok) {
 
 ML_LOG_ERROR("processInputSample failed!");
 
 } else {
 
 if (codec != ML_INVALID_HANDLE && processOutputSample(codec) != MLResult_Ok) {
 
 ML_LOG_ERROR("processOutputSample failed!");
 
 } else if (codec == ML_INVALID_HANDLE && mSawInputEOS) {
 
 while (!mSawOutputEOS) {
 
 processOutputSample(mAudioCodec);
 
 processOutputSample(mVideoCodec);
 
 }
 
 }
 
 }
 
 }
 
 return;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::setupMediaExtractor(const char* fileName) {
 
 // Use media extractor to acquire media info from the content
 
 MLResult result = MLMediaExtractorCreate(&mExtractor);
 
 if (result == MLResult_Ok && mExtractor != ML_INVALID_HANDLE) {
 
 result == MLMediaExtractorSetDataSourceForPath(mExtractor, fileName);
 
 }
 
 return result;
 
 }
 
 
 
 
 
 
 
 MediaCodecPlayer* MediaCodecPlayer::createStreamingMediaPlayer() {
 
 MediaCodecPlayer* self = new MediaCodecPlayer();
 
 if (self) {
 
 if (self->construct() != true) {
 
 delete self;
 
 self = nullptr;
 
 }
 
 }
 
 return self;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::setSource(const char *fileName) {
 
 MLHandle format;
 
 MLResult result = MLResult_UnspecifiedFailure;
 
 int64_t  durationUs = 0;
 
 size_t   numTracks = 0;
 
 int32_t  sampleRate = 0;
 
 int32_t  channelCount = 0;
 
 char     mime[MAX_KEY_STRING_SIZE] = "";
 
 
 
 
 if (nullptr == fileName || strlen(fileName) <= 0) {
 
 return result;
 
 }
 
 
 
 
 CHECK_SUCCESS(setupMediaExtractor(fileName));
 
 CHECK_SUCCESS(MLMediaExtractorGetTrackCount(mExtractor, &numTracks));
 
 
 
 
 for (size_t trackIndex = 0; trackIndex < numTracks; trackIndex++) {
 
 bool bIsVideo = false;
 
 bool bIsAudio = false;
 
 CHECK_SUCCESS(MLMediaExtractorGetTrackFormat(mExtractor, trackIndex, &format));
 
 CHECK_SUCCESS(MLMediaFormatGetKeyString(format, MLMediaFormat_Key_Mime, mime));
 
 CHECK_SUCCESS(MLMediaFormatGetKeyValueInt64(format, MLMediaFormat_Key_Duration, &durationUs));
 
 if (0 == strncmp(mime, "video/", 6)) {
 
 bIsVideo = true;
 
 } else if (0 == strncmp(mime, "audio/", 6)) {
 
 bIsAudio = true;
 
 CHECK_SUCCESS(MLMediaFormatGetKeyValueInt32(format, MLMediaFormat_Key_Sample_Rate, &sampleRate));
 
 CHECK_SUCCESS(MLMediaFormatGetKeyValueInt32(format, MLMediaFormat_Key_Channel_Count, &channelCount));
 
 ML_LOG_INFO("Audio sampling rate[%i] channel count[%i]", sampleRate, channelCount);
 
 }
 
 
 
 
 if (bIsVideo || bIsAudio) {
 
 MLHandle codec = ML_INVALID_HANDLE;
 
 CHECK_SUCCESS(MLMediaExtractorSelectTrack(mExtractor, trackIndex));
 
 result = MLMediaCodecCreateCodec(MLMediaCodecCreation_ByType,
 
 MLMediaCodecType_Decoder, mime, &codec);
 
 if (MLResult_Ok != result || codec == ML_INVALID_HANDLE) {
 
 ML_LOG_ERROR("MLMediaCodecCreateCodec failed");
 
 return result;
 
 }
 
 
 
 
 // For Codec in sync mode use this function
 
 // CHECK_SUCCESS(MLMediaCodecSetCallbacks(codec, &callbacksSync, nullptr));
 
 // For Codec in async mode use this function
 
 CHECK_SUCCESS(MLMediaCodecSetCallbacks(codec, &callbacksAsync, nullptr));
 
 // TODO: NOTE that based on whether Extractor says if the content is encrypted, we need to set up the DRM/Crypto
 
 CHECK_SUCCESS(MLMediaCodecConfigure(codec, format, mCrypto));
 
 
 
 
 mDurationUs  = durationUs;
 
 mPosition     = 0;
 
 mSawInputEOS  = false;
 
 mSawOutputEOS = false;
 
 if (bIsVideo) {
 
 mVideoCodec = codec;
 
 mVideoTrakIndex = trackIndex;
 
 } else if (bIsAudio) {
 
 mAudioCodec = codec;
 
 mAudioTrakIndex = trackIndex;
 
 }
 
 }
 
 }
 
 
 
 
 if (result != MLResult_Ok) {
 
 // Failed to set up the playback - do the clean up before return
 
 if (format) {
 
 MLMediaFormatDestroy(format);
 
 }
 
 cleanUp();
 
 }
 
 
 
 
 return result;
 
 }
 
 
 
 
 MLResult MediaCodecPlayer::start() {
 
 // TODO: Ideally trigger an async event that will call the following function in another thread
 
 playbackLoop();
 
 }
 
 
 
 
 void MediaCodecPlayer::cleanUp() {
 
 // TODO: Stop the playback if its still happening
 
 // TODO: Stop the thread that is spawned by construct()
 
 if (mVideoCodec != ML_INVALID_HANDLE) {
 
 MLMediaCodecDestroy(mVideoCodec);
 
 mVideoCodec = ML_INVALID_HANDLE;
 
 }
 
 if (mAudioCodec != ML_INVALID_HANDLE) {
 
 MLMediaCodecDestroy(mAudioCodec);
 
 mAudioCodec = ML_INVALID_HANDLE;
 
 }
 
 if (mExtractor != ML_INVALID_HANDLE) {
 
 MLMediaExtractorDestroy(mExtractor);
 
 mExtractor = ML_INVALID_HANDLE;
 
 }
 
 if (mDrm != ML_INVALID_HANDLE) {
 
 MLMediaDRMRelease(mDrm);
 
 mDrm = ML_INVALID_HANDLE;
 
 }
 
 if (mCrypto != 0) {
 
 MLMediaCryptoRelease(mCrypto);
 
 mCrypto = 0;
 
 }
 
 mSawInputEOS  = true;
 
 mSawOutputEOS = true;
 
 }
*/

MagicLeap::TOutputThread::TOutputThread() :
	SoyWorkerThread	("MagicLeapOutputThread", SoyWorkerWaitMode::Wake )
{
	Start();
}

void MagicLeap::TOutputThread::OnOutputBufferAvailible(MLHandle CodecHandle,SoyPixelsMeta PixelFormat,int64_t BufferIndex)
{
	std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
	mOutputBuffers.PushBack( BufferIndex );
	mCodecHandle = CodecHandle;
	mPixelFormat = PixelFormat;
	Wake();
}

bool MagicLeap::TOutputThread::CanSleep()
{
	if ( mOutputBuffers.IsEmpty() )
		return true;
	
	return false;
}

void MagicLeap::TOutputThread::PopFrames(std::function<void(const SoyPixelsImpl&,SoyTime)>& OnFrameDecoded)
{
	if ( !mDecodedPixelsValid )
		return;
	
	//	pop any pixels we've got
	std::lock_guard<std::mutex> Lock(mDecodedPixelsLock);
	SoyTime DecodeDuration;
	OnFrameDecoded( mDecodedPixels, DecodeDuration );
	mDecodedPixelsValid = false;
}


void MagicLeap::TOutputThread::PopOutputBuffer(int64_t OutputBufferIndex)
{
	//	gr: hold onto pointer and don't release buffer until it's been read,to avoid a copy
	//		did this on android and it was a boost
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__,0);
	auto BufferHandle = static_cast<MLHandle>( OutputBufferIndex );
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
	IsOkay( Result, "MLMediaCodecGetOutputBufferPointer");
	
	auto ReleaseBuffer = [&]()
	{
		//	release back!
		bool Render = false;
		auto Result = MLMediaCodecReleaseOutputBuffer( mCodecHandle, BufferHandle, Render );
		IsOkay( Result, "MLMediaCodecReleaseOutputBuffer");
	};
	
	std::Debug << "Got OutputBuffer(" << OutputBufferIndex << ") DataSize=" << DataSize << " DataPtr=" << Data << std::endl;
	if ( Data == nullptr || DataSize == 0 )
	{
		std::Debug << "Got OutputBuffer(" << OutputBufferIndex << ") DataSize=" << DataSize << " DataPtr=" << Data << std::endl;
		ReleaseBuffer();
		return;
	}
	
	try
	{
		//	if data is null, then output is a surface
		
		//	output pixels!
		auto* DataMutable = const_cast<uint8_t*>( Data );
		SoyPixelsRemote NewPixels( DataMutable, DataSize, mPixelFormat );
		{
			std::lock_guard<std::mutex> Lock(mDecodedPixelsLock);
			mDecodedPixels.Copy( NewPixels );
			mDecodedPixelsValid = true;
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


bool MagicLeap::TOutputThread::Iteration()
{
	if ( mOutputBuffers.IsEmpty() )
		return true;

	//	read a buffer
	int64_t BufferIndex = -1;
	{
		std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
		BufferIndex = mOutputBuffers.PopAt(0);
	}
	try
	{
		PopOutputBuffer( BufferIndex );
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception getting output buffer " << BufferIndex << "; " << e.what() << std::endl;
		std::lock_guard<std::mutex> Lock(mOutputBuffersLock);
		auto& ElementZero = *mOutputBuffers.InsertBlock(0,1);
		ElementZero = BufferIndex;
		std::this_thread::sleep_for( std::chrono::seconds(4) );
	}
		
	return true;
}
