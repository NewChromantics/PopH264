#include "MagicLeapDecoder.h"

#include <ml_media_codec.h>
#include <ml_media_codeclist.h>
#include <ml_media_format.h>

namespace MagicLeap
{
	void	IsOkay(MLResult Result,const char* Context);
	void	IsOkay(MLResult Result,std::stringstream& Context);
	void	EnumCodecs(std::function<void(const std::string&)> Enum);
	
	const auto*	DefaultH264Codec = "OMX.Nvidia.h264.decode";
	//const auto*	DefaultH264Codec = "OMX.google.h264.decoder";
	
	//	got this mime from googling;
	//	http://hello-qd.blogspot.com/2013/05/choose-decoder-and-encoder-by-google.html
	const auto* H264MimeType = "video/avc";
	
	//	CSD-0 (from android mediacodec api)
	//	not in the ML api
	//	https://forum.magicleap.com/hc/en-us/community/posts/360048067552-Setting-csd-0-byte-buffer-using-MLMediaFormatSetKeyByteBuffer
	MLMediaFormatKey	MLMediaFormat_Key_CSD0 = "csd-0";
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
	
	//	gr: sometimes we get unknown so, put error nmber in
	auto* ResultString = MLGetResultString(Result);
	
	std::stringstream Error;
	Error << "Error in " << Context << ": " << ResultString << " (#" << static_cast<int>(Result) << ")";
	throw Soy::AssertException( Error );
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
		//This.OnInputBufferAvailible();
		std::Debug << "OnInputBufferAvailible(" << BufferIndex << ")" << std::endl;
	};
	
	auto OnOutputBufferAvailible = [](MLHandle Codec,int64_t BufferIndex,MLMediaCodecBufferInfo* BufferInfo,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnOutputBufferAvailible(" << BufferIndex << ")" << std::endl;
	};
	
	auto OnOutputFormatChanged = [](MLHandle Codec,MLHandle NewFormat,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
		std::Debug << "OnOutputFormatChanged" << std::endl;
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
	Result = MLMediaFormatCreateVideo( H264MimeType, 1024, 1024, &Format );
	IsOkay( Result, "MLMediaFormatCreateVideo" );
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
	
	MLHandle Crypto = ML_INVALID_HANDLE;
	//MLHandle Crypto = 0;	//	gr: INVALID_HANDLE doesnt work
	Result = MLMediaCodecConfigure( mHandle, Format, Crypto );
	IsOkay( Result, "MLMediaCodecConfigure" );
	
	Result = MLMediaCodecStart( mHandle );
	IsOkay( Result, "MLMediaCodecStart" );
	
	//	MLMediaCodecFlush makes all inputs invalid... flush on close?
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
	
	throw Soy::AssertException("todo");
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
