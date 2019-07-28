#include "MagicLeapDecoder.h"

#include <ml_media_codec.h>
#include <ml_media_codeclist.h>

namespace MagicLeap
{
	void	IsOkay(MLResult Result,const char* Context);
	void	EnumCodecs(std::function<void(const std::string&)> Enum);
}



void MagicLeap::IsOkay(MLResult Result,const char* Context)
{
	if ( Result == MLResult_Ok )
		return;
	
	auto* ResultString = MLGetResultString(Result);
	
	std::stringstream Error;
	Error << "Error in " << Context << ": " << ResultString;
	throw Soy::AssertException( Error );
}


void MagicLeap::EnumCodecs(std::function<void(const std::string&)> EnumCodec)
{
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

	auto* Mime = "h264";
	
	auto Result = MLMediaCodecCreateCodec( MLMediaCodecCreation_ByType, MLMediaCodecType_Decoder, Mime, &mHandle );
	IsOkay( Result, "MLMediaCodecCreateCodec" );
	
	auto OnInputBufferAvailible = [](MLHandle Codec,int64_t BufferIndex,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	
	auto OnOutputBufferAvailible = [](MLHandle Codec,int64_t BufferIndex,MLMediaCodecBufferInfo* BufferInfo,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	
	auto OnOutputFormatChanged = [](MLHandle Codec,MLHandle NewFormat,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	
	auto OnError = [](MLHandle Codec,int ErrorCode,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	
	auto OnFrameRendered = [](MLHandle Codec,int64_t PresentationTimeMicroSecs,int64_t SystemTimeNano,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	
	auto OnFrameAvailible = [](MLHandle Codec,void* pThis)
	{
		auto& This = *static_cast<MagicLeap::TDecoder*>(pThis);
	};
	

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
	MLHandle Crypto = ML_INVALID_HANDLE;
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
