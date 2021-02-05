#include "MediaFoundationEncoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"
#include "SoyFourcc.h"
#include "PopH264.h"
#include "json11.hpp"

#include <mfapi.h>
#include <mftransform.h>
#include <codecapi.h>
#include <Mferror.h>

#include <SoyAutoReleasePtr.h>

//	https://github.com/sipsorcery/mediafoundationsamples/blob/master/MFH264RoundTrip/MFH264RoundTrip.cpp


MediaFoundation::TEncoderParams::TEncoderParams(json11::Json& Options)
{
	auto SetInt = [&](const char* Name, size_t& ValueUnsigned)
	{
		auto& Handle = Options[Name];
		if (!Handle.is_number())
			return false;
		auto Value = Handle.int_value();
		if (Value < 0)
		{
			std::stringstream Error;
			Error << "Value for " << Name << " is " << Value << ", not expecting negative";
			throw Soy::AssertException(Error);
		}
		ValueUnsigned = Value;
		return true;
	};
	auto SetBool = [&](const char* Name, bool& Value)
	{
		auto& Handle = Options[Name];
		if (!Handle.is_bool())
			return false;
		Value = Handle.bool_value();
		return true;
	};
	SetInt(POPH264_ENCODER_KEY_QUALITY, mQuality);
	SetInt(POPH264_ENCODER_KEY_PROFILELEVEL, mProfileLevel);
	SetInt(POPH264_ENCODER_KEY_AVERAGEKBPS, mAverageKbps);
	SetBool(POPH264_ENCODER_KEY_VERBOSEDEBUG, mVerboseDebug);
}




MediaFoundation::TEncoder::TEncoder(TEncoderParams Params,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket )
{
	Soy::TFourcc InputFourccs[] = { "NV12" };
	Soy::TFourcc OutputFourccs[] = { "H264" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoEncoder, GetArrayBridge(Inputs), GetArrayBridge(Outputs), Params.mVerboseDebug ));
}

MediaFoundation::TEncoder::~TEncoder()
{
	mTransformer.reset();
}


void MediaFoundation::TEncoder::SetOutputFormat(TEncoderParams Params,size_t Width,size_t Height)
{
	if (mTransformer->IsOutputFormatSet())
		return;

	auto& Transformer = *mTransformer->mTransformer;
	
	Soy::AutoReleasePtr<IMFMediaType> pMediaType;

	static bool UseNewFormat = true;
	if ( UseNewFormat )
	{
		auto Result = MFCreateMediaType(&pMediaType.mObject);
		IsOkay(Result, "MFCreateMediaType");
	}
	else //	start from existing output
	{
		//	gr: this doesn't always work, but probe it for defaults like kbps
		auto OutputFormatIndex = 0;
		auto Result = Transformer.GetOutputAvailableType(0, OutputFormatIndex, &pMediaType.mObject);
		IsOkay(Result, "GetOutputAvailableType");
	}
	auto* MediaType = pMediaType.mObject;


	{
		auto Result = MediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		IsOkay(Result, "MediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)");
		auto FormatGuid = MFVideoFormat_H264;
		Result = MediaType->SetGUID(MF_MT_SUBTYPE, FormatGuid);
		IsOkay(Result, "MediaType->SetGUID(MF_MT_SUBTYPE)");
	}

	//	setup required encoder things
	//	kbps required, must be >0
	if (Params.mAverageKbps == 0)
		throw Soy::AssertException("Encoder AverageKbps must be above zero");
	{
		auto BitRate = Params.mAverageKbps * 1024 * 8;
		auto Result = MediaType->SetUINT32(MF_MT_AVG_BITRATE, BitRate);
		IsOkay(Result, "Set encoder bitrate MF_MT_AVG_BITRATE");
	}

	//	interlace mode must be set
	{
		auto Result = MediaType->SetUINT32(MF_MT_INTERLACE_MODE, 2);
		IsOkay(Result, "MF_MT_INTERLACE_MODE");
	}

	{
		auto Numerator = 30000;
		auto Denominator = 1001;
		auto Result = MFSetAttributeRatio(MediaType, MF_MT_FRAME_RATE, Numerator, Denominator);
		IsOkay(Result, "MF_MT_FRAME_RATE");
	}

	{
		//auto Width = 640;
		//auto Height = 400;
		auto Result = MFSetAttributeSize(MediaType, MF_MT_FRAME_SIZE, Width, Height);
		IsOkay(Result, "MF_MT_FRAME_SIZE");
	}

	{
		auto Profile = eAVEncH264VProfile_Base;
		//Result = MFSetAttributeRatio(InputMediaType, MF_MT_MPEG2_PROFILE, Profile, 1);
		auto Result = MediaType->SetUINT32(MF_MT_MPEG2_PROFILE, Profile);
		IsOkay(Result, "Set encoder profile MF_MT_MPEG2_PROFILE");
	}

	//	gr: this results in "attribute not found" in SetOutputType if not set
	//		but docs say its optional
	//if (Params.mProfileLevel != 0)
	{
		double Level = Params.mProfileLevel / 10;	//	30 -> 3.1
		auto Result = MediaType->SetDouble(MF_MT_MPEG2_LEVEL, Level);
		IsOkay(Result, "Set encoder level MF_MT_MPEG2_LEVEL");
	}

	//	should we support zero?
	/*
	if (Params.mQuality != 0)
	{
		Result = MediaType->SetUINT32(CODECAPI_AVEncCommonQuality, Params.mQuality);
		IsOkay(Result, "Set encoder quality CODECAPI_AVEncCommonQuality");
	}
	*/
	mTransformer->SetOutputFormat(*MediaType);
}


void MediaFoundation::TEncoder::SetInputFormat(SoyPixelsMeta PixelsMeta)
{
	if (mTransformer->IsInputFormatSet())
		return;
	
	Soy::TFourcc InputFormat = GetFourcc(PixelsMeta.GetFormat());

	//	gr: check against supported formats, as error later could be vague
	{
		auto& SupportedFormats = mTransformer->mSupportedInputFormats;
		if (!SupportedFormats.Find(InputFormat))
		{
			std::stringstream Error;
			Error << "Input format " << PixelsMeta.GetFormat() << "/" << InputFormat << " not supported (";
			for (auto i=0;	i<SupportedFormats.GetSize();	i++)
			{
				Error << SupportedFormats[i] << ", ";
			}
			Error << ")";
			throw Soy::AssertException(Error);
		}
	}

	auto Configure = [&](IMFMediaType& MediaType)
	{
		EnumAttributes(MediaType);
				
		//	gr: this resolution must match the original output type, so no point setting it here
		//		should already be set using the AvailibleInput anyway
		if ( false)
		{
			//	https://support.microsoft.com/en-gb/help/2829223/video-resolution-limits-for-h-264-and-video-stabilization-in-windows-8
			auto Width = PixelsMeta.GetWidth();
			auto Height = PixelsMeta.GetHeight();
			auto Result = MFSetAttributeSize(&MediaType, MF_MT_FRAME_SIZE, Width, Height);
			IsOkay(Result, "Set MF_MT_FRAME_SIZE");
		}
		
	};
	mTransformer->SetInputFormat(InputFormat, Configure);
}

void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe)
{
	//	todo: need to convert this format to a supported format
	Soy_AssertTodo();
}


void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& _Pixels, const std::string& Meta, bool Keyframe)
{
	const SoyPixelsImpl* pEncodePixels = &_Pixels;

	//	work out if we need to change formats to one that's supported
	SoyPixels ConvertedPixels;
	auto EncodeFormat = GetInputFormat(_Pixels.GetFormat());
	if (EncodeFormat != _Pixels.GetFormat())
	{
		Soy::TScopeTimerPrint Timer("MediaFoundation::TEncoder::Encode re-encode", 2);
		ConvertedPixels.Copy(_Pixels);
		ConvertedPixels.SetFormat(EncodeFormat);
		pEncodePixels = &ConvertedPixels;
	}
	auto& EncodePixels = *pEncodePixels;

	//	encoder needs to set output type before input type
	//	and we need to know the resolution before we can set it
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/h-264-video-encoder
	SetOutputFormat(mParams, EncodePixels.GetWidth(), EncodePixels.GetHeight());
	
	SetInputFormat(EncodePixels.GetMeta());

	auto FrameNumber = PushFrameMeta(Meta);

	auto& PixelsArray = EncodePixels.GetPixelsArray();
	if (!mTransformer->PushFrame(GetArrayBridge(PixelsArray), FrameNumber))
	{
		std::Debug << "Input rejected... dropping frame?" << std::endl;
	}

	//	pop H264 frames that have been output
	//	gr: other thread for this, and get as many as possible in one go
	while (true)
	{
		auto More = FlushOutputFrame();
		if (!More)
			break;
	}
	
}

bool MediaFoundation::TEncoder::FlushOutputFrame()
{
	try
	{
		json11::Json::object Meta;
		PopH264::TPacket Packet;
		Packet.mData.reset(new Array<uint8_t>());
		int64_t FrameNumber = -1;
		mTransformer->PopFrame(GetArrayBridge(*Packet.mData), FrameNumber, Meta);

		//	no packet
		if (Packet.mData->IsEmpty())
			return false;

		try
		{
			Packet.mInputMeta = this->GetFrameMeta(FrameNumber);
		}
		catch (std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << "; " << e.what() << std::endl;
		}
		
		OnOutputPacket(Packet);
		return true;
	}
	catch (std::exception& e)
	{
		std::Debug << e.what() << std::endl;
		return false;
	}
}


SoyPixelsFormat::Type MediaFoundation::TEncoder::GetInputFormat(SoyPixelsFormat::Type Format)
{
	auto& Transformer = *mTransformer;

	//	is this fourcc supported?
	try
	{
		auto Fourcc = MediaFoundation::GetFourcc(Format);
		//	is it in the supported list?
		if (Transformer.mSupportedInputFormats.Find(Fourcc))
			return Format;
	}
	catch (std::exception& e)
	{
		//	not supported
	}

	//	convert to the first supported format (assuming first is best)
	auto& SupportedFormats = Transformer.mSupportedInputFormats;
	for (auto i = 0; i < SupportedFormats.GetSize(); i++)
	{
		try
		{
			auto NewFourcc = Transformer.mSupportedInputFormats[i];
			auto NewFormat = MediaFoundation::GetPixelFormat(NewFourcc);
			return NewFormat;
		}
		catch (std::exception& e)
		{
			//	fourcc not supported
		}
	}

	std::stringstream Error;
	Error << "No supported input fourcc's to convert " << Format << " to";
	throw Soy::AssertException(Error);
}

void MediaFoundation::TEncoder::FinishEncoding()
{

}




/*

In Windows 8, the H.264 decoder also supports the following attributes.

TABLE 4
Attribute	Description
CODECAPI_AVLowLatencyMode	Enables or disables low-latency decoding mode.
CODECAPI_AVDecNumWorkerThreads	Sets the number of worker threads used by the decoder.
CODECAPI_AVDecVideoMaxCodedWidth	Sets the maximum picture width that the decoder will accept as an input type.
CODECAPI_AVDecVideoMaxCodedHeight	Sets the maximum picture height that the decoder will accept as an input type.
MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT	Specifies the maximum number of output samples.
MFT_DECODER_EXPOSE_OUTPUT_TYPES_IN_NATIVE_ORDER	Specifies whether a decoder exposes IYUV/I420 output types (suitable for transcoding) before other formats.


	//	stream is always 0?
	auto StreamIndex = 0;

	IMFMediaType OutputFormatTypes[] =
	{
		MFVideoFormat_NV12,
		MFVideoFormat_YUY2,
		MFVideoFormat_YV12,
		MFVideoFormat_IYUV,
		MFVideoFormat_I420
	};

	//	try and set output type
	for (auto ot = 0; ot < std::size(OutputFormatTypes); ot++)
	{
		try
		{
			auto OutputFormat = OutputFormatTypes[ot];
			auto OutputFormatName = GetName(OutputFormat);
			auto Flags = 0;
			auto Result = Decoder->SetOutputType(StreamIndex, &OutputFormat, Flags);
			IsOkay(Result, std::string("SetOutputType ") + OutputFormatName);
			break;
		}
		catch (std::exception& e)
		{
			std::Debug << "Error setting output format " << e.what() << std::endl;
			continue;
		}
	}
	*/

