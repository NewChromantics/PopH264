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

namespace MediaFoundation
{
	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result,const std::string& Context);
	GUID	GetGuid(Soy::TFourcc Fourcc);
}



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
}




MediaFoundation::TEncoder::TEncoder(TEncoderParams Params,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket ),
	mParams				( Params )
{
	Soy::TFourcc InputFourccs[] = { "NV12" };
	Soy::TFourcc OutputFourccs[] = { "H264" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);
		
	mTransformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoEncoder, GetArrayBridge(Inputs), GetArrayBridge(Outputs)));
}

MediaFoundation::TEncoder::~TEncoder()
{
	mTransformer.reset();
}

Soy::TFourcc GetFourcc(SoyPixelsFormat::Type Format)
{
	//	https://www.fourcc.org/yuv.php
	switch (Format)
	{
	//case SoyPixelsFormat::Yvu_8_8_8_Full:	return Soy::TFourcc("IYUV"); //	same as I420
	case SoyPixelsFormat::Yuv_8_8_8_Full:
	case SoyPixelsFormat::Yuv_8_8_8_Ntsc:
		return Soy::TFourcc("YV12");

	case SoyPixelsFormat::Yuv_844_Full:
	case SoyPixelsFormat::Yuv_844_Ntsc:
		return Soy::TFourcc("NV12");

	case SoyPixelsFormat::Yvu_844_Ntsc:
		return Soy::TFourcc("NV21");	//	also 420O
	}

	std::stringstream Error;
	Error << "No encoding fourcc for pixel format " << magic_enum::enum_name(Format);
	throw Soy::AssertException(Error);
}

void MediaFoundation::TEncoder::SetInputFormat(SoyPixelsFormat::Type PixelFormat)
{
	if (mTransformer->IsInputFormatReady())
		return;
	
	Soy::TFourcc InputFormat = GetFourcc(PixelFormat);

	IMFMediaType* InputMediaType = nullptr;
	auto Result = MFCreateMediaType(&InputMediaType);
	IsOkay(Result, "MFCreateMediaType");

	Result = InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)");
	auto InputFormatGuid = GetGuid(InputFormat);
	Result = InputMediaType->SetGUID(MF_MT_SUBTYPE, InputFormatGuid);
	IsOkay(Result, "InputMediaType->SetGUID(MF_MT_SUBTYPE)");

	//	setup required encoder things
	if (mParams.mAverageKbps != 0)
	{
		auto BitRate = mParams.mAverageKbps * 1024 * 8;
		Result = InputMediaType->SetUINT32(MF_MT_AVG_BITRATE, BitRate);
		IsOkay(Result, "Set encoder bitrate MF_MT_AVG_BITRATE");
	}

	//CHECK_HR(pMFTOutputMediaType->SetUINT32(MF_MT_INTERLACE_MODE, 2), "Error setting interlace mode.");
	auto Profile = eAVEncH264VProfile_Base;
	Result = MFSetAttributeRatio(InputMediaType, MF_MT_MPEG2_PROFILE, Profile, 1);
	IsOkay(Result, "Set encoder profile MF_MT_MPEG2_PROFILE");

	if (mParams.mProfileLevel != 0)
	{
		double Level = mParams.mProfileLevel / 10;	//	30 -> 3.1
		Result = InputMediaType->SetDouble(MF_MT_MPEG2_LEVEL, Level);
		IsOkay(Result, "Set encoder level MF_MT_MPEG2_LEVEL");
	}

	//	should we support zero?
	if (mParams.mQuality != 0)
	{
		Result = InputMediaType->SetUINT32(CODECAPI_AVEncCommonQuality, mParams.mQuality);
		IsOkay(Result, "Set encoder quality CODECAPI_AVEncCommonQuality");
	}

	mTransformer->SetInputFormat(*InputMediaType);
}

void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe)
{
	//	convert to the format required
	//	todo: allow override of base Encode so we dont split & rejoin
	SetInputFormat(SoyPixelsFormat::Yuv_8_8_8_Full);


	//	pop H264 frames
	try
	{
		PopH264::TPacket Packet;
		Packet.mData.reset(new Array<uint8_t>());
		SoyTime Time;
		mTransformer->PopFrame(GetArrayBridge(*Packet.mData), Time);
		if (!Packet.mData->IsEmpty())
		{
			OnOutputPacket(Packet);
		}
	}
	catch (std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}
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

