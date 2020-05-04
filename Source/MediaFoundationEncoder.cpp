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


std::string MediaFoundation::GetName(const GUID& Guid)
{
#define CASE_GUID(MatchGuid)	if ( Guid == MatchGuid )	return #MatchGuid
	CASE_GUID(MF_MT_MAJOR_TYPE);
	CASE_GUID(MF_MT_SUBTYPE);
	CASE_GUID(MF_MT_ALL_SAMPLES_INDEPENDENT);
	CASE_GUID(MF_MT_FIXED_SIZE_SAMPLES);
	CASE_GUID(MF_MT_COMPRESSED);
	CASE_GUID(MF_MT_SAMPLE_SIZE);
	CASE_GUID(MF_MT_WRAPPED_TYPE);
	CASE_GUID(MF_MT_VIDEO_3D);
	CASE_GUID(MF_MT_VIDEO_3D_FORMAT);
	CASE_GUID(MF_MT_VIDEO_3D_NUM_VIEWS);
	CASE_GUID(MF_MT_VIDEO_3D_LEFT_IS_BASE);
	CASE_GUID(MF_MT_VIDEO_3D_FIRST_IS_LEFT);
	CASE_GUID(MFSampleExtension_3DVideo);
	CASE_GUID(MF_MT_VIDEO_ROTATION);
	CASE_GUID(MF_DEVICESTREAM_MULTIPLEXED_MANAGER);
	CASE_GUID(MF_MEDIATYPE_MULTIPLEXED_MANAGER);
	CASE_GUID(MFSampleExtension_MULTIPLEXED_MANAGER);
	CASE_GUID(MF_MT_SECURE);
	CASE_GUID(MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES);
	CASE_GUID(MF_MT_ALPHA_MODE);
	CASE_GUID(MF_MT_DEPTH_MEASUREMENT);
	CASE_GUID(MF_MT_DEPTH_VALUE_UNIT);
	CASE_GUID(MF_MT_VIDEO_NO_FRAME_ORDERING);
	CASE_GUID(MF_MT_VIDEO_H264_NO_FMOASO);
	CASE_GUID(MF_MT_FORWARD_CUSTOM_NALU);
	CASE_GUID(MF_MT_FORWARD_CUSTOM_SEI);
	CASE_GUID(MF_MT_VIDEO_RENDERER_EXTENSION_PROFILE);
	CASE_GUID(MF_DECODER_FWD_CUSTOM_SEI_DECODE_ORDER);
	CASE_GUID(MF_MT_AUDIO_NUM_CHANNELS);
	CASE_GUID(MF_MT_AUDIO_SAMPLES_PER_SECOND);
	CASE_GUID(MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
	CASE_GUID(MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
	CASE_GUID(MF_MT_AUDIO_BLOCK_ALIGNMENT);
	CASE_GUID(MF_MT_AUDIO_BITS_PER_SAMPLE);
	CASE_GUID(MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
	CASE_GUID(MF_MT_AUDIO_SAMPLES_PER_BLOCK);
	CASE_GUID(MF_MT_AUDIO_CHANNEL_MASK);
	CASE_GUID(MF_MT_AUDIO_FOLDDOWN_MATRIX);
	CASE_GUID(MF_MT_AUDIO_WMADRC_PEAKREF);
	CASE_GUID(MF_MT_AUDIO_WMADRC_PEAKTARGET);
	CASE_GUID(MF_MT_AUDIO_WMADRC_AVGREF);
	CASE_GUID(MF_MT_AUDIO_WMADRC_AVGTARGET);
	CASE_GUID(MF_MT_AUDIO_PREFER_WAVEFORMATEX);
	CASE_GUID(MF_MT_AAC_PAYLOAD_TYPE);
	CASE_GUID(MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
	CASE_GUID(MF_MT_AUDIO_FLAC_MAX_BLOCK_SIZE);
	CASE_GUID(MF_MT_FRAME_SIZE);
	CASE_GUID(MF_MT_FRAME_RATE);
	CASE_GUID(MF_MT_PIXEL_ASPECT_RATIO);
	CASE_GUID(MF_MT_DRM_FLAGS);
	CASE_GUID(MF_MT_TIMESTAMP_CAN_BE_DTS);
	CASE_GUID(MF_MT_PAD_CONTROL_FLAGS);
	CASE_GUID(MF_MT_SOURCE_CONTENT_HINT);
	CASE_GUID(MF_MT_VIDEO_CHROMA_SITING);
	CASE_GUID(MF_MT_INTERLACE_MODE);
	CASE_GUID(MF_MT_TRANSFER_FUNCTION);
	CASE_GUID(MF_MT_VIDEO_PRIMARIES);
	CASE_GUID(MF_MT_MAX_LUMINANCE_LEVEL);
	CASE_GUID(MF_MT_MAX_FRAME_AVERAGE_LUMINANCE_LEVEL);
	CASE_GUID(MF_MT_MAX_MASTERING_LUMINANCE);
	CASE_GUID(MF_MT_MIN_MASTERING_LUMINANCE);
	CASE_GUID(MF_MT_DECODER_USE_MAX_RESOLUTION);
	CASE_GUID(MF_MT_DECODER_MAX_DPB_COUNT);
	CASE_GUID(MF_MT_CUSTOM_VIDEO_PRIMARIES);
	CASE_GUID(MF_MT_YUV_MATRIX);
	CASE_GUID(MF_MT_VIDEO_LIGHTING);
	CASE_GUID(MF_MT_VIDEO_NOMINAL_RANGE);
	CASE_GUID(MF_MT_GEOMETRIC_APERTURE);
	CASE_GUID(MF_MT_MINIMUM_DISPLAY_APERTURE);
	CASE_GUID(MF_MT_PAN_SCAN_APERTURE);
	CASE_GUID(MF_MT_PAN_SCAN_ENABLED);
	CASE_GUID(MF_MT_AVG_BITRATE);
	CASE_GUID(MF_MT_AVG_BIT_ERROR_RATE);
	CASE_GUID(MF_MT_MAX_KEYFRAME_SPACING);
	CASE_GUID(MF_MT_USER_DATA);
	CASE_GUID(MF_MT_OUTPUT_BUFFER_NUM);
	CASE_GUID(MF_MT_REALTIME_CONTENT);
	CASE_GUID(MF_MT_DEFAULT_STRIDE);
	CASE_GUID(MF_MT_PALETTE);
	CASE_GUID(MF_MT_AM_FORMAT_TYPE);
	CASE_GUID(MF_MT_VIDEO_PROFILE);
	CASE_GUID(MF_MT_VIDEO_LEVEL);
	CASE_GUID(MF_MT_MPEG_START_TIME_CODE);
	CASE_GUID(MF_MT_MPEG2_PROFILE);
	CASE_GUID(MF_MT_MPEG2_LEVEL);
	CASE_GUID(MF_MT_MPEG2_FLAGS);
	CASE_GUID(MF_MT_MPEG_SEQUENCE_HEADER);
	CASE_GUID(MF_MT_MPEG2_STANDARD);
	CASE_GUID(MF_MT_MPEG2_TIMECODE);
	CASE_GUID(MF_MT_MPEG2_CONTENT_PACKET);
	CASE_GUID(MF_MT_MPEG2_ONE_FRAME_PER_PACKET);
	CASE_GUID(MF_MT_MPEG2_HDCP);
	CASE_GUID(MF_MT_H264_MAX_CODEC_CONFIG_DELAY);
	CASE_GUID(MF_MT_H264_SUPPORTED_SLICE_MODES);
	CASE_GUID(MF_MT_H264_SUPPORTED_SYNC_FRAME_TYPES);
	CASE_GUID(MF_MT_H264_RESOLUTION_SCALING);
	CASE_GUID(MF_MT_H264_SIMULCAST_SUPPORT);
	CASE_GUID(MF_MT_H264_SUPPORTED_RATE_CONTROL_MODES);
	CASE_GUID(MF_MT_H264_MAX_MB_PER_SEC);
	CASE_GUID(MF_MT_H264_SUPPORTED_USAGES);
	CASE_GUID(MF_MT_H264_CAPABILITIES);
	CASE_GUID(MF_MT_H264_SVC_CAPABILITIES);
	CASE_GUID(MF_MT_H264_USAGE);
	CASE_GUID(MF_MT_H264_RATE_CONTROL_MODES);
	CASE_GUID(MF_MT_H264_LAYOUT_PER_STREAM);
	CASE_GUID(MF_MT_IN_BAND_PARAMETER_SET);
	CASE_GUID(MF_MT_MPEG4_TRACK_TYPE);
	CASE_GUID(MF_MT_DV_AAUX_SRC_PACK_0);
	CASE_GUID(MF_MT_DV_AAUX_CTRL_PACK_0);
	CASE_GUID(MF_MT_DV_AAUX_SRC_PACK_1);
	CASE_GUID(MF_MT_DV_AAUX_CTRL_PACK_1);
	CASE_GUID(MF_MT_DV_VAUX_SRC_PACK);
	CASE_GUID(MF_MT_DV_VAUX_CTRL_PACK);
	CASE_GUID(MF_MT_ARBITRARY_HEADER);
	CASE_GUID(MF_MT_ARBITRARY_FORMAT);
	CASE_GUID(MF_MT_IMAGE_LOSS_TOLERANT);
	CASE_GUID(MF_MT_MPEG4_SAMPLE_DESCRIPTION);
	CASE_GUID(MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);

	CASE_GUID(MFMediaType_Default);
	CASE_GUID(MFMediaType_Audio);
	CASE_GUID(MFMediaType_Video);
	CASE_GUID(MFMediaType_Protected);
	CASE_GUID(MFMediaType_SAMI);
	CASE_GUID(MFMediaType_Script);
	CASE_GUID(MFMediaType_Image);
	CASE_GUID(MFMediaType_HTML);
	CASE_GUID(MFMediaType_Binary);
	CASE_GUID(MFMediaType_FileTransfer);
	CASE_GUID(MFMediaType_Stream);
	CASE_GUID(MFMediaType_MultiplexedFrames);
	CASE_GUID(MFMediaType_Subtitle);
	
	CASE_GUID(MFT_ENCODER_SUPPORTS_CONFIG_EVENT);
	CASE_GUID(CODECAPI_AVDecVideoAcceleration_H264);

	WCHAR StringBuffer[100];
	StringFromGUID2( Guid, StringBuffer, std::size(StringBuffer) );
	auto String = Soy::WStringToString(StringBuffer);
	return String;
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
	PopH264::TEncoder	( OnOutputPacket )
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

void MediaFoundation::TEncoder::SetOutputFormat(TEncoderParams Params,size_t Width,size_t Height)
{
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
		auto Width = 640;
		auto Height = 400;
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


std::ostream& operator<<(std::ostream &out, const PROPVARIANT& in)
{
	VARENUM Type = static_cast<VARENUM>(in.vt);
	switch (Type)
	{
	case VT_BOOL:	out << in.boolVal;	break;
	case VT_I2:		out << in.iVal;	break;
	case VT_UI2:	out << in.uiVal;	break;
	case VT_I4:		out << in.intVal;	break;
	case VT_UI4:	out << in.uintVal;	break;
	case VT_I8:		out << in.hVal.QuadPart;	break;
	case VT_UI8:	out << in.uhVal.QuadPart;	break;

	case VT_CLSID:	
	{
		out << MediaFoundation::GetName(*in.puuid);
	}
	break;

	default:
	{
		out << "<Unhandled " << magic_enum::enum_name(Type) << ">";
	}
	break;
	}
	return out;
}

std::string MediaFoundation::GetValue(const PROPVARIANT& Variant, const GUID& Key)
{
	std::stringstream out;
	//	special cases

	//	pair of u32
	if (Key == MF_MT_FRAME_SIZE)
	{
		if (Variant.vt != VT_UI8)	throw Soy::AssertException("Expected 64bit type");
		auto* u64 = reinterpret_cast<const uint32_t*>(&Variant.uhVal);
		auto Low = u64[1];
		auto High = u64[0];
		out << Low << "," << High;
	}
	else
	{
		out << Variant;
	}
	return out.str();
}


void MediaFoundation::EnumAttributes(IMFAttributes& Attributes)
{
	uint32_t Count = 0;
	{
		auto Result = Attributes.GetCount(&Count);
		IsOkay(Result, "Failed to get attribute count");
	}

	for (auto i = 0; i < Count; i++)
	{
		try
		{ 
			GUID Key;
			PROPVARIANT Value;
			auto Result = Attributes.GetItemByIndex(i, &Key, &Value);
			IsOkay(Result, "GetAttributeItem x");
			VARENUM Type = static_cast<VARENUM>(Value.vt);
			std::Debug << "Attribute[" << i << "] " << GetName(Key) << "=" << Value << " (" << magic_enum::enum_name(Type) << ")" << std::endl;
		}
		catch (std::exception& e)
		{
			std::Debug << "Attribute[" << i << "] error; " << e.what() << std::endl;
		}
	}
}


void MediaFoundation::TEncoder::SetInputFormat(SoyPixelsMeta PixelsMeta)
{
	if (mTransformer->IsInputFormatReady())
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
	//	encoder needs to set output type before input type
	//	and we need to know the resolution before we can set it
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/h-264-video-encoder
	SetOutputFormat(mParams, Luma.GetWidth(), Luma.GetHeight());

	//	convert to the format required
	//	todo: allow override of base Encode so we dont split & rejoin
	//SetInputFormat(SoyPixelsFormat::Yuv_8_8_8_Full);
	SoyPixelsMeta PixelMeta(Luma.GetWidth(), Luma.GetHeight(), SoyPixelsFormat::Yuv_844_Full);
	SetInputFormat(PixelMeta);

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

