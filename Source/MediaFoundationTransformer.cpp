#include "MediaFoundationDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"
#include "SoyFourcc.h"

#include <mfapi.h>
#include <mftransform.h>
#include <codecapi.h>
#include <Mferror.h>

#pragma comment(lib,"dxva2.lib")	
#pragma comment(lib,"evr.lib")	
#pragma comment(lib,"mf.lib")	
#pragma comment(lib,"mfplat.lib")	
#pragma comment(lib,"mfplay.lib")	
#pragma comment(lib,"mfreadwrite.lib")	
#pragma comment(lib,"mfuuid.lib")	


//	https://github.com/sipsorcery/mediafoundationsamples/blob/master/MFH264RoundTrip/MFH264RoundTrip.cpp

namespace MediaFoundation
{
	class TActivateList;
	class TContext;
	class TActivateMeta;

	std::ostream& operator<<(std::ostream &out, const TActivateMeta& in);
	
	TActivateList	EnumTransforms(const GUID& Category);
	TActivateMeta	GetBestTransform(const GUID& Category, const ArrayBridge<Soy::TFourcc>& InputFilter, const ArrayBridge<Soy::TFourcc>& OutputFilter);
	
	Soy::AutoReleasePtr<IMFSample>		CreateSample(const ArrayBridge<uint8_t>& Data, Soy::TFourcc Fourcc);
	Soy::AutoReleasePtr<IMFMediaBuffer>	CreateBuffer(const ArrayBridge<uint8_t>& Data);
	Soy::AutoReleasePtr<IMFMediaBuffer>	CreateBuffer(DWORD Size, DWORD Alignment);
	Soy::AutoReleasePtr<IMFSample>		CreateSample(DWORD Size, DWORD Alignment);
	void								ReadData(IMFSample& Sample,ArrayBridge<uint8_t>& Data);
}



class MediaFoundation::TContext
{
public:
	TContext();
	~TContext();
};

class MediaFoundation::TActivateMeta
{
public:
	TActivateMeta() {}
	TActivateMeta(IMFActivate& Activate);

public:
	std::string						mName;
	bool							mHardwareAccelerated = false;
	BufferArray<Soy::TFourcc, 20>	mInputs;
	BufferArray<Soy::TFourcc, 20>	mOutputs;
	Soy::AutoReleasePtr<IMFActivate>	mActivate;
};


class MediaFoundation::TActivateList
{
public:
	TActivateList() {}
	TActivateList(TActivateList&& Move)
	{
		this->mActivates = Move.mActivates;
		this->mCount = Move.mCount;
		Move.mActivates = nullptr;
	}
	~TActivateList();

	void			EnumActivates(std::function<void(TActivateMeta&)> EnumMeta);

public:
	IMFActivate**	mActivates = nullptr;
	uint32_t		mCount = 0;
};


void MediaFoundation::IsOkay(HRESULT Result, const char* Context)
{
	Platform::IsOkay(Result, Context);
}

void MediaFoundation::IsOkay(HRESULT Result,const std::string& Context)
{
	Platform::IsOkay(Result, Context.c_str());
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
	StringFromGUID2(Guid, StringBuffer, std::size(StringBuffer));
	auto String = Soy::WStringToString(StringBuffer);
	return String;
}


Soy::TFourcc MediaFoundation::GetFourcc(SoyPixelsFormat::Type Format)
{
	//	https://www.fourcc.org/yuv.php
	switch (Format)
	{
		//case SoyPixelsFormat::Yvu_8_8_8_Full:	return Soy::TFourcc("IYUV"); //	same as I420
	case SoyPixelsFormat::Yuv_8_8_8_Full:
	case SoyPixelsFormat::Yuv_8_8_8_Ntsc:
		return Soy::TFourcc("YV12");

	case SoyPixelsFormat::Yuv_8_88_Full:
	case SoyPixelsFormat::Yuv_8_88_Ntsc:
		return Soy::TFourcc("NV12");

	case SoyPixelsFormat::Yvu_8_88_Ntsc:
		return Soy::TFourcc("NV21");	//	also 420O
	}

	std::stringstream Error;
	Error << "No encoding fourcc for pixel format " << magic_enum::enum_name(Format);
	throw Soy::AssertException(Error);
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


MediaFoundation::TActivateList::~TActivateList()
{
	if (mActivates)
	{
		for (auto i = 0; i < mCount; i++)
		{
			auto* Activate = mActivates[i];
			Activate->Release();
		}
	}
	if (mActivates)
	{
		CoTaskMemFree(mActivates);
		mActivates = nullptr;
	}
}

void MediaFoundation::TActivateList::EnumActivates(std::function<void(TActivateMeta&)> EnumMeta)
{
	for (auto a = 0; a < mCount; a++)
	{
		auto* Activate = mActivates[a];
		TActivateMeta Meta(*Activate);
		EnumMeta(Meta);
	}
}



template<typename T>
void MemZero(T& Object)
{
	memset(&Object, 0, sizeof(T));
}

std::ostream& MediaFoundation::operator<<(std::ostream &out, const MediaFoundation::TActivateMeta& in)
{
	out << "Name=" << in.mName;
	if (in.mHardwareAccelerated)
		out << "(Hardware)";

	out << " Inputs=";
	for (auto i = 0; i < in.mInputs.GetSize(); i++)
		out << in.mInputs[i] << ",";

	out << " Outputs=";
	for (auto i = 0; i < in.mOutputs.GetSize(); i++)
		out << in.mOutputs[i] << ",";

	return out;
}


MediaFoundation::TContext::TContext()
{
	CoInitializeEx(nullptr,COINIT_MULTITHREADED);

	auto Result = MFStartup(MF_VERSION, MFSTARTUP_FULL);
	IsOkay(Result, "MFStartup");
}

MediaFoundation::TContext::~TContext()
{
	auto Result = MFShutdown();
	try
	{
		IsOkay(Result, "MFShutdown");
	}
	catch (std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}
}

template<typename TYPE>
void GetBlobs(IMFActivate& Activate, const GUID& Key, ArrayBridge<TYPE>&& Array)
{
	//	get blob size
	uint8_t* BlobData8 = nullptr;
	uint32_t BlobDataSize = 0;
	auto Result = Activate.GetAllocatedBlob(Key, &BlobData8, &BlobDataSize);

	auto Cleanup = [&]()
	{
		if (!BlobData8)
			return;
		CoTaskMemFree(BlobData8);
		BlobData8 = nullptr;
	};
	try
	{
		MediaFoundation::IsOkay(Result, "GetAllocatedBlob");
		auto* BlobData = reinterpret_cast<TYPE*>(BlobData8);
		auto BlobDataCount = BlobDataSize / sizeof(TYPE);

		auto BlobArray = GetRemoteArray(BlobData, BlobDataCount);
		Array.Copy(BlobArray);
		Cleanup();
	}
	catch (...)
	{
		Cleanup();
		throw;
	}
}

std::string GetString(IMFActivate& Activate, const GUID& Key)
{
	//	get length not including terminator
	uint32_t Length = 0;
	wchar_t StringWBuffer[1024] = { 0 };
	auto Result = Activate.GetString(Key, StringWBuffer, std::size(StringWBuffer), &Length);
	MediaFoundation::IsOkay(Result, "GetString");
	std::wstring StringW(StringWBuffer, Length);

	auto String = Soy::WStringToString(StringW);
	return String;
}

std::string GetStringSafe(IMFActivate& Activate, const GUID& Key)
{
	try
	{
		return GetString(Activate, Key);
	}
	catch (std::exception& e)
	{
		//std::Debug << e.what() << std::endl;
		return std::string();
	}
}



Soy::TFourcc MediaFoundation::GetFourCC(const GUID& Guid)
{
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/video-subtype-guids#creating-subtype-guids-from-fourccs-and-d3dformat-values
	//	XXXXXXXX - 0000 - 0010 - 8000 - 00AA00389B71
	Soy::TFourcc Fourcc(Guid.Data1);
	return Fourcc;
}

constexpr uint32_t MediaFoundation::GetFourcc(const char Str[])
{
	//	gr: figure out how to make this class const-expr-happy
	//Soy::TFourcc Fourcc(Str);
	//return Fourcc.mFourcc32;
	uint8_t a = Str[0];
	uint8_t b = Str[1];
	uint8_t c = Str[2];
	uint8_t d = Str[3];
	uint32_t abcd = (a << 0) | (b << 8) | (c << 16) | (d << 24);
	return abcd;
}

SoyPixelsFormat::Type MediaFoundation::GetPixelFormat(const GUID& Guid)
{
	auto Fourcc = GetFourCC(Guid);
	return GetPixelFormat(Fourcc);
}


SoyPixelsFormat::Type MediaFoundation::GetPixelFormat(Soy::TFourcc Fourcc)
{
	switch (Fourcc.mFourcc32)
	{
	case MediaFoundation::GetFourcc("NV12"):	return SoyPixelsFormat::Nv12;
	//case MediaFoundation::GetFourcc("IYUV"):	return SoyPixelsFormat::Yvu_8_8_8_Full;
	//case MediaFoundation::GetFourcc("I420"):	return SoyPixelsFormat::Yvu_8_8_8_Full;
	}
	

	std::stringstream Error;
	Error << "Failed to get pixelformat from fourcc " << Fourcc;
	throw Soy::AssertException(Error);
}

GUID MediaFoundation::GetGuid(Soy::TFourcc Fourcc)
{
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/video-subtype-guids#creating-subtype-guids-from-fourccs-and-d3dformat-values
	//	XXXXXXXX - 0000 - 0010 - 8000 - 00 AA 00 38 9B 71
	//	see DEFINE_MEDIATYPE_GUID
	//auto Fourcc32 = _byteswap_ulong(Fourcc.mFourcc32);
	auto Fourcc32 = (Fourcc.mFourcc32);
	GUID Guid = { Fourcc32, 0x000, 0x0010, { 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71 } };
	return Guid;
}

GUID MediaFoundation::GetGuid(TransformerCategory::Type Category)
{
	switch (Category)
	{
	case TransformerCategory::VideoDecoder:	return MFT_CATEGORY_VIDEO_DECODER;
	case TransformerCategory::VideoEncoder:	return MFT_CATEGORY_VIDEO_ENCODER;
	}
	throw Soy::AssertException("Unhandled Transformer category");
}

MediaFoundation::TActivateMeta::TActivateMeta(IMFActivate& Activate) :
	mActivate	( &Activate, true )
{
	mName = GetStringSafe(Activate, MFT_FRIENDLY_NAME_Attribute);
	auto HardwareUrl = GetStringSafe(Activate, MFT_ENUM_HARDWARE_URL_Attribute);
	mHardwareAccelerated = !HardwareUrl.empty();

	Array<MFT_REGISTER_TYPE_INFO> InputTypes;
	Array<MFT_REGISTER_TYPE_INFO> OutputTypes;
	GetBlobs(Activate, MFT_INPUT_TYPES_Attributes, GetArrayBridge(InputTypes));
	GetBlobs(Activate, MFT_OUTPUT_TYPES_Attributes, GetArrayBridge(OutputTypes));
	
	for (auto i = 0; i < InputTypes.GetSize(); i++)
	{
		auto Fourcc = GetFourCC(InputTypes[i].guidSubtype);
		mInputs.PushBack(Fourcc);
	}

	for (auto i = 0; i < OutputTypes.GetSize(); i++)
	{
		auto Fourcc = GetFourCC(OutputTypes[i].guidSubtype);
		mOutputs.PushBack(Fourcc);
	}
}

//	MFT_CATEGORY_VIDEO_ENCODER	MFT_CATEGORY_VIDEO_DECODER
MediaFoundation::TActivateList MediaFoundation::EnumTransforms(const GUID& Category)
{
	//	auto init context once
	static TContext Context;

	//	gr: these filters always seem to return zero results, so get all and filter after
	//MFT_REGISTER_TYPE_INFO InputFilter = { 0 };
	//InputFilter.guidMajorType = MFMediaType_Video;
	//InputFilter.guidSubtype = MFVideoFormat_H264;
	//InputFilter.guidSubtype = MFVideoFormat_H264_ES;
	MFT_REGISTER_TYPE_INFO* InputFilter = nullptr;
	MFT_REGISTER_TYPE_INFO* OutputFilter = nullptr;

	uint32_t Flags = MFT_ENUM_FLAG_ALL;
	
	TActivateList Activates;
	IMFAttributes* Attributes = nullptr;
	auto Result = MFTEnum2(Category, Flags, InputFilter, OutputFilter, Attributes, &Activates.mActivates, &Activates.mCount);
	IsOkay(Result, "MFTEnum2");

	return Activates;
}

std::string GetName(const GUID Guid)
{
	if (Guid == MFVideoFormat_NV12)	return "MFVideoFormat_NV12";
	if (Guid == MFVideoFormat_YUY2)	return "MFVideoFormat_YUY2";
	if (Guid == MFVideoFormat_YV12)	return "MFVideoFormat_YV12";
	if (Guid == MFVideoFormat_IYUV)	return "MFVideoFormat_IYUV";
	if (Guid == MFVideoFormat_I420)	return "MFVideoFormat_I420";

	return "<guid>";
}

//	get the best activate which matches the input list, and output list
//	sorted results by hardware, then highest input, then highest output
MediaFoundation::TActivateMeta MediaFoundation::GetBestTransform(const GUID& Category,const ArrayBridge<Soy::TFourcc>& InputFilter,const ArrayBridge<Soy::TFourcc>& OutputFilter)
{
	auto Transformers = EnumTransforms(Category);
	TActivateMeta MatchingTransform;
	int MatchingTransformScore = 0;
	const auto HardwareScore = -1000;
	const auto InputScore = 10;
	const auto OutputScore = 1;

	auto AddMatch = [&](TActivateMeta Meta, Soy::TFourcc Input, Soy::TFourcc Output, int Score)
	{
		//	is this better than the current match?
		if (Score <= MatchingTransformScore)
			return;

		MatchingTransform = Meta;
		MatchingTransformScore = Score;
	};

	//	get the indexes of MatchList that match
	auto GetMatchingIndexes = [](BufferArray<Soy::TFourcc, 20>& List,const ArrayBridge<Soy::TFourcc>& MatchList)
	{
		BufferArray<int, 20> MatchingInputIndexes;
		for (auto mi = 0; mi < MatchList.GetSize(); mi++)
		{
			auto Match = MatchList[mi];
			auto ListIndex = List.FindIndex(Match);
			if (ListIndex == -1)
				continue;
			MatchingInputIndexes.PushBack(mi);
		}
		return MatchingInputIndexes;
	};

	auto GetLowestMatchingIndex = [&](BufferArray<Soy::TFourcc, 20>& List,const ArrayBridge<Soy::TFourcc>& MatchList)
	{
		auto MatchingIndexes = GetMatchingIndexes( List, MatchList );
		if (MatchingIndexes.IsEmpty())
			return -1;
		auto Lowest = MatchingIndexes[0];
		return Lowest;
	};


	auto FindTransform = [&](TActivateMeta& Meta)
	{
		std::Debug << "Transformer: " << Meta << std::endl;

		auto BestInputIndex = GetLowestMatchingIndex(Meta.mInputs, InputFilter);
		auto BestOutputIndex = GetLowestMatchingIndex(Meta.mOutputs, OutputFilter);
		//	not a match
		if (BestInputIndex==-1 || BestOutputIndex==-1)
			return;
		//	Catch programming error
		if (BestInputIndex >= InputFilter.GetSize())
			throw Soy::AssertException("Internal Error: Input index out of bounds");
		if (BestOutputIndex >= OutputFilter.GetSize())
			throw Soy::AssertException("Internal Error: Output index out of bounds");

		//	calc a score
		auto Score = 0;
		if (Meta.mHardwareAccelerated)
			Score += HardwareScore;
		Score += (InputFilter.GetSize() - BestInputIndex) * InputScore;
		Score += (OutputFilter.GetSize() - BestOutputIndex) * OutputScore;
		AddMatch(Meta, InputFilter[BestInputIndex], OutputFilter[BestOutputIndex], Score);
	};

	Transformers.EnumActivates(FindTransform);

	if (MatchingTransformScore == 0)
	{
		throw Soy::AssertException("No transformers matching the input/output filters");
	}

	return MatchingTransform;
}



MediaFoundation::TTransformer::TTransformer(TransformerCategory::Type Category, const ArrayBridge<Soy::TFourcc>&& InputFormats, const ArrayBridge<Soy::TFourcc>&& OutputFormats)
{
	auto CategoryGuid = GetGuid(Category);

	//	todo: support user-selected names
	auto Transform = GetBestTransform(CategoryGuid, InputFormats, OutputFormats);
	std::Debug << "Picked Transform " << Transform.mName << std::endl;

	mSupportedInputFormats = Transform.mInputs;

	//	activate a transformer
	{
		auto* Activate = Transform.mActivate.mObject;
		auto Result = Activate->ActivateObject(IID_PPV_ARGS(&mTransformer));
		IsOkay(Result, "Activate transform");
	}

	auto& Transformer = *mTransformer;

	//	debug attribs
	{
		Soy::AutoReleasePtr<IMFAttributes> Attributes;
		auto Result = Transformer.GetAttributes(&Attributes.mObject);
		if (Attributes)
			MediaFoundation::EnumAttributes(*Attributes);
	}
}

bool MediaFoundation::TTransformer::IsInputFormatReady()
{
	return mInputFormatSet;
	auto& Transformer = *this->mTransformer;
	{
		DWORD StatusFlags = 0;
		auto Result = Transformer.GetInputStatus(mInputStreamId, &StatusFlags);
		IsOkay(Result, "GetInputStatus");
		std::Debug << "Input status is " << StatusFlags << std::endl;

		auto CanAcceptData = (StatusFlags & MFT_INPUT_STATUS_ACCEPT_DATA) != 0;
		if (!CanAcceptData)
		{
			std::stringstream Error;
			Error << "Not ready for input data";
			throw Soy::AssertException(Error);
		}
	}

	return true;
}

void MediaFoundation::TTransformer::LockTransformer(std::function<void()> Execute)
{
	auto& Transformer = *this->mTransformer;

	//	for async transforms we get an error if not locked
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/asynchronous-mfts
	//	this func locks, executes the lambda, then unlocks
	
	//	todo; cache this state
	Soy::AutoReleasePtr<IMFAttributes> Attributes;
	auto IsAsync = false;
	{
		auto Result = Transformer.GetAttributes(&Attributes.mObject);
		IsOkay(Result, "GetAttributes for async check");
		uint32_t IsAsync32 = 0;
		Result = Attributes->GetUINT32(MF_TRANSFORM_ASYNC, &IsAsync32);
		if (Result != MF_E_ATTRIBUTENOTFOUND)
		{
			IsOkay(Result, "Get MF_TRANSFORM_ASYNC attribute");
			IsAsync = IsAsync32 != 0;
		}
	}

	if (IsAsync)
	{
		//	unlock
		auto Result = Attributes->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, true);
		IsOkay(Result, "Set MF_TRANSFORM_ASYNC_UNLOCK attribute");
	}

	Execute();

	//	relock?
}

void MediaFoundation::TTransformer::SetOutputFormat(IMFMediaType& MediaType)
{
	auto& Transformer = *this->mTransformer;
	
	auto Set = [&]()
	{
		auto Result = Transformer.SetOutputType(mOutputStreamId, &MediaType, 0);
		if (Result == MF_E_INVALIDTYPE)
			throw Soy::AssertException("Invalid");
		if (Result == MF_E_TRANSFORM_TYPE_NOT_SET)
			throw Soy::AssertException("Input type must be set before output");
		IsOkay(Result, "SetOutputType");
	};
	LockTransformer(Set);
}


void MediaFoundation::TTransformer::SetInputFormat(Soy::TFourcc Fourcc, std::function<void(IMFMediaType&)> ConfigMedia)
{
	//	nvidia(+others?) encoder seems to need to use an input
	//	media type provided, find the one matching our format and then 
	//	callback for modifications before setting
	auto& Transformer = *mTransformer;

	Soy::AutoReleasePtr<IMFMediaType> pMatchedMediaType;
	for (auto i = 0; i < 1000; i++)
	{
		Soy::AutoReleasePtr<IMFMediaType> MediaType;
		auto Result = Transformer.GetInputAvailableType(mInputStreamId, i, &MediaType.mObject);
		if (Result == MF_E_NO_MORE_TYPES)
			break;

		//	input hasn't been set
		if (Result == MF_E_TRANSFORM_TYPE_NOT_SET)
			throw Soy::AssertException("Cannot get Input types until output is set");
		IsOkay(Result, "GetInputAvailableType");

		GUID SubType;
		Result = MediaType->GetGUID(MF_MT_SUBTYPE, &SubType);
		IsOkay(Result, "OutputFormat GetGuid Subtype");
		//	todo: is it a format we support?
		auto MatchFourcc = GetFourCC(SubType);
		if (MatchFourcc != Fourcc)
			continue;

		//	match!
		pMatchedMediaType = MediaType;
		break;
	}

	if (!pMatchedMediaType)
		throw Soy::AssertException("Didn't find a matching input type");

	//	let user modify before submitting
	if (ConfigMedia)
		ConfigMedia(*pMatchedMediaType);

	//	now set it
	SetInputFormat(*pMatchedMediaType);
}

void MediaFoundation::TTransformer::ProcessCommand(MFT_MESSAGE_TYPE Command)
{
	auto& Transformer = *this->mTransformer;
	ULONG_PTR Param = 0;// nullptr;
	auto Result = Transformer.ProcessMessage(Command, Param);
	IsOkay(Result, std::string("ProcessMessage ") + std::string(magic_enum::enum_name(Command)));
}


void MediaFoundation::TTransformer::SetInputFormat(IMFMediaType& MediaType)
{
	auto& Transformer = *this->mTransformer;
	/* gr: not implemented on all transforms
	{
		DWORD InputStreamIds[10];
		DWORD OutputStreamIds[10];
		auto Result = Decoder.GetStreamIDs(std::size(InputStreamIds), InputStreamIds, std::size(OutputStreamIds), OutputStreamIds);
		IsOkay(Result, "GetStreamIDs");
	}
	*/
	auto Set = [&]()
	{
		auto Result = Transformer.SetInputType(0, &MediaType, 0);
		IsOkay(Result, "SetInputType");
	};
	LockTransformer(Set);

	auto ProcessCommand = [&](MFT_MESSAGE_TYPE Message)
	{
		ULONG_PTR Param = 0;// nullptr;
		auto Result = Transformer.ProcessMessage(Message, Param);
		IsOkay(Result, std::string("ProcessMessage ") + std::string(magic_enum::enum_name(Message)));
	};

	//	gr: are these needed?
	ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH);
	ProcessCommand(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING);
	ProcessCommand(MFT_MESSAGE_NOTIFY_START_OF_STREAM);

	auto InputReady = IsInputFormatReady();
	mInputFormatSet = true;
}

MediaFoundation::TTransformer::~TTransformer()
{
	
}


Soy::AutoReleasePtr<IMFMediaBuffer> MediaFoundation::CreateBuffer(const ArrayBridge<uint8_t>& Data)
{
	Soy::AutoReleasePtr<IMFMediaBuffer> pBuffer;
	auto Result = MFCreateMemoryBuffer(Data.GetDataSize(), &pBuffer.mObject);
	IsOkay(Result, "MFCreateMemoryBuffer");
	pBuffer.Retain();

	//	copy data
	uint8_t* DestData = nullptr;
	DWORD DestMaxSize = 0;
	DWORD DestCurrentSize = 0;
	Result = pBuffer->Lock(&DestData, &DestMaxSize, &DestCurrentSize);
	IsOkay(Result, "Buffer Lock");

	size_t NewSize = 0;
	auto DestArray = GetRemoteArray(DestData, DestMaxSize, NewSize);
	DestArray.Copy(Data);
	
	Result = pBuffer->Unlock();
	IsOkay(Result, "Buffer Unlock");
	Result = pBuffer->SetCurrentLength(NewSize);
	IsOkay(Result, "Buffer SetCurrentLength");

	return pBuffer;
}

Soy::AutoReleasePtr<IMFMediaBuffer> MediaFoundation::CreateBuffer(DWORD Size, DWORD Alignment)
{
	Soy::AutoReleasePtr<IMFMediaBuffer> pBuffer;

	//	gr: just always align?
	if (Alignment > 0)
	{
		auto Result = MFCreateAlignedMemoryBuffer(Size, Alignment, &pBuffer.mObject);
		IsOkay(Result, "MFCreateAlignedMemoryBuffer");
	}
	else
	{
		auto Result = MFCreateMemoryBuffer(Size, &pBuffer.mObject);
		IsOkay(Result, "MFCreateMemoryBuffer");
	}

	return pBuffer;
}


Soy::AutoReleasePtr<IMFSample> MediaFoundation::CreateSample(const ArrayBridge<uint8_t>& Data,Soy::TFourcc Fourcc)
{
//#pragma message("This binary will not load on windows 7")
	auto Buffer = CreateBuffer(Data);

	Soy::AutoReleasePtr<IMFSample> pSample;
	auto Result = MFCreateSample(&pSample.mObject);
	IsOkay(Result, "MFCreateSample");
	pSample.Retain();

	Result = pSample.mObject->AddBuffer(Buffer.mObject);
	IsOkay(Result, "AddBuffer");
	//CHECK_HR(reConstructedVideoSample->SetSampleTime(llVideoTimeStamp), "Error setting the recon video sample time.\n");
	//CHECK_HR(reConstructedVideoSample->SetSampleDuration(llSampleDuration), "Error setting recon video sample duration.\n");

	return pSample;
}


Soy::AutoReleasePtr<IMFSample> MediaFoundation::CreateSample(DWORD Size, DWORD Alignment)
{
	auto Buffer = CreateBuffer(Size, Alignment);

	Soy::AutoReleasePtr<IMFSample> pSample;
	auto Result = MFCreateSample(&pSample.mObject);
	IsOkay(Result, "MFCreateSample");
	pSample.Retain();

	Result = pSample.mObject->AddBuffer(Buffer.mObject);
	IsOkay(Result, "AddBuffer");

	return pSample;
}

void MediaFoundation::ReadData(IMFSample& Sample, ArrayBridge<uint8_t>& Data)
{
	//	ConvertToContiguousBuffer automatically retains
	Soy::AutoReleasePtr<IMFMediaBuffer> pBuffer;
	auto Result = Sample.ConvertToContiguousBuffer(&pBuffer.mObject);
	IsOkay(Result, "ConvertToContiguousBuffer");
	if ( !pBuffer )
		throw Soy::AssertException("Missing Media buffer object");

	auto& Buffer = *pBuffer;

	//	lock
	uint8_t* SrcData = nullptr;
	DWORD SrcSize = 0;
	
	//	note: lock is garunteed to be contiguous
	Result = Buffer.Lock(&SrcData, nullptr, &SrcSize);
	IsOkay(Result, "MediaBuffer::Lock");

	auto LockedArray = GetRemoteArray(SrcData, SrcSize);
	Data.Copy(LockedArray);

	Buffer.Unlock();
}



bool MediaFoundation::TTransformer::PushFrame(const ArrayBridge<uint8_t>&& Data)
{
	if (!mTransformer)
		throw Soy::AssertException("Decoder is null");

	auto& Transformer = *mTransformer;

	auto DebugInputStatus = [&]()
	{
		DWORD StatusFlags = 0;
		auto Result = Transformer.GetInputStatus(mInputStreamId, &StatusFlags);
		try
		{
			IsOkay(Result, "GetInputStatus");
			std::Debug << "Input status is " << StatusFlags << std::endl;
		}
		catch (std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	};

	auto DebugOutputStatus = [&]()
	{
		DWORD StatusFlags = 0;
		auto Result = Transformer.GetOutputStatus(&StatusFlags);
		try
		{
			IsOkay(Result, "GetOutputStatus");
			std::Debug << "Output status is " << StatusFlags << std::endl;
		}
		catch (std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	};

	auto pSample = CreateSample(Data,Soy::TFourcc());

	DebugInputStatus();

	DWORD Flags = 0;
	auto Result = Transformer.ProcessInput(mInputStreamId, pSample.mObject, Flags);
	
	//	gr: getting this result, even though status says "is accepting"
	if (Result == MF_E_NOTACCEPTING)
		return false;

	IsOkay(Result, "Decoder.ProcessInput");

	DebugInputStatus();
	return true;
}

void MediaFoundation::TTransformer::SetOutputFormat()
{
	auto& Transformer = *mTransformer;

	//	gr: a manual one never seems to work
	if ( false )
	{
		IMFMediaType* MediaType = nullptr;
		auto Result = MFCreateMediaType(&MediaType);
		IsOkay(Result, "MFCreateMediaType");
		Result = MediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Image)");
		//auto FormatGuid = GetGuid(Transform.mOutputs[0]);
		auto FormatGuid = MFVideoFormat_NV12;
		Result = MediaType->SetGUID(MF_MT_SUBTYPE, FormatGuid);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_SUBTYPE)");
		/*
		Result = MFSetAttributeRatio( MediaType, MF_MT_FRAME_RATE, 30, 1);
		IsOkay(Result, "MFSetAttributeRatio");
		Result = MFSetAttributeSize(MediaType,MF_MT_FRAME_SIZE, 640, 480);
		IsOkay(Result, "MFSetAttributeSize");
		*/
		//Result = Decoder.SetOutputType(0, MediaType, 0);
		//IsOkay(Result, "SetOutputType");
		Result = Transformer.SetOutputType(0, nullptr, 0);
		IsOkay(Result, "SetOutputType");
	}

	//	enum availiable formats and try each
	for ( auto i=0;	i<1000;	i++ )
	{
		Soy::AutoReleasePtr<IMFMediaType> pMediaType;
		auto Result = Transformer.GetOutputAvailableType(mOutputStreamId, i, &pMediaType.mObject);
		
		if (Result == MF_E_NO_MORE_TYPES)
			break;

		//	input hasn't been set
		if (Result == MF_E_TRANSFORM_TYPE_NOT_SET)
			IsOkay(Result, "GetOutputAvailableType");
		IsOkay(Result, "GetOutputAvailableType");
		
		GUID SubType;
		auto& MediaType = *pMediaType;
		Result = MediaType.GetGUID(MF_MT_SUBTYPE, &SubType);
		IsOkay(Result, "OutputFormat GetGuid Subtype");
		//	todo: is it a format we support?
		auto Fourcc = GetFourCC(SubType);
				
		//	set format
		DWORD Flags = 0;
		Result = Transformer.SetOutputType(mOutputStreamId, &MediaType, Flags );
		IsOkay(Result, "SetOutputType");

		//	gr: this media info (specifically width& height)
		//		is not correct. GetCurrentOutput is also wrong,
		//		the correct one comes after the streams changed notification 
		//		and with the GetAvailibleFormat
		//mOutputMediaType = pMediaType;
		mOutputMediaType.Release();	//	make sure its not set

		return;
	}

	throw Soy::AssertException("Ran out of availible output formats");
}

SoyPixelsMeta MediaFoundation::TTransformer::GetOutputPixelMeta()
{
	auto& MediaType = GetOutputMediaType();

	uint32_t Width = 0;
	uint32_t Height = 0;
	auto Result = MFGetAttributeSize(&MediaType, MF_MT_FRAME_SIZE, &Width, &Height);
	IsOkay(Result, "GetOutputPixelMeta MFGetAttributeSize");
	
	uint32_t Stride = 0;
	Result = MediaType.GetUINT32( MF_MT_DEFAULT_STRIDE, &Stride );
	IsOkay(Result, "GetOutputPixelMeta MFGetAttributeSize");

	//	get format
	GUID VideoFormatGuid;
	Result = MediaType.GetGUID(MF_MT_SUBTYPE, &VideoFormatGuid);
	IsOkay(Result, "GetOutputPixelMeta MF_MT_SUBTYPE");
	auto PixelFormat = GetPixelFormat(VideoFormatGuid);

	return SoyPixelsMeta(Width, Height, PixelFormat);
}


IMFMediaType& MediaFoundation::TTransformer::GetOutputMediaType()
{
	if (mOutputMediaType)
		return *mOutputMediaType;

	//	get latest media type
	//	gr: OutputCurrent is WRONG, but availible IS (after we get the streams changed result)
	//auto Result = mTransformer->GetOutputCurrentType(mOutputStreamId, &mOutputMediaType.mObject);
	auto Result = mTransformer->GetOutputAvailableType(mOutputStreamId,0, &mOutputMediaType.mObject);
	IsOkay(Result, "GetOutputAvailableType");
	//mOutputMediaType.Retain();
	
	if (!mOutputMediaType)
		throw Soy::AssertException("GetOutputMediaType but output media type has not yet been set");
	
	return *mOutputMediaType.mObject;
}

void MediaFoundation::TTransformer::PopFrame(ArrayBridge<uint8_t>&& Data,SoyTime& Time)
{
	if (!mTransformer)
		throw Soy::AssertException("Transformer is null");

	auto& Transformer = *mTransformer;
	DWORD StatusFlags = 0;
	{
		auto Result = Transformer.GetOutputStatus(&StatusFlags);
		if (Result == MF_E_TRANSFORM_TYPE_NOT_SET)
		{
			SetOutputFormat();
			Result = Transformer.GetOutputStatus(&StatusFlags);
		}
		IsOkay(Result, "GetOutputStatus");
	}

	//	gr: this also says false, when ProcessOutput tries to copy a sample!
	auto FrameReady = (StatusFlags & MFT_OUTPUT_STATUS_SAMPLE_READY) != 0;
	if (!FrameReady)
	{
		//return;
	}

	MFT_OUTPUT_STREAM_INFO OutputInfo;
	auto Result = Transformer.GetOutputStreamInfo(mOutputStreamId, &OutputInfo);
	IsOkay(Result, "GetOutputStreamInfo");

	bool PreAllocatedSamples = (OutputInfo.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
	if (PreAllocatedSamples)
	{
		std::Debug << "Todo: process pre-allocated samples" << std::endl;
	}

	//	get output
	auto pSample = CreateSample(OutputInfo.cbSize,OutputInfo.cbAlignment);
	MFT_OUTPUT_DATA_BUFFER output_buffer = { mOutputStreamId, pSample.mObject, 0, nullptr };
	DWORD Flags = 0;
	DWORD Status = 0;
	Result = Transformer.ProcessOutput( Flags, 1, &output_buffer, &Status );

	//	handle some special retu
	if (Result == MF_E_TRANSFORM_NEED_MORE_INPUT)
	{
		return;
	}
	else if (Result == MF_E_TRANSFORM_STREAM_CHANGE)
	{
		std::Debug << "Stream changed, expecting 0 byte buffer..." << std::endl;
		//	reset output media type
		mOutputMediaType.Release();

		//	refresh format now
		auto& Format = GetOutputMediaType();
		//	only for pixel streams!
		auto PixelMeta = GetOutputPixelMeta();
		std::Debug << "Streams changed: " << PixelMeta << std::endl;
	}
	else
	{
		IsOkay(Result, "ProcessOutput");
	}
	
	//	read a frame!
	ReadData(*pSample.mObject, Data);
	std::Debug << "size is " << Data.GetDataSize() << std::endl;

	try
	{
		LONGLONG TimestampNs = 0;
		auto Result = pSample->GetSampleTime(&TimestampNs);
		IsOkay(Result, "GetSampleTime");
		Time.mTime = TimestampNs / 10000;
	}
	catch (std::exception& e)
	{
		std::Debug <<  e.what() << std::endl;
	}

}
