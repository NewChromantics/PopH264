#include "MediaFoundationDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"

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

#include <SoyAutoReleasePtr.h>


namespace MediaFoundation
{
	class TActivateList;

	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result,const std::string& Context);

	TActivateList	EnumEncoders();
	TActivateList	EnumDecoders();
}


void MediaFoundation::IsOkay(HRESULT Result, const char* Context)
{
	Platform::IsOkay(Result, Context);
}

void MediaFoundation::IsOkay(HRESULT Result,const std::string& Context)
{
	Platform::IsOkay(Result, Context.c_str());
}

class MediaFoundation::TActivateList
{
public:
	~TActivateList();

public:
	IMFActivate**	mActivates = nullptr;
	uint32_t		mCount = 0;
};

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


MediaFoundation::TActivateList MediaFoundation::EnumEncoders()
{
	//	get all availible transform[er]s
	MFT_REGISTER_TYPE_INFO* InputFilter = nullptr;
	MFT_REGISTER_TYPE_INFO OutputFilter;
	OutputFilter.guidMajorType = MFMediaType_Video;
	OutputFilter.guidSubtype = MFVideoFormat_H264;

	uint32_t Flags = MFT_ENUM_FLAG_ALL;
	auto Category = MFT_CATEGORY_VIDEO_ENCODER;

	TActivateList Activates;
	auto Result = MFTEnumEx(Category, Flags, InputFilter, &OutputFilter, &Activates.mActivates, &Activates.mCount);
	IsOkay(Result, "MFTEnumEx");

	return Activates;
}


template<typename T>
void MemZero(T& Object)
{
	memset(&Object, 0, sizeof(T));
}


MediaFoundation::TActivateList MediaFoundation::EnumDecoders()
{
	//	get all availible transform[er]s
	MFT_REGISTER_TYPE_INFO InputFilter;
	MemZero(InputFilter);
	InputFilter.guidMajorType = MFMediaType_Video;
	InputFilter.guidSubtype = MFVideoFormat_H264;
	//InputFilter.guidSubtype = MFVideoFormat_H264_ES;

	MFT_REGISTER_TYPE_INFO* OutputFilter = nullptr;

	uint32_t Flags = MFT_ENUM_FLAG_ALL;
	auto Category = MFT_CATEGORY_VIDEO_ENCODER;

	TActivateList Activates;
	//auto Result = MFTEnumEx(Category, Flags, &InputFilter, OutputFilter, &Activates.mActivates, &Activates.mCount);
	auto Result = MFTEnum2(Category, Flags, &InputFilter, OutputFilter, nullptr, &Activates.mActivates, &Activates.mCount);
	IsOkay(Result, "MFTEnumEx");

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


MediaFoundation::TDecoder::TDecoder()
{
	auto Activates = EnumDecoders();
	
	if (Activates.mCount == 0)
		throw Soy::AssertException("No H264 decoders installed");

	//	activate a transformer
	{
		auto* Activate = Activates.mActivates[0];
		auto Result = Activate->ActivateObject(IID_PPV_ARGS(&mDecoder));
		IsOkay(Result, "Activate transform");
	}

	auto& Decoder = *mDecoder;

	//	grab attribs
	{
		IMFAttributes* Attributes = nullptr;
		auto Result = Decoder.GetAttributes(&Attributes);
		if (Attributes)
		{
			uint32_t HasAcceleration = 0;
			auto Result = Attributes->GetUINT32(CODECAPI_AVDecVideoAcceleration_H264, &HasAcceleration);
			if (Result == MF_E_ATTRIBUTENOTFOUND)
				std::Debug << "no CODECAPI_AVDecVideoAcceleration_H264 key" << std::endl;
			else
				std::Debug << "HasAcceleration = " << HasAcceleration << std::endl;
			Attributes->Release();
		}
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
	{
		auto Result = Decoder.AddInputStreams(1, &mStreamId);
		IsOkay(Result, "AddInputStream");
	}
}

MediaFoundation::TDecoder::~TDecoder()
{
	
}

bool MediaFoundation::TDecoder::DecodeNextPacket(std::function<void(const SoyPixelsImpl&, SoyTime)> OnFrameDecoded)
{
	Array<uint8_t> Nalu;
	if ( !PopNalu( GetArrayBridge(Nalu) ) )
		return false;

	if (!mDecoder)
		throw Soy::AssertException("Decoder is null");

	auto& Decoder = *mDecoder;

	auto DebugInputStatus = [&]()
	{
		DWORD StatusFlags = 0;
		auto Result = Decoder.GetInputStatus(mStreamId, &StatusFlags);
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
		auto Result = Decoder.GetOutputStatus(&StatusFlags);
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

	Soy::AutoReleasePtr<IMFSample> pSample;
	{
		auto Result = MFCreateSample(&pSample.mObject);
		IsOkay(Result, "MFCreateSample");
	}

	DebugInputStatus();

	DWORD Flags = 0;
	auto Result = Decoder.ProcessInput(mStreamId, pSample.mObject, Flags);
	IsOkay(Result, "Decoder.ProcessInput");

	DebugInputStatus();
	DebugOutputStatus();

	//	todo:
	//	do we check output here, or stick it on another thread
	return true;
}
