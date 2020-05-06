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

#include <SoyAutoReleasePtr.h>

//	https://github.com/sipsorcery/mediafoundationsamples/blob/master/MFH264RoundTrip/MFH264RoundTrip.cpp

namespace MediaFoundation
{
	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result,const std::string& Context);
	GUID	GetGuid(Soy::TFourcc Fourcc);
}


MediaFoundation::TDecoder::TDecoder(std::function<void(const SoyPixelsImpl&, size_t)> OnDecodedFrame) :
	PopH264::TDecoder	( OnDecodedFrame )
{
	Soy::TFourcc InputFourccs[] = { "H264" };
	Soy::TFourcc OutputFourccs[] = { "NV12" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer( TransformerCategory::VideoDecoder, GetArrayBridge(Inputs), GetArrayBridge(Outputs)));
}

MediaFoundation::TDecoder::~TDecoder()
{
	mTransformer.reset();
}

void MediaFoundation::TDecoder::SetInputFormat()
{
	if (mTransformer->IsInputFormatReady())
		return;

	//	should get this from the initial meta
	Soy::TFourcc InputFormat("H264");

	IMFMediaType* InputMediaType = nullptr;
	auto Result = MFCreateMediaType(&InputMediaType);
	IsOkay(Result, "MFCreateMediaType");
	Result = InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)");
	auto InputFormatGuid = GetGuid(InputFormat);
	Result = InputMediaType->SetGUID(MF_MT_SUBTYPE, InputFormatGuid);
	IsOkay(Result, "InputMediaType->SetGUID(MF_MT_SUBTYPE)");
	mTransformer->SetInputFormat(*InputMediaType);
	//Result = Transformer.SetInputType(0, InputMediaType, 0);
	//IsOkay(Result, "SetInputType");
}

bool MediaFoundation::TDecoder::DecodeNextPacket()
{
	Array<uint8_t> Nalu;
	if (!PopNalu(GetArrayBridge(Nalu)))
		return false;

	SetInputFormat();

	auto NaluType = H264::GetPacketType(GetArrayBridge(Nalu));
	std::Debug << "MediaFoundation decode " << magic_enum::enum_name(NaluType) << " x" << Nalu.GetSize() << std::endl;

	if (NaluType == H264NaluContent::SequenceParameterSet)
	{
		static bool SpsSet = false;
		if (SpsSet)return true;
		SpsSet = true;
	}
	if (NaluType == H264NaluContent::PictureParameterSet)
	{
		static bool SpsSet = false;
		if (SpsSet)return true;
		SpsSet = true;
	}
	if (NaluType == H264NaluContent::SupplimentalEnhancementInformation)
	{
		static bool SpsSet = false;
		if (SpsSet)return true;
		SpsSet = true;
	}

	if (NaluType == H264NaluContent::EndOfStream)
	{
		//	flush ditches pending inputs!
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH);
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_DRAIN);
		mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_STREAMING);
		mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_OF_STREAM);
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH_OUTPUT_STREAM);
	}

	if (!mTransformer->PushFrame(GetArrayBridge(Nalu)))
	{
		//	data was rejected
		UnpopNalu(GetArrayBridge(Nalu));
	}

	//	try and pop frames
	//	todo: other thread
	{
		Array<uint8_t> OutFrame;
		SoyTime Time;
		mTransformer->PopFrame(GetArrayBridge(OutFrame), Time);

		//	no frame
		if (OutFrame.IsEmpty())
		{
			//	try and decode more nalu though
			return true;
		}

		auto PixelMeta = mTransformer->GetOutputPixelMeta();
		SoyPixelsRemote Pixels(OutFrame.GetArray(), OutFrame.GetDataSize(), PixelMeta);
		auto FrameNumber = Time.mTime;
		OnDecodedFrame(Pixels, FrameNumber);
	}

	return true;
}
