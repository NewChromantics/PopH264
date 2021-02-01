#include "MediaFoundationDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"
#include "SoyFourcc.h"
#include "SoyH264.h"

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


MediaFoundation::TDecoder::TDecoder(PopH264::TDecoderParams& Params,std::function<void(const SoyPixelsImpl&, size_t)> OnDecodedFrame) :
	PopH264::TDecoder	( OnDecodedFrame ),
	mVerboseDebug		( Params.mVerboseDebug )
{
	Soy::TFourcc InputFourccs[] = { "H264" };
	Soy::TFourcc OutputFourccs[] = { "NV12" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoDecoder, GetArrayBridge(Inputs), GetArrayBridge(Outputs), mVerboseDebug));
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
	//	try and pop even if we dont push data in, in case bail early
	PopFrames();

	Array<uint8_t> Nalu;
	PopH264::FrameNumber_t FrameNumber = 0;
	if (!PopNalu(GetArrayBridge(Nalu), FrameNumber))
		return false;

	//	gr: this will change to come from PopNalu to sync with meta
	SetInputFormat();

	auto NaluType = H264::GetPacketType(GetArrayBridge(Nalu));
	//std::Debug << "MediaFoundation got " << magic_enum::enum_name(NaluType) << " x" << Nalu.GetSize() << std::endl;

	bool PushData = true;

	//	skip some packets
	//	gr: we require SPS before PPS, before anything else otherwise nothing is output
	static bool SetSpsOnce = true;
	static bool SkipIfNotSpsSent = true;
	static bool SpsFirst = true;

	switch (NaluType)
	{
	case H264NaluContent::SequenceParameterSet:
		if (mSpsSet && SetSpsOnce)
			PushData = false;
		break;

	case H264NaluContent::PictureParameterSet:
		if (mPpsSet && SetSpsOnce)
			PushData = false;
		if (SpsFirst && !mSpsSet)
			PushData = false;
		break;

	case H264NaluContent::SupplimentalEnhancementInformation:
		if (mSeiSet && SetSpsOnce)
			PushData = false;
		if (SpsFirst && !mPpsSet)
			PushData = false;
		break;

	default:
		if (SkipIfNotSpsSent)
		{
			if (!mSpsSet || !mPpsSet /*|| !mSeiSet*/)
			{
				PushData = false;
			}
		}
		break;
	}

	if (NaluType == H264NaluContent::EndOfStream)
	{
		PushData = false;
		//	flush ditches pending inputs!
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH);

		//	"Requests a Media Foundation transform (MFT) to that streaming is about to end."
		//mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_STREAMING);

		//	gr: to flush, we just need _DRAIN
		//		with Picked Transform Microsoft H264 Video Decoder MFT

		//	notify there will be no more input
		mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_OF_STREAM);
		
		//	drain means we're going to drain all outputs until Needs_more_input
		mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_DRAIN);
		
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH_OUTPUT_STREAM);
	}

	if (PushData)
	{
		if (!mTransformer->PushFrame(GetArrayBridge(Nalu), FrameNumber))
		{
			//	data was rejected
			UnpopNalu(GetArrayBridge(Nalu), FrameNumber);
			//	gr: important, always show this
			//if (mVerboseDebug)
				std::Debug << __PRETTY_FUNCTION__ << " rejected " << NaluType << " unpopped" << std::endl;
		}
		else
		{
			if ( mVerboseDebug )
				std::Debug << __PRETTY_FUNCTION__ << " pushed " << NaluType << std::endl;

			//	mark some as pushed
			switch (NaluType)
			{
			case H264NaluContent::SequenceParameterSet:
				mSpsSet = true;
				break;

			case H264NaluContent::PictureParameterSet:
				mPpsSet = true;
				break;

			case H264NaluContent::SupplimentalEnhancementInformation:
				mSeiSet = true;
				break;

			default:break;
			}
		}
	}
	else
	{
		if (mVerboseDebug)
			std::Debug << __PRETTY_FUNCTION__ << " skipped " << NaluType << std::endl;
	}

	//	pop any frames that have come out in the mean time
	PopFrames();

	//	even if we didn't get a frame, try to decode again as we processed a packet
	return true;
}

//	return number of frames pushed (out)
//	maybe this can/should be on another thread
size_t MediaFoundation::TDecoder::PopFrames()
{
	size_t FramesPushed = 0;
	int LoopSafety = 10;
	bool PopAgain = true;
	while (PopAgain && --LoopSafety > 0)
	{
		Array<uint8_t> OutFrame;
		int64_t PacketNumber = -1;
		try
		{
			PopAgain = mTransformer->PopFrame(GetArrayBridge(OutFrame), PacketNumber);
		}
		catch (std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
			return FramesPushed;
		}
		
		//	no frame
		if (OutFrame.IsEmpty())
			return FramesPushed;
		
		auto PixelMeta = mTransformer->GetOutputPixelMeta();
		SoyPixelsRemote Pixels(OutFrame.GetArray(), OutFrame.GetDataSize(), PixelMeta);
		OnDecodedFrame(Pixels, PacketNumber);
		FramesPushed++;
	}
	return FramesPushed;
}