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

#include "json11.hpp"

//	https://github.com/sipsorcery/mediafoundationsamples/blob/master/MFH264RoundTrip/MFH264RoundTrip.cpp

namespace MediaFoundation
{
	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result,const std::string& Context);
	GUID	GetGuid(Soy::TFourcc Fourcc);
}


MediaFoundation::TDecoder::TDecoder(PopH264::TDecoderParams& Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	PopH264::TDecoder	( OnDecodedFrame, OnFrameError ),
	mParams				( Params )
{
	Soy::TFourcc InputFourccs[] = { "H264" };
	Soy::TFourcc OutputFourccs[] = { "NV12" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoDecoder, GetArrayBridge(Inputs), GetArrayBridge(Outputs), mParams.mVerboseDebug ));

	try
	{
		mTransformer->SetLowLatencyMode(!Params.mAllowBuffering);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set low-latency mode; " << e.what() << std::endl;
	}

	try
	{
		mTransformer->SetLowPowerMode(Params.mLowPowerMode);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set low power mode; " << e.what() << std::endl;
	}

	try
	{
		mTransformer->SetDropBadFrameMode(Params.mDropBadFrames);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set drop bad frames mode; " << e.what() << std::endl;
	}
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

	//	gr: doesn't make much difference with frames
	auto FramesPerSecond = 30;
	Result = MFSetAttributeRatio(InputMediaType, MF_MT_FRAME_RATE, FramesPerSecond, 1);
	IsOkay(Result, "Set MF_MT_FRAME_RATE");

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
	bool PushTwice = false;
	bool DoDrain = false;

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

	case H264NaluContent::Slice_CodedIDRPicture:
		DoDrain = mParams.mDrainOnKeyframe;	//	try and force some frame output after a keyframe
		PushTwice = mParams.mDoubleDecodeKeyframe;	//	push keyframes twice to trick the decoder into decoding keyframe immediately
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
		//	gr: don't push as this is a special empty packet
		PushData = false;
		//	flush ditches pending inputs!
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH);

		//	"Requests a Media Foundation transform (MFT) to that streaming is about to end."
		//mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_STREAMING);

		//	gr: to flush, we just need _DRAIN
		//		with Picked Transform Microsoft H264 Video Decoder MFT

		//	notify there will be no more input
		mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_OF_STREAM);
		
		DoDrain = true;
		
		//mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_FLUSH_OUTPUT_STREAM);
	}

	auto PushCount = (PushData ? (PushTwice ? 2 : 1) : 0);

	for ( auto pi=0;	pi<PushCount;	pi++ )
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
			if ( mParams.mVerboseDebug )
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
	
	if ( PushCount == 0 )
	{
		if (mParams.mVerboseDebug)
			std::Debug << __PRETTY_FUNCTION__ << " skipped " << NaluType << std::endl;
	}

	if (DoDrain)
	{
		//	drain means we're going to drain all outputs until Needs_more_input
		//	gr: calling EOS makes output come immediately from a keyframe, but drain doesn't seem to do anything
		//mTransformer->ProcessCommand(MFT_MESSAGE_NOTIFY_END_OF_STREAM);
		mTransformer->ProcessCommand(MFT_MESSAGE_COMMAND_DRAIN);
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
		json11::Json::object Meta;
		try
		{
			PopAgain = mTransformer->PopFrame(GetArrayBridge(OutFrame), PacketNumber, Meta);
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
		OnDecodedFrame( Pixels, PacketNumber, Meta );
		FramesPushed++;
	}
	return FramesPushed;
}
