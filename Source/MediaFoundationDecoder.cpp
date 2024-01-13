#include "MediaFoundationDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "magic_enum.hpp"
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
	PopH264::TDecoder	( Params, OnDecodedFrame, OnFrameError )
{
	//	todo: let user specifiy a decoder/content type in params so we can allocate -and fail- a new tranformer immediately
}

MediaFoundation::TDecoder::~TDecoder()
{
	//	make sure we're not pushing frames etc whilst
	std::scoped_lock Lock(mTransformerLock);
	mTransformer.reset();
}

void MediaFoundation::TDecoder::CreateTransformer(ContentType::Type ContentType)
{
	std::scoped_lock Lock(mTransformerLock);
	if ( mTransformer )
		throw std::runtime_error("Transformer already allocated");

	std::array<Soy::TFourcc,1> Inputs;
	if ( ContentType == ContentType::Jpeg )
	{
		Inputs[0] = "MJPG";
	}
	else if ( ContentType == ContentType::Unknown )
	{
		//	assuming H264!
		Inputs[0] = "H264";
	}
	else
	{
		std::stringstream Error;
		Error << "Unhandled content type " << ContentType << " for CreateTransformer()";
		throw std::runtime_error(Error.str());
	}

	//	jpeg MFT decoder also has YUY2
	//Soy::TFourcc Outputs[] = { "YUY2" };
	Soy::TFourcc Outputs[] = { "NV12" };

	mTransformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoDecoder, std::span(Inputs), std::span(Outputs), mParams.mVerboseDebug ));

	try
	{
		mTransformer->SetLowLatencyMode(!mParams.mAllowBuffering);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set low-latency mode; " << e.what() << std::endl;
	}

	try
	{
		mTransformer->SetLowPowerMode(mParams.mLowPowerMode);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set low power mode; " << e.what() << std::endl;
	}

	try
	{
		mTransformer->SetDropBadFrameMode(mParams.mDropBadFrames);
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to set drop bad frames mode; " << e.what() << std::endl;
	}
}

void MediaFoundation::TDecoder::SetInputFormat(ContentType::Type ContentType)
{
	std::scoped_lock Lock(mTransformerLock);

	//	already setup
	if ( mTransformer )
		return;

	CreateTransformer(ContentType);

	//	gr: this doesn't mean "is the input format setup", it means "is the encoder ready for data"
	//if (mTransformer->IsReadyForInput())
	//	return;


	if ( ContentType == ContentType::Jpeg && false )
	{
		IMFMediaType* InputMediaType = nullptr;
		auto Result = MFCreateMediaType(&InputMediaType);
		IsOkay(Result, "MFCreateMediaType");
		Result = InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Image);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Image)");
		//auto InputFormatGuid = GUID_ContainerFormatJpeg;
		auto InputFormatGuid = MFImageFormat_JPEG;
		Result = InputMediaType->SetGUID(MF_MT_SUBTYPE, InputFormatGuid);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_SUBTYPE)");

		mTransformer->SetInputFormat(*InputMediaType);
	}
	else if ( ContentType == ContentType::Jpeg )
	{
		//	https://learn.microsoft.com/en-us/windows/win32/medfound/basic-mft-processing-model?redirectedfrom=MSDN#set-media-types
		auto Configure = [&](IMFMediaType& MediaType)
		{
			EnumAttributes(MediaType);
		};
		mTransformer->SetInputFormat('GPJM',Configure);
	}
	else if ( ContentType == ContentType::Jpeg )
	{
		//	I think we can reuse mjpeg for jpeg, when there's no jpeg decoders
		IMFMediaType* InputMediaType = nullptr;
		auto Result = MFCreateMediaType(&InputMediaType);
		IsOkay(Result, "MFCreateMediaType");
		Result = InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)");
		
		//auto InputFormatGuid = MFVideoFormat_MJPG;	//	'MJPG'
		Soy::TFourcc InputFormat("MJPG");
		auto InputFormatGuid = GetGuid(InputFormat);
		Result = InputMediaType->SetGUID(MF_MT_SUBTYPE, InputFormatGuid);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_SUBTYPE)");
		
		//	apparently a working version, which doesnt set an input
		//	https://stackoverflow.com/questions/9111362/media-foundation-mftransform-to-convert-mfsample-from-mjpg-to-yuy2-or-rgb24/9681384#9681384
		mTransformer->SetInputFormat(*InputMediaType);
	}
	else
	{
		Soy::TFourcc InputFormat("H264");
	
		IMFMediaType* InputMediaType = nullptr;
		auto Result = MFCreateMediaType(&InputMediaType);
		IsOkay(Result, "MFCreateMediaType");
		Result = InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		IsOkay(Result, "InputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)");
		//auto InputFormatGuid = MFVideoFormat_H264;
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
}

bool MediaFoundation::TDecoder::DecodeNextPacket()
{
	//	try and pop even if we dont push data in, in case bail early
	PopFrames();

	auto pNextPacket = PopNextPacket();
	if ( !pNextPacket )
		return false;
	auto& NextPacket = *pNextPacket;

	try
	{
		if ( NextPacket.mContentType == ContentType::Jpeg )
		{
			DecodeJpegPacket( *pNextPacket );
		}
		else
		{
			DecodeH264Packet( *pNextPacket );
		}
	}
	catch(std::exception& e)
	{
		UnpopPacket( pNextPacket );
		throw;
	}

	PopFrames();
	return true;
}



void MediaFoundation::TDecoder::DecodeJpegPacket(PopH264::TInputNaluPacket& NextPacket)
{
	SetInputFormat( NextPacket.mContentType );
}

void MediaFoundation::TDecoder::DecodeH264Packet(PopH264::TInputNaluPacket& NextPacket)
{
	H264NaluContent::Type NaluType = H264NaluContent::EndOfStream;

	if ( NextPacket.mContentType == ContentType::EndOfFile )
	{
		NaluType = H264NaluContent::EndOfStream;
	}
	else
	{
		SetInputFormat( NextPacket.mContentType );


		NaluType = H264::GetPacketType(NextPacket.GetData());
		//std::Debug << "MediaFoundation got " << magic_enum::enum_name(NaluType) << " x" << Nalu.GetSize() << std::endl;
	}

	std::scoped_lock Lock(mTransformerLock);
	if ( !mTransformer )
		throw std::runtime_error("No transformer");

	auto& Transformer = *mTransformer;


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
		Transformer.PushEndOfStream();
	}

	auto PushCount = (PushData ? (PushTwice ? 2 : 1) : 0);

	for ( auto pi=0;	pi<PushCount;	pi++ )
	{
		if (!Transformer.PushFrame( NextPacket.GetData(), NextPacket.mFrameNumber ) )
		{
			//	data was rejected
			//UnpopPacket(pNextPacket);
			//	gr: important, always show this
			//if (mVerboseDebug)
				std::Debug << __PRETTY_FUNCTION__ << " rejected " << NaluType << " unpopped" << std::endl;
			throw std::runtime_error("Packet rejected");
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
		//Transformer.ProcessCommand(MFT_MESSAGE_NOTIFY_END_OF_STREAM);
		Transformer.ProcessCommand(MFT_MESSAGE_COMMAND_DRAIN);
	}

}

//	return number of frames pushed (out)
//	maybe this can/should be on another thread
size_t MediaFoundation::TDecoder::PopFrames()
{
	std::scoped_lock Lock(mTransformerLock);
	//	may not have a transformer if data hasn't been input yet
	if ( !mTransformer )
		return 0;

	size_t FramesPushed = 0;
	int LoopSafety = 10;
	bool PopAgain = true;
	while (PopAgain && --LoopSafety > 0)
	{
		std::vector<uint8_t> OutFrame;
		int64_t PacketNumber = -1;
		json11::Json::object Meta;
		bool EndOfStream = false;
		try
		{
			PopAgain = mTransformer->PopFrame(OutFrame, PacketNumber, Meta, EndOfStream );
		}
		catch (std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
			return FramesPushed;
		}
		
		//	send frame
		if (!OutFrame.empty())
		{
			auto PixelMeta = mTransformer->GetOutputPixelMeta();
			SoyPixelsRemote Pixels(OutFrame.data(), OutFrame.size(), PixelMeta);
			OnDecodedFrame(Pixels, PacketNumber, Meta);
			FramesPushed++;
		}

		if (EndOfStream)
			OnDecodedEndOfStream();

		//	not expecting more frames this iteration
		if ( OutFrame.empty() )
			break;
	}
	return FramesPushed;
}
