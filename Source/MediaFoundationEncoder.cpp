#include "MediaFoundationEncoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "SoyLib/src/magic_enum/include/magic_enum.hpp"
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
	PopH264::TEncoder	( OnOutputPacket ),
	mParams				( Params )
{
	//	todo: allow some params to create transformer immediately
}

MediaFoundation::TEncoder::~TEncoder()
{
	mTransformer.reset();
}


void MediaFoundation::TEncoder::SetOutputFormat(TTransformer& Transformer,SoyPixelsMeta ImageMeta)
{
	if ( !Transformer.mTransformer )
		throw std::runtime_error("Transformer(sub transformer) not initialised correctly");
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
		auto Result = Transformer.mTransformer->GetOutputAvailableType(0, OutputFormatIndex, &pMediaType.mObject);
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
	if (mParams.mAverageKbps == 0)
		throw Soy::AssertException("Encoder AverageKbps must be above zero");
	{
		auto BitRate = mParams.mAverageKbps * 1024 * 8;
		auto Result = MediaType->SetUINT32(MF_MT_AVG_BITRATE, BitRate);
		IsOkay(Result, "Set encoder bitrate MF_MT_AVG_BITRATE");
	}

	//	interlace mode must be set
	{
		auto Result = MediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
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
		auto Width = ImageMeta.GetWidth();
		auto Height = ImageMeta.GetHeight();
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
		double Level = mParams.mProfileLevel / 10;	//	30 -> 3.1
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
	Transformer.SetOutputFormat(*MediaType);
}

void MediaFoundation::TEncoder::SetFormat(SoyPixelsMeta ImageMeta)
{
	if ( mTransformer )
		return;
	//	encoder needs to set output type before input type
	//	and we need to know the resolution before we can set it
	//	https://docs.microsoft.com/en-us/windows/win32/medfound/h-264-video-encoder
	
	//	we also need to pick a transformer that supports our input format
	//	gr: here, we should also fallback to something that we can then do a slow path and do pixel conversion
	//		in c++
	Soy::TFourcc InputFormat = GetFourcc(ImageMeta.GetFormat());
	Soy::TFourcc Inputs[] = { InputFormat };
	Soy::TFourcc Outputs[] = { "H264" };

	std::shared_ptr<MediaFoundation::TTransformer> Transformer;
	Transformer.reset(new MediaFoundation::TTransformer(TransformerCategory::VideoEncoder, std::span(Inputs), std::span(Outputs), mParams.mVerboseDebug ));
		
	SetOutputFormat( *Transformer, ImageMeta );
	SetInputFormat( *Transformer, ImageMeta, InputFormat );

	mTransformer = Transformer;
}

void MediaFoundation::TEncoder::SetInputFormat(TTransformer& Transformer,SoyPixelsMeta PixelsMeta,Soy::TFourcc InputFormat)
{
	//	gr: check against supported formats, as error later could be vague
	{
		auto& SupportedFormats = Transformer.mActivate.mInputs;// mSupportedInputFormats;
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
	Transformer.SetInputFormat(InputFormat, Configure);
}

void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe)
{
	//	todo: need to convert this format to a supported format
	Soy_AssertTodo();
}


void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& Pixels, const std::string& Meta, bool Keyframe)
{
	SetFormat( Pixels.GetMeta() );

	auto FrameNumber = PushFrameMeta(Meta);

	//	can't push at the moment, save the frame
	if ( !FlushInputFrames() )
	{
		AddPendingFrame( Pixels, FrameNumber, Meta, Keyframe );
		FlushOutputFrames();
		return;
	}

	if ( !PushInputFrame( Pixels, FrameNumber, Meta, Keyframe ) )
	{
		AddPendingFrame( Pixels, FrameNumber, Meta, Keyframe );
		FlushOutputFrames();
		return;
	}

	FlushOutputFrames();
}

bool MediaFoundation::TEncoder::PushInputFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber, const std::string& Meta, bool Keyframe)
{
	auto& PixelsArray = Pixels.GetPixelsArray();
	auto* PixelsBytes = const_cast<uint8_t*>(PixelsArray.GetArray());
	std::span<uint8_t> PixelsSpan( PixelsBytes, PixelsArray.GetDataSize() );

	auto IsInputReady = mTransformer->IsInputFormatReady();
	if ( !mTransformer->PushFrame(PixelsSpan, FrameNumber) )
		return false;

	return true;
}

void MediaFoundation::TEncoder::FlushOutputFrames()
{
	//	pop H264 frames that have been output
	//	gr: other thread for this, and get as many as possible in one go
	while (true)
	{
		auto More = FlushOutputFrame();
		if (!More)
			break;
	}

}


void MediaFoundation::TEncoder::AddPendingFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber, const std::string& Meta, bool Keyframe)
{
	FrameImage_t Frame;
	Frame.mFrameNumber = FrameNumber;
	Frame.mKeyframe = Keyframe;
	Frame.mMeta = Meta;
	Frame.mPixels.reset( new SoyPixels(Pixels) );
	mPendingInputFrames.push_back(Frame);
	std::Debug << "Queued frame, now " << mPendingInputFrames.size() << " pending" << std::endl;
}


bool MediaFoundation::TEncoder::FlushInputFrames()
{
	while ( !mPendingInputFrames.empty() )
	{
		auto& Next = mPendingInputFrames[0];
		auto& Pixels = *Next.mPixels;
		if ( !PushInputFrame( Pixels, Next.mFrameNumber, Next.mMeta, Next.mKeyframe ) )
			return false;

		//	done, remove from pending!
		mPendingInputFrames.erase( mPendingInputFrames.begin() );
	}
	return true;
}

bool MediaFoundation::TEncoder::FlushOutputFrame()
{
	try
	{
		json11::Json::object Meta;
		PopH264::TPacket Packet;
		Packet.mData.reset(new std::vector<uint8_t>());
		int64_t FrameNumber = -1;
		bool EndOfStream = false;
		mTransformer->PopFrame( *Packet.mData, FrameNumber, Meta, EndOfStream);

		if (EndOfStream)
			OnFinished();

		//	no packet
		if ( Packet.mData->empty() )
		{
			if (EndOfStream)
				OnFinished();
			return false;
		}

		try
		{
			Packet.mInputMeta = this->GetFrameMeta(FrameNumber);
		}
		catch (std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << "; " << e.what() << std::endl;
		}
		
		OnOutputPacket(Packet);
		if (EndOfStream)
			OnFinished();
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
		auto& SupportedInputFormats = Transformer.mActivate.mInputs;
		if (SupportedInputFormats.Find(Fourcc))
			return Format;
	}
	catch (std::exception& e)
	{
		//	not supported
	}

	//	convert to the first supported format (assuming first is best)
	auto& SupportedFormats = Transformer.mActivate.mInputs;
	for (auto i = 0; i < SupportedFormats.GetSize(); i++)
	{
		try
		{
			auto NewFourcc = SupportedFormats[i];
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
	auto Transformer = mTransformer;
	if ( !Transformer )
		throw std::runtime_error("No transformer");

	Transformer->PushEndOfStream();

	while (true)
	{
		auto More = FlushOutputFrame();
		if (!More)
			break;
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

std::string MediaFoundation::TEncoder::GetEncoderName()
{
	auto Transformer = mTransformer;
	if ( !Transformer )
		return {};

	return Transformer->GetName();
}
