#include "X264Encoder.h"
#include "SoyPixels.h"


namespace X264
{
	void	IsOkay(int Result, const char* Context);
	void	Log(void *data, int i_level, const char *psz, va_list args);
	
	int		GetColourSpace(SoyPixelsFormat::Type Format);
	
	class TPacket;
}


void X264::IsOkay(int Result, const char* Context)
{
	if (Result == 0)
		return;
	
	std::stringstream Error;
	Error << "X264 error " << Result << " (" << Context << ")";
	throw Soy::AssertException(Error);
}


X264::TEncoder::TEncoder(size_t PresetValue,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket )
{
	if ( PresetValue > 9 )
		throw Soy_AssertException("Expecting preset value <= 9");
	
	//	trigger dll load
#if defined(TARGET_WINDOWS)
	Soy::TRuntimeLibrary Dll("x264.dll");
#endif
	
	//	todo: tune options. takes , seperated values
	const char* Tune = nullptr;
	auto* PresetName = x264_preset_names[PresetValue];
	auto Result = x264_param_default_preset(&mParam, PresetName, Tune);
	IsOkay(Result,"x264_param_default_preset");
}

X264::TEncoder::~TEncoder()
{
	
}


std::string X264::TEncoder::GetVersion()
{
	std::stringstream Version;
	Version << "x264 " << X264_POINTVER;
	return Version.str();
}

void X264::Log(void *data, int i_level, const char *psz, va_list args)
{
	std::stringstream Debug;
	Debug << "x264 ";
	
	switch (i_level)
	{
		case X264_LOG_ERROR:	Debug << "Error: ";	break;
		case X264_LOG_WARNING:	Debug << "Warning: ";	break;
		case X264_LOG_INFO:		Debug << "Info: ";	break;
		case X264_LOG_DEBUG:	Debug << "Debug: ";	break;
		default:				Debug << "???: ";	break;
	}
	
	auto temp = std::vector<char>{};
	auto length = std::size_t{ 63 };
	while (temp.size() <= length)
	{
		temp.resize(length + 1);
		//va_start(args, psz);
		const auto status = std::vsnprintf(temp.data(), temp.size(), psz, args);
		//va_end(args);
		if (status < 0)
			throw std::runtime_error{ "string formatting error" };
		length = static_cast<std::size_t>(status);
	}
	auto FormattedString = std::string{ temp.data(), length };
	Debug << FormattedString;
	//msg_GenericVa(p_enc, i_level, MODULE_STRING, psz, args);
	std::Debug << Debug.str() << std::endl;
}

int X264::GetColourSpace(SoyPixelsFormat::Type Format)
{
	switch (Format)
	{
		case SoyPixelsFormat::Yuv_8_8_8_Ntsc:	return X264_CSP_I420;
			//case SoyPixelsFormat::Greyscale:		return X264_CSP_I400;
			//case SoyPixelsFormat::Yuv_8_88_Ntsc:	return X264_CSP_NV12;
			//case SoyPixelsFormat::Yvu_8_88_Ntsc:	return X264_CSP_NV21;
			//case SoyPixelsFormat::Yvu_844_Ntsc:		return X264_CSP_NV16;
			//	these are not supported by x264
			//case SoyPixelsFormat::BGR:		return X264_CSP_BGR;
			//case SoyPixelsFormat::BGRA:		return X264_CSP_BGRA;
			//case SoyPixelsFormat::RGB:		return X264_CSP_BGR;
		default:break;
	}
	
	std::stringstream Error;
	Error << "X264::GetColourSpace unsupported format " << Format;
	throw Soy::AssertException(Error);
}

void X264::TEncoder::AllocEncoder(const SoyPixelsMeta& Meta)
{
	//	todo: change PPS if content changes
	if (mPixelMeta.IsValid())
	{
		if (mPixelMeta == Meta)
			return;
		std::stringstream Error;
		Error << "H264 encoder pixel format changing from " << mPixelMeta << " to " << Meta << ", currently unsupported";
		throw Soy_AssertException(Error);
	}
	
	//	do final configuration & alloc encoder
	mParam.i_csp = GetColourSpace(Meta.GetFormat());
	mParam.i_width = size_cast<int>(Meta.GetWidth());
	mParam.i_height = size_cast<int>(Meta.GetHeight());
	mParam.b_vfr_input = 0;
	mParam.b_repeat_headers = 1;
	mParam.b_annexb = 1;
	mParam.p_log_private = reinterpret_cast<void*>(&X264::Log);
	mParam.i_log_level = X264_LOG_DEBUG;
	
	//	reduce mem usage by reducing threads
	mParam.i_threads = 1;
	mParam.i_lookahead_threads = 0;
	mParam.b_sliced_threads = false;
	
	//	h264 profile level
	mParam.i_level_idc = 30;//	3.0
	
	auto Profile = "baseline";
	auto Result = x264_param_apply_profile(&mParam, Profile);
	IsOkay(Result, "x264_param_apply_profile");
	
	Result = x264_picture_alloc(&mPicture, mParam.i_csp, mParam.i_width, mParam.i_height);
	IsOkay(Result, "x264_picture_alloc");
	
	mHandle = x264_encoder_open(&mParam);
	if (!mHandle)
		throw Soy::AssertException("Failed to open x264 encoder");
	mPixelMeta = Meta;
}

void X264::TEncoder::Encode(const SoyPixelsImpl& Pixels,const std::string& Meta)
{
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__, 2);
	AllocEncoder(Pixels.GetMeta());
	
	//	need planes
	auto& YuvPixels = Pixels;
	//YuvPixels.SetFormat(SoyPixelsFormat::Yuv_8_8_8_Ntsc);
	BufferArray<std::shared_ptr<SoyPixelsImpl>, 3> Planes;
	YuvPixels.SplitPlanes(GetArrayBridge(Planes));
	
	//auto& LumaPlane = *Planes[0];
	//auto& ChromaUPlane = *Planes[1];
	//auto& ChromaVPlane = *Planes[2];
	
	//	checks from example code https://github.com/jesselegg/x264/blob/master/example.c
	//	gr: look for proper validation funcs
	auto Width = Pixels.GetWidth();
	auto Height = Pixels.GetHeight();
	int LumaSize = Width * Height;
	int ChromaSize = LumaSize / 4;
	int ExpectedBufferSizes[] = { LumaSize, ChromaSize, ChromaSize };
	
	for (auto i = 0; i < Planes.GetSize(); i++)
	{
		auto* OutPlane = mPicture.img.plane[i];
		auto& InPlane = *Planes[i];
		auto& InPlaneArray = InPlane.GetPixelsArray();
		auto OutSize = ExpectedBufferSizes[i];
		auto InSize = InPlaneArray.GetDataSize();
		if (OutSize != InSize)
		{
			std::stringstream Error;
			Error << "Copying plane " << i << " for x264, but plane size mismatch " << InSize << " != " << OutSize;
			throw Soy_AssertException(Error);
		}
		memcpy(OutPlane, InPlaneArray.GetArray(), InSize );
	}
	
	mPicture.i_pts = PushFrameMeta(Meta);
	
	Encode(&mPicture);
	
	//	flush any other frames
	//	gr: this is supposed to only be called at the end of the stream...
	//		if DelayedFrameCount non zero, we may haveto call multiple times before nal size is >0
	//		so just keep calling until we get 0
	//	maybe add a safety iteration check
	//	gr: need this on OSX (latest x264) but on windows (old build) every subsequent frame fails
	//	gr: this was backwards? brew (old 2917) DID need to flush?
	if (X264_REV < 2969)
	{
		//	gr: flushing on OSX (X264_REV 2917) causing
		//	log: x264 [error]: lookahead thread is already stopped
#if !defined(TARGET_OSX)
		{
			//FlushFrames();
		}
#endif
	}
}



void X264::TEncoder::FinishEncoding()
{
	//	when we're done with frames, we need to make the encoder flush out any more packets
	int Safety = 1000;
	while (--Safety > 0)
	{
		auto DelayedFrameCount = x264_encoder_delayed_frames(mHandle);
		if (DelayedFrameCount == 0)
			break;
		
		Encode(nullptr);
	}
}


void X264::TEncoder::Encode(x264_picture_t* InputPicture)
{
	//	we're assuming here mPicture has been setup, or we're flushing
	
	//	gr: currently, decoder NEEDS to have nal packets split
	auto OnNalPacket = [&](FixedRemoteArray<uint8_t>& Data)
	{
		auto FrameNumber = mPicture.i_pts;
		auto FrameMeta = PopFrameMeta(FrameNumber);
		
		//	todo: either store these to make sure decode order (dts) is kept correct
		//		or send DTS order to TPacket for host to order
		PopH264::TPacket OutputPacket;
		OutputPacket.mData.reset(new Array<uint8_t>());
		OutputPacket.mInputMeta = FrameMeta;
		OutputPacket.mData->PushBackArray(Data);
		OnOutputPacket(OutputPacket);
	};
	
	x264_picture_t OutputPicture;
	x264_nal_t* Nals = nullptr;
	int NalCount = 0;
	
	auto FrameSize = x264_encoder_encode(mHandle, &Nals, &NalCount, InputPicture, &OutputPicture);
	if (FrameSize < 0)
		throw Soy::AssertException("x264_encoder_encode error");
	
	//	processed, but no data output
	if (FrameSize == 0)
	{
		auto DelayedFrameCount = x264_encoder_delayed_frames(mHandle);
		std::Debug << "x264::Encode processed, but no output; DelayedFrameCount=" << DelayedFrameCount << std::endl;
		return;
	}
	
	//	process each nal
	auto TotalNalSize = 0;
	for (auto n = 0; n < NalCount; n++)
	{
		auto& Nal = Nals[n];
		auto NalSize = Nal.i_payload;
		auto PacketArray = GetRemoteArray(Nal.p_payload, NalSize);
		OnNalPacket(PacketArray);
		TotalNalSize += NalSize;
	}
	if (TotalNalSize != FrameSize)
		throw Soy::AssertException("NALs output size doesn't match frame size");
}


size_t X264::TEncoder::PushFrameMeta(const std::string& Meta)
{
	TFrameMeta FrameMeta;
	FrameMeta.mFrameNumber = mFrameCount;
	FrameMeta.mMeta = Meta;
	mFrameMetas.PushBack(FrameMeta);
	mFrameCount++;
	return FrameMeta.mFrameNumber;
}

std::string X264::TEncoder::PopFrameMeta(size_t FrameNumber)
{
	for ( auto i=0;	i<mFrameMetas.GetSize();	i++ )
	{
		auto& FrameMeta = mFrameMetas[i];
		if ( FrameMeta.mFrameNumber != FrameNumber )
			continue;
		
		auto Meta = mFrameMetas.PopAt(i);
		return Meta.mMeta;
	}
	
	std::stringstream Error;
	Error << "No frame meta matching frame number " << FrameNumber;
	throw Soy::AssertException(Error);
}
