#include "X264Encoder.h"
#include "SoyPixels.h"
#include "PopH264.h"	//	param keys
#include "json11.hpp"
#include "SoyRuntimeLibrary.h"

namespace X264
{
	void	IsOkay(int Result, const char* Context);
	void	Log(void *data, int i_level, const char *psz, va_list args);
	
	int		GetColourSpace(SoyPixelsFormat::Type Format);
	
	class TPacket;
}


//	without making seperate targets, can't build arm vs intel at the moment, and we don't have arm
//	builds of x264 (we should move to OpenH264 anyway)
//	but we still link to x86 x264, linker ignores it, so we get missing symbols 
#if defined(TARGET_OSX) && !defined(TARGET_ARCH_INTEL64)
#define NOT_SUPPORTED	{	throw Soy::AssertException("x264 not supported on this architecture");	}
int	x264_encoder_delayed_frames(x264_t*)	NOT_SUPPORTED;	
int	x264_encoder_encode( x264_t *, x264_nal_t **pp_nal, int *pi_nal, x264_picture_t *pic_in, x264_picture_t *pic_out )	NOT_SUPPORTED;	
x264_t*	x264_encoder_open( x264_param_t * )	NOT_SUPPORTED;	
int	x264_param_apply_profile( x264_param_t *, const char *profile )	NOT_SUPPORTED;	
int	x264_param_default_preset( x264_param_t *, const char *preset, const char *tune )	NOT_SUPPORTED;	
int	x264_picture_alloc( x264_picture_t *pic, int i_csp, int i_width, int i_height )	NOT_SUPPORTED;	
#endif




void X264::IsOkay(int Result, const char* Context)
{
	if (Result == 0)
		return;
	
	std::stringstream Error;
	Error << "X264 error " << Result << " (" << Context << ")";
	throw Soy::AssertException(Error);
}

X264::TEncoderParams::TEncoderParams(json11::Json& Options)
{
	auto SetInt = [&](const char* Name,size_t& ValueUnsigned)
	{
		auto& Handle = Options[Name];
		if ( !Handle.is_number() )
			return false;
		auto Value = Handle.int_value();
		if ( Value < 0 )
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
	SetInt(POPH264_ENCODER_KEY_QUALITY, mPreset);
	SetInt(POPH264_ENCODER_KEY_PROFILELEVEL, mProfileLevel);
	SetInt(POPH264_ENCODER_KEY_ENCODERTHREADS, mEncoderThreads);
	SetInt(POPH264_ENCODER_KEY_LOOKAHEADTHREADS, mLookaheadThreads);
	SetBool(POPH264_ENCODER_KEY_BSLICEDTHREADS, mBSlicedThreads);
	SetBool(POPH264_ENCODER_KEY_VERBOSEDEBUG, mEnableLog);
	SetBool(POPH264_ENCODER_KEY_DETERMINISTIC, mDeterministic);
	SetBool(POPH264_ENCODER_KEY_CPUOPTIMISATIONS, mCpuOptimisations);

	//	0 is auto on AVF, so handle that
	if (mProfileLevel == 0)
		mProfileLevel = 30;
}

X264::TEncoder::TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket ),
	mParams				( Params )
{
	if ( mParams.mPreset > 9 )
		throw Soy_AssertException("Expecting preset value <= 9");
	
	//	trigger dll load
#if defined(TARGET_WINDOWS)
	Soy::TRuntimeLibrary Dll("x264.dll");
#endif
	
	//	todo: allow user to set tune options. takes , seperated values
	const char* Tune = "zerolatency";
	auto* PresetName = x264_preset_names[mParams.mPreset];
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
		case SoyPixelsFormat::Yuv_8_8_8:	return X264_CSP_I420;
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

	mParam.i_log_level = mParams.mEnableLog ? X264_LOG_DEBUG : X264_LOG_WARNING;
	
	//	reduce mem usage by reducing threads
	//	gr: this heavily slows encoding (20ms -> 50/60) on desktop
	mParam.i_threads = mParams.mEncoderThreads;
	mParam.i_lookahead_threads = mParams.mLookaheadThreads;
	mParam.b_sliced_threads = mParams.mBSlicedThreads;
	mParam.b_deterministic = mParams.mDeterministic;
	mParam.b_cpu_independent = mParams.mCpuOptimisations;

	//	h264 profile level
	mParam.i_level_idc = mParams.mProfileLevel;//	3.0
	
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


void X264::TEncoder::Encode(const SoyPixelsImpl& Pixels, const std::string& Meta, bool Keyframe)
{
	//	x264 needs 3 planes, convert
	if ( Pixels.GetFormat() != SoyPixelsFormat::Yuv_8_8_8 )
	{
		SoyPixels Yuv;
		Yuv.Copy(Pixels);
		Yuv.SetFormat(SoyPixelsFormat::Yuv_8_8_8);
		Encode(Yuv,Meta,Keyframe);
		return;
	}
	
	BufferArray<std::shared_ptr<SoyPixelsImpl>,4> Planes;
	Pixels.SplitPlanes(GetArrayBridge(Planes));
	Encode( *Planes[0], *Planes[1], *Planes[2], Meta, Keyframe );
}

void X264::TEncoder::Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta,bool Keyframe)
{
	Soy::TScopeTimerPrint Timer(__PRETTY_FUNCTION__, 2);
	{
		auto YuvFormat = SoyPixelsFormat::GetMergedFormat( Luma.GetFormat(), ChromaU.GetFormat(), ChromaV.GetFormat() );
		auto YuvWidth = Luma.GetWidth();
		auto YuvHeight = Luma.GetHeight();
		SoyPixelsMeta YuvMeta( YuvWidth, YuvHeight, YuvFormat );
		AllocEncoder(YuvMeta);
	}
	
	BufferArray<const SoyPixelsImpl*, 3> Planes;
	Planes.PushBack(&Luma);
	Planes.PushBack(&ChromaU);
	Planes.PushBack(&ChromaV);

	//	checks from example code https://github.com/jesselegg/x264/blob/master/example.c
	//	gr: look for proper validation funcs
	auto Width = Luma.GetWidth();
	auto Height = Luma.GetHeight();
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
	
	mPicture.i_type = Keyframe ? X264_TYPE_KEYFRAME : X264_TYPE_AUTO;

	Encode(&mPicture);
	
	//	flush any other frames
	//	gr: this is supposed to only be called at the end of the stream...
	//		if DelayedFrameCount non zero, we may haveto call multiple times before nal size is >0
	//		so just keep calling until we get 0
	//	maybe add a safety iteration check
	//	gr: need this on OSX (latest x264) but on windows (old build) every subsequent frame fails
	//	gr: this was backwards? brew (old 2917) DID need to flush?
#if !defined(TARGET_LINUX)
{
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
#endif
}



void X264::TEncoder::FinishEncoding()
{
	if ( !mHandle )
		return;
	
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
	static int LastDecodeOrderNumber = -1;
	static int LastPictureOrderNumber = -1;

	//	gr: currently, decoder NEEDS to have nal packets split
	auto OnNalPacket = [&](std::span<uint8_t> Data)
	{
		Soy::TScopeTimerPrint Timer("OnNalPacket",2);
		auto DecodeOrderNumber = mPicture.i_dts;
		auto FrameNumber = mPicture.i_pts;
		
		auto DecodeInOrder = (DecodeOrderNumber == LastDecodeOrderNumber + 1) || (DecodeOrderNumber == LastDecodeOrderNumber);
		auto PictureInOrder = (FrameNumber == LastPictureOrderNumber + 1) || (FrameNumber == LastPictureOrderNumber);
		if ( !DecodeInOrder || !PictureInOrder )
			std::Debug << "OnNalPacket out of order( pts=" << FrameNumber << ", lastpts=" << LastPictureOrderNumber << " dts=" << DecodeOrderNumber << ") DecodeInOrder=" << DecodeInOrder << " PictureInOrder=" << PictureInOrder << std::endl;
		LastDecodeOrderNumber = DecodeOrderNumber;
		LastPictureOrderNumber = FrameNumber;

		auto FrameMeta = GetFrameMeta(FrameNumber);
		
		//	todo: either store these to make sure decode order (dts) is kept correct
		//		or send DTS order to TPacket for host to order
		//	todo: insert DTS into meta anyway!
		//	gr: DTS is 0 all of the time, I think there's a setting to allow out of order
		PopH264::TPacket OutputPacket;
		OutputPacket.mData.reset(new std::vector<uint8_t>());
		OutputPacket.mInputMeta = FrameMeta;
		std::copy( Data.begin(), Data.end(), std::back_inserter(*OutputPacket.mData) );
		OnOutputPacket(OutputPacket);
	};
	
	x264_picture_t OutputPicture;
	x264_nal_t* Nals = nullptr;
	int NalCount = 0;
	
	Soy::TScopeTimerPrint EncodeTimer("x264_encoder_encode",10);
	auto FrameSize = x264_encoder_encode(mHandle, &Nals, &NalCount, InputPicture, &OutputPicture);
	EncodeTimer.Stop();
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
		std::span<uint8_t> PacketArray( Nal.p_payload, NalSize );
		//	if this throws we lose a packet!
		OnNalPacket(PacketArray);
		TotalNalSize += NalSize;
	}
	if (TotalNalSize != FrameSize)
		throw Soy::AssertException("NALs output size doesn't match frame size");
}

