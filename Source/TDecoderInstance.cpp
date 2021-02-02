#include "TDecoderInstance.h"
#include "SoyLib/src/SoyPng.h"
#include "SoyLib/src/SoyImage.h"
#include "PopH264.h"
#include "json11.hpp"

//	gr: this works on osx, but currently, none of the functions are implemented :)
//	gr: also needs SDK
#if defined(TARGET_LUMIN) //|| defined(TARGET_OSX)
#define ENABLE_MAGICLEAP_DECODER
#endif

#if defined(TARGET_OSX)||defined(TARGET_IOS)
#define ENABLE_AVF
#endif

#if !defined(TARGET_LINUX)
#define ENABLE_BROADWAY
#endif

#if defined(TARGET_WINDOWS)
//#define ENABLE_INTELMEDIA
#define ENABLE_MEDIAFOUNDATION
#endif

#if defined(ENABLE_AVF)
#include "AvfDecoder.h"
#endif

#if defined(ENABLE_MAGICLEAP_DECODER)
#include "MagicLeapDecoder.h"
#endif

#if defined(ENABLE_BROADWAY)
#include "BroadwayDecoder.h"
#endif

#if defined(ENABLE_INTELMEDIA)
#include "IntelMediaDecoder.h"
#endif

#if defined(ENABLE_MEDIAFOUNDATION)
#include "MediaFoundationDecoder.h"
#endif

#if defined(TARGET_ANDROID)
#include "AndroidDecoder.h"
#endif


void PopH264::EnumDecoderNames(std::function<void(const std::string&)> EnumDecoderName)
{
#if defined(TARGET_ANDROID)
	EnumDecoderName(Android::TDecoder::Name);
#endif

#if defined(ENABLE_AVF)
	EnumDecoderName(Avf::TDecoder::Name);
#endif

	//	todo: enum sub-decoder names
#if defined(ENABLE_MAGICLEAP_DECODER)
	EnumDecoderName(MagicLeap::TDecoder::Name);
#endif

#if defined(ENABLE_INTELMEDIA)
	EnumDecoderName(Intel::TDecoder::Name);
#endif

	//	todo: enum sub-decoder names
#if defined(ENABLE_MEDIAFOUNDATION)
	EnumDecoderName(MediaFoundation::TDecoder::Name);
#endif

#if defined(ENABLE_BROADWAY)
	EnumDecoderName(Broadway::TDecoder::Name);
#endif
}


PopH264::TDecoderParams::TDecoderParams(json11::Json& Options)
{
	auto SetBool = [&](const char* Name,bool& Value)
	{
		auto& Handle = Options[Name];
		if ( !Handle.is_bool() )
			return false;
		Value = Handle.bool_value();
		return true;
	};
	
	mDecoderName = Options[std::string(POPH264_DECODER_KEY_DECODERNAME)].string_value();
	SetBool( POPH264_DECODER_KEY_VERBOSEDEBUG, mVerboseDebug );
	SetBool( POPH264_DECODER_KEY_MINBUFFERING, mMinmalBuffering );
}



PopH264::TDecoderInstance::TDecoderInstance(json11::Json& Options)
{
	auto OnFrameDecoded = [this](const SoyPixelsImpl& Pixels,size_t FrameNumber)
	{
		this->PushFrame(Pixels, FrameNumber );
	};
	
	TDecoderParams Params(Options);
	auto AnyDecoder = Params.mDecoderName.empty();

#if defined(TARGET_ANDROID)
	if ( AnyDecoder || Params.mDecoderName == Android::TDecoder::Name )
	{
		try
		{
			mDecoder.reset(new Android::TDecoder(Params,OnFrameDecoded));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create Android decoder: " << e.what() << std::endl;
			
			if ( !AnyDecoder )
				throw;
		}
	}
#endif



#if defined(ENABLE_AVF)
	if ( AnyDecoder || Params.mDecoderName == Avf::TDecoder::Name )
	{
		try
		{
			mDecoder.reset(new Avf::TDecoder(OnFrameDecoded));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create Avf decoder: " << e.what() << std::endl;
			if ( !AnyDecoder )
				throw;
		}
	}
#endif
	
#if defined(ENABLE_MAGICLEAP_DECODER)
	if ( AnyDecoder || Soy::StringBeginsWith(Params.mDecoderName,MagicLeap::TDecoder::Name,false))
	{
		try
		{
			mDecoder.reset(new MagicLeap::TDecoder(Mode,OnFrameDecoded));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create MagicLeap decoder: " << e.what() << std::endl;
			if ( !AnyDecoder )
				throw;
		}
	}
#endif

#if defined(ENABLE_MEDIAFOUNDATION)
	if ( AnyDecoder || Soy::StringBeginsWith(Params.mDecoderName,MediaFoundation::TDecoder::Name,false) )
	{
		try
		{
			mDecoder.reset(new MediaFoundation::TDecoder(Params,OnFrameDecoded));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create MediaFoundation decoder: " << e.what() << std::endl;
			if ( !AnyDecoder )
				throw;
		}
	}
#endif

#if defined(ENABLE_INTELMEDIA)
	{
		try
		{
			mDecoder.reset(new IntelMedia::TDecoder(OnFrameDecoded));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create IntelMedia decoder: " << e.what() << std::endl;
			if ( !AnyDecoder )
				throw;
		}
	}
#endif
	
	
#if defined(ENABLE_BROADWAY)
	if ( AnyDecoder || Params.mDecoderName == Broadway::TDecoder::Name)
	{
		mDecoder.reset( new Broadway::TDecoder( Params, OnFrameDecoded ) );
		return;
	}
#endif
	
	std::stringstream Error;
	Error << "No decoder supported (DecoderName=" << Params.mDecoderName << ")";
	throw Soy::AssertException(Error);
}


void PopH264::TDecoderInstance::PushData(const uint8_t* Data,size_t DataSize,size_t FrameNumber)
{
	//	if user passes null, we want to end stream/flush
	if ( Data == nullptr )
	{
		mDecoder->PushEndOfStream();
		return;
	}
	
	auto DataArray = GetRemoteArray( Data, DataSize );
	
	//	gr: temporary hack, if the data coming in is a different format, detect it, and switch decoders
	//		maybe we can do something more elegant (eg. wait until first frame before allocating decoder)
	//	gr: don't even need to interrupt decoder
	try
	{
		//	do fast PNG check, STB is sometimes matching TGA
		if (TPng::IsPngHeader(GetArrayBridge(DataArray)))
		{
			//	calc duration
			SoyTime DecodeDuration;
			auto ImageMeta = Soy::IsImage(GetArrayBridge(DataArray));
			if (ImageMeta.IsValid())
			{
				SoyPixels Pixels;
				Soy::DecodeImage(Pixels, GetArrayBridge(DataArray));
				this->PushFrame(Pixels, FrameNumber );
				return;
			}
		}
	}
	catch (std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " trying to detect image caused exception; " << e.what() << std::endl;
	}
	
	mDecoder->Decode( GetArrayBridge(DataArray), FrameNumber );
}


void PopH264::TDecoderInstance::PopFrame(int32_t& FrameNumber,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2)
{
	TFrame Frame;
	if ( !PopFrame( Frame ) )
	{
		FrameNumber = -1;
		return;
	}
	
	//	if we don't set the correct time the c# thinks we have a bad frame!
	FrameNumber = Frame.mFrameNumber;
	
	//	emulating TPixelBuffer interface
	BufferArray<SoyPixelsImpl*, 10> Textures;
	if ( Frame.mPixels )
		Textures.PushBack( Frame.mPixels.get() );
	
	BufferArray<std::shared_ptr<SoyPixelsImpl>, 10> Planes;
	
	//	get all the planes
	for ( auto t = 0; t < Textures.GetSize(); t++ )
	{
		auto& Texture = *Textures[t];
		Texture.SplitPlanes(GetArrayBridge(Planes));
	}
	
	ArrayBridge<uint8_t>* PlanePixels[] = { &Plane0, &Plane1, &Plane2 };
	for ( auto p = 0; p < Planes.GetSize() && p<3; p++ )
	{
		auto& Plane = *Planes[p];
		auto& PlaneDstPixels = *PlanePixels[p];
		auto& PlaneSrcPixels = Plane.GetPixelsArray();
		
		auto MaxSize = std::min(PlaneDstPixels.GetDataSize(), PlaneSrcPixels.GetDataSize());
		//	copy as much as possible
		auto PlaneSrcPixelsMin = GetRemoteArray(PlaneSrcPixels.GetArray(), MaxSize);
		PlaneDstPixels.Copy(PlaneSrcPixelsMin);
	}
	
	//std::Debug << "PoppedFrame(" << FrameNumber << ") Frames Ready x" << mFrames.GetSize() << std::endl;
}

bool PopH264::TDecoderInstance::PopFrame(TFrame& Frame)
{
	std::lock_guard<std::mutex> Lock(mFramesLock);
	if ( mFrames.IsEmpty() )
		return false;
	
	Frame = mFrames[0];
	mFrames.RemoveBlock(0,1);
	return true;
}


PopH264::TDecoderFrameMeta PopH264::TDecoderInstance::GetMeta()
{
	TDecoderFrameMeta Meta;

	//	set the cached pixel meta in case we have no frames
	Meta.mPixelsMeta = this->mMeta;

	{
		std::lock_guard<std::mutex> Lock(mFramesLock);
		if ( !mFrames.IsEmpty() )
		{
			auto& Frame0 = mFrames[0];
			Meta.mEndOfStream = Frame0.mEndOfStream;
			Meta.mFrameNumber = Frame0.mFrameNumber;
			Meta.mFramesQueued = mFrames.GetSize();
			if ( Frame0.mPixels )
				Meta.mPixelsMeta = Frame0.mPixels->GetMeta();
		}
	}
	return Meta;
}

void PopH264::TDecoderInstance::PushFrame(const SoyPixelsImpl& Frame,size_t FrameNumber)
{
	TFrame NewFrame;
	NewFrame.mFrameNumber = FrameNumber;

	//	if we get an invalid pixels we're assuming it's the EndOfStream packet
	if ( !Frame.GetMeta().IsValid() && FrameNumber == 0 )
	{
		std::Debug << __PRETTY_FUNCTION__ << " detected EndOfStream frame" << std::endl;
		NewFrame.mEndOfStream = true;
	}
	else
	{
		//	todo: get rid of the copy here, maybe swap for a lockable TPixelBuffer so it can be pooled
		NewFrame.mPixels.reset( new SoyPixels( Frame ) );
	}

	{
		std::lock_guard<std::mutex> Lock(mFramesLock);
		mFrames.PushBack(NewFrame);
		
		//	gr: don't overwrite cached meta with an invalid one!
		//		but do update it, in case something has changed
		auto FrameMeta = Frame.GetMeta();
		if ( FrameMeta.IsValid() )
			mMeta = FrameMeta;
		//std::Debug << __PRETTY_FUNCTION__ << mFrames.GetSize() << " frames pending" << std::endl;
	}
	if ( mOnNewFrame )
		mOnNewFrame();
}

void PopH264::TDecoderInstance::AddOnNewFrameCallback(std::function<void()> Callback)
{
	//	does this need to be threadsafe?
	mOnNewFrame = Callback;
}

