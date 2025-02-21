#include "TDecoderInstance.h"
#include "SoyLib/src/SoyPng.h"
#include "SoyLib/src/SoyImage.h"
#include "PopH264.h"
#include "json11.hpp"
#include <span>

//	gr: this works on osx, but currently, none of the functions are implemented :)
//	gr: also needs SDK
#if defined(TARGET_LUMIN) //|| defined(TARGET_OSX)
#define ENABLE_MAGICLEAP_DECODER
#endif

#if defined(TARGET_OSX)||defined(TARGET_IOS)
#define ENABLE_AVF
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
	auto SetInt = [&](const char* Name, int32_t& Value)
	{
		auto& Handle = Options[Name];
		if (!Handle.is_number())
			return false;
		Value = Handle.int_value();
		return true;
	};
	
	mDecoderName = Options[std::string(POPH264_DECODER_KEY_DECODERNAME)].string_value();
	SetBool( POPH264_DECODER_KEY_VERBOSEDEBUG, mVerboseDebug );
	SetBool( POPH264_DECODER_KEY_ALLOWBUFFERING, mAllowBuffering );
	SetBool( POPH264_DECODER_KEY_DOUBLEDECODEKEYFRAME, mDoubleDecodeKeyframe);
	SetBool( POPH264_DECODER_KEY_DRAINONKEYFRAME, mDrainOnKeyframe);
	SetBool( POPH264_DECODER_KEY_LOWPOWERMODE, mLowPowerMode);
	SetBool( POPH264_DECODER_KEY_DROPBADFRAMES, mDropBadFrames );
	SetBool( POPH264_DECODER_KEY_DECODESEI, mDecodeSei );
	SetBool( POPH264_DECODER_KEY_STRIPEMULATIONPREVENTION, mStripEmulationPrevention );
	
	SetInt( POPH264_DECODER_KEY_WIDTHHINT, mWidthHint );
	SetInt( POPH264_DECODER_KEY_HEIGHTHINT, mHeightHint );
	SetInt( POPH264_DECODER_KEY_INPUTSIZEHINT, mInputSizeHint );
}



PopH264::TDecoderInstance::TDecoderInstance(json11::Json& Options)
{
	auto OnFrameDecoded = [this](const SoyPixelsImpl& Pixels,FrameNumber_t FrameNumber,const json11::Json& Meta)
	{
		auto& MetaJson = Meta.object_items();
		this->PushFrame( Pixels, FrameNumber, MetaJson );
	};
	
	auto OnError = [this](const std::string& Error,FrameNumber_t* FrameNumber)
	{
		//	No frame, so assuming fatal error
		if ( !FrameNumber )
		{
			OnFatalError(Error);
			return;
		}
		else
		{
			PushErrorFrame( Error, *FrameNumber );
		}
	};
	
	TDecoderParams Params(Options);
	auto AnyDecoder = Params.mDecoderName.empty();

#if defined(TARGET_ANDROID)
	if ( AnyDecoder || Params.mDecoderName == Android::TDecoder::Name )
	{
		try
		{
			mDecoder.reset(new Android::TDecoder(Params,OnFrameDecoded,OnError));
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
			mDecoder.reset(new Avf::TDecoder(Params,OnFrameDecoded,OnError));
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
			mDecoder.reset(new MagicLeap::TDecoder(Mode,OnFrameDecoded,OnError));
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
			mDecoder.reset(new MediaFoundation::TDecoder(Params,OnFrameDecoded,OnError));
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
			mDecoder.reset(new IntelMedia::TDecoder(OnFrameDecoded,OnError));
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
	
	std::stringstream Error;
	Error << "No decoder supported (DecoderName=" << Params.mDecoderName << ")";
	throw Soy::AssertException(Error);
}

PopH264::TDecoderInstance::~TDecoderInstance()
{
	//	make sure decoder destroys before lock
	Soy::TScopeTimerPrint Timer3(__PRETTY_FUNCTION__,20);
	//std::Debug << __PRETTY_FUNCTION__ << " Destroying decoder..." <<std::endl;
	mDecoder.reset();
	//std::Debug << __PRETTY_FUNCTION__ << " decoder destroyed." <<std::endl;
}

void PopH264::TDecoderInstance::PushData(std::span<uint8_t> Data,FrameNumber_t FrameNumber)
{
	auto Decoder = mDecoder;
	if ( !Decoder )
	{
		std::stringstream Error;
		Error << "PushData(x" << Data.size() << " frame=" << FrameNumber << ") to instance with null decoder";
		throw std::runtime_error(Error.str());
	}

	//	if user passes no data, we want to end stream/flush
	if ( Data.empty() )
	{
		//	gr: get rid of the CheckDecoderUpdate hack
		if (FrameNumber == 1)
		{
			Decoder->CheckDecoderUpdates();
		}
		else
		{
			Decoder->PushEndOfStream();
		}
		return;
	}
	
	Decoder->Decode( Data, FrameNumber );
}

void PopH264::TDecoderInstance::PushEndOfStream()
{
	auto Decoder = mDecoder;
	if ( !Decoder )
	{
		std::stringstream Error;
		Error << "PushEndOfStream to instance with null decoder";
		throw std::runtime_error(Error.str());
	}
	Decoder->PushEndOfStream();	
}




void PopH264::TDecoderInstance::PopFrame(int32_t& FrameNumber,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2)
{
	TFrame Frame;
	if ( !PopFrame( Frame ) )
	{
		FrameNumber = -1;
		return;
	}
	
	Soy::TScopeTimerPrint Timer3("PopH264::TDecoderInstance::PopFrame Split&Copy planes",2);
		
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
	Soy::TScopeTimerPrint Timer1("PopH264::TDecoderInstance::PopFrame",2);
	std::lock_guard<std::mutex> Lock(mFramesLock);
	if ( mFrames.IsEmpty() )
		return false;
	
	{
		Soy::TScopeTimerPrint Timer2("PopH264::TDecoderInstance::PopFrame copy",2);
		Frame = mFrames[0];
	}
		
	{
		Soy::TScopeTimerPrint Timer3("PopH264::TDecoderInstance::PopFrame remove block",2);
		mFrames.RemoveBlock(0,1);
	}

	Frame.mPoppedTime = SoyTime::Now();
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
			Meta.mMeta = Frame0.mMeta;
			Meta.mDecodedTime = Frame0.mDecodedTime;
		}
	}
	return Meta;
}

void PopH264::TDecoderInstance::PushErrorFrame(const std::string& Error,PopH264::FrameNumber_t FrameNumber)
{
	//	invalid pixels
	SoyPixels Pixels;
	
	//	maybe this should be a warning
	json11::Json::object Meta;
	Meta["Error"] = Error;
	
	try
	{
		PushFrame( Pixels, FrameNumber, Meta );
	}
	catch(std::exception&e)
	{
		std::Debug << "PushFrame exception " << e.what() << std::endl;
	}
}

void PopH264::TDecoderInstance::OnFatalError(const std::string& Error)
{
	std::Debug << "Decoder fatal error: " << Error << std::endl;
	PushErrorFrame(std::string("OnFatalError=")+Error,PopH264::FrameNumber_t(-1));
}


void PopH264::TDecoderInstance::PushFrame(const SoyPixelsImpl& Frame,PopH264::FrameNumber_t FrameNumber,const json11::Json::object& Meta)
{
	TFrame NewFrame;
	NewFrame.mFrameNumber = FrameNumber;
	NewFrame.mMeta = Meta;
	NewFrame.mDecodedTime = SoyTime::Now();

	//	if we get an invalid pixels we're assuming it's the EndOfStream packet
	if ( !Frame.GetMeta().IsValid() && FrameNumber == 0 )
	{
		//std::Debug << __PRETTY_FUNCTION__ << " detected EndOfStream frame" << std::endl;
		NewFrame.mEndOfStream = true;
	}
	else
	{
		//	todo: get rid of the copy here, maybe swap for a lockable TPixelBuffer so it can be pooled
		Soy::TScopeTimerPrint Timer1("PopH264::TDecoderInstance::PushFrame CopyPixels",2);
		NewFrame.mPixels.reset( new SoyPixels( Frame ) );
	}

	{
		Soy::TScopeTimerPrint Timer2("PopH264::TDecoderInstance::PushFrame lock",2);
		std::lock_guard<std::mutex> Lock(mFramesLock);
		Timer2.Stop();

		Soy::TScopeTimerPrint Timer3("PopH264::TDecoderInstance::PushFrame PushBack frame",2);
		mFrames.PushBack(NewFrame);
		Timer3.Stop();
		
		//	gr: don't overwrite cached meta with an invalid one!
		//		but do update it, in case something has changed
		auto FrameMeta = Frame.GetMeta();
		if ( FrameMeta.IsValid() )
			mMeta = FrameMeta;
		//std::Debug << __PRETTY_FUNCTION__ << mFrames.GetSize() << " frames pending" << std::endl;
	}
	if ( mOnNewFrame )
	{
		Soy::TScopeTimerPrint Timer3("PopH264::TDecoderInstance::PushFrame mOnNewFrame",1);
		mOnNewFrame();
	}
}

void PopH264::TDecoderInstance::AddOnNewFrameCallback(std::function<void()> Callback)
{
	//	does this need to be threadsafe?
	mOnNewFrame = Callback;
}

