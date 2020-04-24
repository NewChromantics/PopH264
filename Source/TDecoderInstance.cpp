#include "TDecoderInstance.h"
#include "SoyLib/src/SoyPng.h"
#include "SoyLib/src/SoyImage.h"


//	gr: this works on osx, but currently, none of the functions are implemented :)
//	gr: also needs SDK
#if defined(TARGET_LUMIN) //|| defined(TARGET_OSX)
#define ENABLE_MAGICLEAP_DECODER
#endif

#define ENABLE_BROADWAY

#if defined(TARGET_WINDOWS)
#define ENABLE_INTELMEDIA
#define ENABLE_MEDIAFOUNDATION
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


PopH264::TDecoderInstance::TDecoderInstance(int32_t Mode)
{
#if defined(ENABLE_MAGICLEAP_DECODER)
	if (Mode != MODE_BROADWAY)
	{
		try
		{
			mDecoder.reset(new MagicLeap::TDecoder(Mode));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create MagicLeap decoder: " << e.what() << std::endl;
		}
	}
#endif

#if defined(ENABLE_MEDIAFOUNDATION)
	{
		try
		{
			mDecoder.reset(new MediaFoundation::TDecoder());
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create MediaFoundation decoder: " << e.what() << std::endl;
		}
	}
#endif

#if defined(ENABLE_INTELMEDIA)
	{
		try
		{
			mDecoder.reset(new IntelMedia::TDecoder());
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << "Failed to create IntelMedia decoder: " << e.what() << std::endl;
		}
	}
#endif
	
	
#if defined(ENABLE_BROADWAY)
	mDecoder.reset( new Broadway::TDecoder );
	return;
#endif
	
	std::stringstream Error;
	Error << "No decoder supported (mode=" << Mode << ")";
	throw Soy::AssertException(Error);
}


void PopH264::TDecoderInstance::PushData(const uint8_t* Data,size_t DataSize,int32_t FrameNumber)
{
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
				this->PushFrame(Pixels, FrameNumber, DecodeDuration.GetMilliSeconds());
				return;
			}
		}
	}
	catch (std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " trying to detect image caused exception; " << e.what() << std::endl;
	}
	
	auto PushFrame = [this,FrameNumber](const SoyPixelsImpl& Pixels,SoyTime DecodeDuration)
	{
		this->PushFrame( Pixels, FrameNumber, DecodeDuration.GetMilliSeconds() );
	};
	mDecoder->Decode( GetArrayBridge(DataArray), PushFrame );
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
	
	std::Debug << "PoppedFrame(" << FrameNumber << ") Frames Ready x" << mFrames.GetSize() << std::endl;
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

void PopH264::TDecoderInstance::PushFrame(const SoyPixelsImpl& Frame,int32_t FrameNumber,std::chrono::milliseconds DecodeDuration)
{
	TFrame NewFrame;
	NewFrame.mFrameNumber = FrameNumber;
	NewFrame.mPixels.reset( new SoyPixels( Frame ) );
	NewFrame.mDecodeDuration = DecodeDuration;
	
	{
		std::lock_guard<std::mutex> Lock(mFramesLock);
		mFrames.PushBack(NewFrame);
		mMeta = Frame.GetMeta();
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

