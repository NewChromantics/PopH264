#include "PopH264.h"
#include "PopH264DecoderInstance.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyPng.h"
#include "SoyLib/src/SoyImage.h"
#include "Json11/json11.hpp"

//	gr: this works on osx, but currently, none of the functions are implemented :)
//	gr: also needs SDK
#if defined(TARGET_LUMIN) //|| defined(TARGET_OSX)
#define ENABLE_MAGICLEAP_DECODER
#endif

#define ENABLE_BROADWAY
#if defined(TARGET_WINDOWS)
#define ENABLE_INTELMEDIA
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

namespace PopH264
{
	const Soy::TVersion	Version(1,1,1);
	const int32_t		MODE_BROADWAY = 0;
	const int32_t		MODE_HARDWARE = 1;
}




class TInstanceParams
{
public:
	TInstanceParams(int32_t Mode) :
		Mode	( Mode )
	{
	}
	int32_t Mode = -1;
};

#define TInstanceObject	PopH264::TDecoderInstance

#include "InstanceManager.inc"

#if defined(TARGET_LUMIN) || defined(TARGET_ANDROID)
const char* Platform::LogIdentifer = "PopH264";
#endif


#if defined(TARGET_WINDOWS)
BOOL APIENTRY DllMain(HMODULE /* hModule */, DWORD ul_reason_for_call, LPVOID /* lpReserved */)
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
		case DLL_PROCESS_DETACH:
			break;
	}
	return TRUE;
}
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
		std::Debug << mFrames.GetSize() << " frames pending" << std::endl;
	}
	if ( mOnNewFrame )
		mOnNewFrame();
}

	

__export int32_t PopH264_PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size)
{
	auto Function = [&]()
	{
		auto& Decoder = InstanceManager::GetInstance(Instance);
		//	Decoder.PopFrame
		auto Plane0Array = GetRemoteArray(Plane0, Plane0Size);
		auto Plane1Array = GetRemoteArray(Plane1, Plane1Size);
		auto Plane2Array = GetRemoteArray(Plane2, Plane2Size);
		int32_t FrameTimeMs = -1;
		Decoder.PopFrame( FrameTimeMs, GetArrayBridge(Plane0Array), GetArrayBridge(Plane1Array), GetArrayBridge(Plane2Array));
		return FrameTimeMs;
	};
	return SafeCall(Function, __func__, -99 );
}

__export int32_t PopH264_PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber)
{
	auto Function = [&]()
	{
		auto& Decoder = InstanceManager::GetInstance(Instance);
		Decoder.PushData( Data, DataSize, FrameNumber );
		return 0;
	};
	return SafeCall(Function, __func__, -1 );
}

std::string GetMetaJson(const SoyPixelsMeta& Meta)
{
	using namespace json11;
	Json::array PlaneArray;

	auto AddPlaneJson = [&](SoyPixelsMeta& Plane)
	{
		auto Width = static_cast<int>(Plane.GetWidth());
		auto Height = static_cast<int>(Plane.GetHeight());
		auto Channels = static_cast<int>(Plane.GetChannels());
		auto DataSize = static_cast<int>(Plane.GetDataSize());
		auto FormatStr = SoyPixelsFormat::ToString(Plane.GetFormat());
		Json PlaneMeta = Json::object{
			{ "Width",Width },
			{ "Height",Height },
			{ "Channels",Channels },
			{ "DataSize",DataSize },
			{ "Format",FormatStr }
		};
		PlaneArray.push_back(PlaneMeta);
	};

	BufferArray<SoyPixelsMeta, 4> PlaneMetas;
	Meta.GetPlanes(GetArrayBridge(PlaneMetas));
	for (auto p = 0; p < PlaneMetas.GetSize(); p++)
	{
		auto& PlaneMeta = PlaneMetas[p];
		AddPlaneJson(PlaneMeta);
	}

	//	make final object
	//	todo: add frame numbers etc here
	Json MetaJson = Json::object{
		{ "Planes",PlaneArray }
	};
	auto MetaJsonString = MetaJson.dump();
	return MetaJsonString;
}

__export void PopH264_PeekFrame(int32_t Instance, char* JsonBuffer, int32_t JsonBufferSize)
{
	auto Function = [&]()
	{
		auto& Device = InstanceManager::GetInstance(Instance);
		auto& Meta = Device.GetMeta();

		auto Json = GetMetaJson(Meta);
		Soy::StringToBuffer(Json, JsonBuffer, JsonBufferSize);
		return 0;
	};
	SafeCall(Function, __func__, 0);
}


__export void PopH264_GetMeta(int32_t Instance, int32_t* pMetaValues, int32_t MetaValuesCount)
{
	auto Function = [&]()
	{
		auto& Device = InstanceManager::GetInstance(Instance);
		
		auto& Meta = Device.GetMeta();
		
		size_t MetaValuesCounter = 0;
		auto MetaValues = GetRemoteArray(pMetaValues, MetaValuesCount, MetaValuesCounter);
		
		BufferArray<SoyPixelsMeta, 3> PlaneMetas;
		Meta.GetPlanes(GetArrayBridge(PlaneMetas));
		MetaValues.PushBack(PlaneMetas.GetSize());
		
		for ( auto p=0;	p<PlaneMetas.GetSize();	p++ )
		{
			auto& PlaneMeta = PlaneMetas[p];
			//std::Debug << "Outputting plane " << p << "/" << PlaneMetas.GetSize() << "; " << PlaneMeta << std::endl;
			MetaValues.PushBack(PlaneMeta.GetWidth());
			MetaValues.PushBack(PlaneMeta.GetHeight());
			MetaValues.PushBack(PlaneMeta.GetChannels());
			MetaValues.PushBack(PlaneMeta.GetFormat());
			MetaValues.PushBack(PlaneMeta.GetDataSize());
		}
		
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}

