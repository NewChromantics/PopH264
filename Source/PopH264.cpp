#include "PopH264.h"
#include "TDecoderInstance.h"
#include "TEncoderInstance.h"
#include "SoyLib/src/SoyPixels.h"
#include "Json11/json11.hpp"
#include "TInstanceManager.h"
#include "PopH264TestData.h"

namespace PopH264
{
	//	1.2.0	removed access to c++ decoder object
	//	1.2.1	added encoding
	//	1.2.2	Added PopH264_DecoderAddOnNewFrameCallback
	//	1.2.3	PopH264_CreateEncoder now takes JSON instead of encoder name
	//	1.2.4	Added ProfileLevel
	//	1.2.5	Encoder now uses .Keyframe meta setting
	//	1.2.6	X264 now uses ProfileLevel + lots of x264 settings exposed
	//	1.2.7	Added MediaFoundation decoder to windows
	//	1.2.8	Added Test data
	//	1.2.9	Added PopH264_EncoderEndOfStream
	//	1.2.10	Added PopH264_Shutdown
	//	1.2.11	Added nvidia hardware decoder + new settings
	//	1.2.12/13 Temp numbers for continious build fixes
	//	1.2.14	Fixed MediaFoundation encoder not outputing meta
	//	1.2.15	Added KeyFrameFrequency option. AVF now encodes timestamps/framenumbers better producing much smaller packets
	//	1.2.16	Nvidia encoder now outputting input meta
	//	1.3.0	Decoder now created with Json. EnumDecoders added
	//	1.3.1	Mediafoundation decoder working properly 
	//	1.3.x	Meta versions for packaging
	//	1.3.15	Fixed/fixing correct frame number output to match input of decoder
	//	1.3.17	MediaFoundation now doesn't get stuck if we try and decode PPS or frames before SPS
	//	1.3.18	Broadway doesn't get stuck if we dont process in the order SPS, PPS, Keyframe
	//	1.3.19	Android NDK MediaCodec implementation
	//	1.3.20	Added PopH264_PushEndOfStream API as a clear wrapper for PopH264_PushData(null)
	//	1.3.21	Added AllowBuffering option to decoder so by default LowLatency mode is on for Mediafoundation, which reduces buffering
	//	1.3.22	Version bump for github build
	//	1.3.23	Added extra meta output from decoder (Just MediaFoundation initially)
	//	1.3.24	Version bump for github release
	//	1.3.25	Android wasn't handling COLOR_FormatYUV420SemiPlanar from some devices (samsung s7), is now
	//	1.3.26	Fixed android erroring with mis-aligned/padded buffers. 
	//	1.3.27	Android now outputting ImageRect (cropping rect) meta
	//	1.3.28	Fixed various MediaFoundation memory leaks
	//	1.3.29	Fixed MediaFoundation memory leak. Added Decoder name to meta. Added Unity GetFrameAndMeta interface
	//	1.3.30	Avf (Mac & ios) no longer try and decode SEI nalus (causes -12349 error). Added option to disable this
	//	1.3.31	Added width/height/inputsize hints for android to try and get bigger input buffers; issue #48
	//	1.3.32	Added extra timer debug. Android, if given 0 as size hints, will try and extract hint sizes from SPS
	//	1.3.33	First wasm version of broadway compiled from our repository
	//	1.3.34	Added some extra macos/ios version number settings for mac-swift app compatibility
	const Soy::TVersion	Version(1,3,34);
}



namespace PopH264
{
	TInstanceManager<TEncoderInstance,std::string>		EncoderInstanceManager;
	TInstanceManager<TDecoderInstance,json11::Json&>	DecoderInstanceManager;
	
	void		Shutdown(bool DllExit);
}




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
	
	//	hail mary cleanup for globals
	if ( ul_reason_for_call == DLL_PROCESS_DETACH)
	{
		PopH264::Shutdown(true);
	}
	
	return TRUE;
}
#endif

#if !defined(TARGET_WINDOWS)
void __attribute__((destructor)) DllExit() 
{
	PopH264::Shutdown(true);
}
#endif

template<typename RETURN,typename FUNC>
RETURN SafeCall(FUNC Function,const char* FunctionName,RETURN ErrorReturn)
{
	try
	{
		return Function();
	}
	catch(std::exception& e)
	{
		std::Debug << FunctionName << " exception: " << e.what() << std::endl;
		return ErrorReturn;
	}
	catch(...)
	{
		std::Debug << FunctionName << " unknown exception." << std::endl;
		return ErrorReturn;
	}
}

	

__export int32_t PopH264_PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size)
{
	auto Function = [&]()
	{
		auto& Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
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
		auto& Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
		Decoder.PushData( Data, DataSize, FrameNumber );
		return 0;
	};
	return SafeCall(Function, __func__, -1 );
}

__export int32_t PopH264_PushEndOfStream(int32_t Instance)
{
	return PopH264_PushData(Instance, nullptr, 0, 0);
}


json11::Json::object GetMetaJson(json11::Json::object& MetaJson,const SoyPixelsMeta& Meta)
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
	MetaJson["Planes"] = PlaneArray;

	return MetaJson;
}

std::string GetMetaJson(const PopH264::TDecoderFrameMeta& Meta)
{
	auto Json = Meta.mMeta;
	GetMetaJson( Json, Meta.mPixelsMeta);
	
	if ( Meta.mEndOfStream )
		Json["EndOfStream"] = true;
	
	Json["FrameNumber"] = Meta.mFrameNumber;
	Json["QueuedFrames"] = static_cast<int>(Meta.mFramesQueued);
	
	json11::Json TheJson = Json;
	std::string MetaJsonString = TheJson.dump();
	return MetaJsonString;
}


__export void PopH264_PeekFrame(int32_t Instance, char* JsonBuffer, int32_t JsonBufferSize)
{
	auto Function = [&]()
	{
		auto& Device = PopH264::DecoderInstanceManager.GetInstance(Instance);
		auto Meta = Device.GetMeta();

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
		auto& Device = PopH264::DecoderInstanceManager.GetInstance(Instance);
		
		auto Meta = Device.GetMeta();
		
		size_t MetaValuesCounter = 0;
		auto MetaValues = GetRemoteArray(pMetaValues, MetaValuesCount, MetaValuesCounter);
		
		BufferArray<SoyPixelsMeta, 3> PlaneMetas;
		Meta.mPixelsMeta.GetPlanes(GetArrayBridge(PlaneMetas));
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


__export int32_t PopH264_GetVersion()
{
	auto Function = [&]()
	{
		return PopH264::Version.GetMillion();
	};
	return SafeCall( Function, __func__, -1 );
}


__export void PopH264_Shutdown()
{
	auto Function = [&]()
	{
		PopH264::Shutdown(false);
		return 0;
	};
	SafeCall( Function, __func__, 0 );
}



__export void PopH264_EnumDecoders(char* DecodersJsonBuffer, int32_t DecodersJsonBufferLength)
{
	try
	{
		json11::Json::array DecoderNames;
		auto EnumDecoder = [&](const std::string& DecoderName)
		{
			DecoderNames.push_back(DecoderName);
		};
		PopH264::EnumDecoderNames(EnumDecoder);

		json11::Json::object Meta;
		Meta["DecoderNames"] = DecoderNames;
		
		auto MetaString = json11::Json(Meta).dump();
		Soy::StringToBuffer(MetaString, DecodersJsonBuffer, DecodersJsonBufferLength);
	}
	catch (std::exception& e)
	{
		Soy::StringToBuffer(e.what(), DecodersJsonBuffer, DecodersJsonBufferLength);
	}
	catch (...)
	{
		Soy::StringToBuffer("Unknown exception", DecodersJsonBuffer, DecodersJsonBufferLength);
	}
}


__export int32_t PopH264_CreateDecoder(const char* OptionsJson, char* ErrorBuffer, int32_t ErrorBufferLength)
{
	try
	{
		std::string ParseError;
		json11::Json Options = json11::Json::parse(OptionsJson ? OptionsJson : "{}", ParseError);
		if (!ParseError.empty())
		{
			ParseError = std::string("PopCameraDevice_CreateCameraDevice parse json error; ") + ParseError;
			throw Soy::AssertException(ParseError);
		}
		auto InstanceId = PopH264::DecoderInstanceManager.CreateInstance(Options);
		return InstanceId;
	}
	catch (std::exception& e)
	{
		Soy::StringToBuffer(e.what(), ErrorBuffer, ErrorBufferLength);
		return 0;
	}
	catch (...)
	{
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferLength);
		return 0;
	}
}

__export void PopH264_DestroyDecoder(int32_t Instance)
{
	auto Function = [&]()
	{
		PopH264::DecoderInstanceManager.FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_DestroyInstance(int32_t Instance)
{
	PopH264_DestroyDecoder(Instance);
}




__export int32_t PopH264_CreateEncoder(const char* OptionsJson,char* ErrorBuffer,int32_t ErrorBufferSize)
{
	try
	{
		std::string ParamsJson(OptionsJson ? OptionsJson : "{}");
		auto InstanceId = PopH264::EncoderInstanceManager.CreateInstance( ParamsJson );
		return InstanceId;
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
		return -1;
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferSize );
		return -1;
	}
}

__export void PopH264_DestroyEncoder(int32_t Instance)
{
	auto Function = [&]()
	{
		PopH264::EncoderInstanceManager.FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_EncoderPushFrame(int32_t Instance,const char* MetaJson,const uint8_t* LumaData,const uint8_t* ChromaUData,const uint8_t* ChromaVData,char* ErrorBuffer,int32_t ErrorBufferSize)
{
	try
	{
		auto& Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		std::string Meta( MetaJson ? MetaJson : "" );
		Encoder.PushFrame( Meta, LumaData, ChromaUData, ChromaVData );
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferSize );
	}
}


__export void PopH264_EncoderEndOfStream(int32_t Instance)
{
	try
	{
		auto& Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		Encoder.EndOfStream();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
	}
}


__export int32_t PopH264_EncoderPopData(int32_t Instance,uint8_t* DataBuffer,int32_t DataBufferSize)
{
	auto Function = [&]()
	{
		auto& Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		//	no data buffer, just peeking size
		if ( !DataBuffer || DataBufferSize <= 0 )
			return Encoder.PeekNextFrameSize();
		
		size_t DataBufferUsed = 0;
		auto DataArray = GetRemoteArray( DataBuffer, DataBufferSize, DataBufferUsed );
		Encoder.PopPacket( GetArrayBridge(DataArray) );
		return DataBufferUsed;
	};
	return SafeCall(Function, __func__, -1 );
}

__export void PopH264_EncoderPeekData(int32_t Instance,char* MetaJsonBuffer,int32_t MetaJsonBufferSize)
{
	//	wrapped in a safe call in case the json generation fails in some way
	auto Function = [&]()
	{
		using namespace json11;
		Json::object MetaJson;
		try
		{
			//	get next frame's meta
			auto& Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);

			//	add generic meta
			MetaJson["OutputQueueCount"] = static_cast<int32_t>(Encoder.GetPacketQueueCount());
			
			Encoder.PeekPacket(MetaJson);
		}
		catch(std::exception& e)
		{
			MetaJson["Error"] = e.what();
		}
		catch(...)
		{
			MetaJson["Error"] = "Unknown exception";
		}
		Json FinalJson(MetaJson);
		auto MetaJsonString = FinalJson.dump();
		Soy::StringToBuffer( MetaJsonString, MetaJsonBuffer, MetaJsonBufferSize );

		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_EncoderAddOnNewPacketCallback(int32_t Instance,PopH264_Callback* Callback, void* Meta)
{
	auto Function = [&]()
	{
		if (!Callback)
			throw Soy::AssertException("PopH264_EncoderAddOnNewPacketCallback callback is null");
		
		auto Lambda = [Callback, Meta]()
		{
			Callback(Meta);
		};
		auto& Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		Encoder.AddOnNewFrameCallback(Lambda);
		return 0;
	};
	SafeCall(Function, __func__, 0);
}


__export void PopH264_DecoderAddOnNewFrameCallback(int32_t Instance,PopH264_Callback* Callback, void* Meta)
{
	auto Function = [&]()
	{
		if (!Callback)
			throw Soy::AssertException("PopH264_DecoderAddOnNewFrameCallback callback is null");
		
		auto Lambda = [Callback, Meta]()
		{
			Callback(Meta);
		};
		auto& Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
		Decoder.AddOnNewFrameCallback(Lambda);
		return 0;
	};
	SafeCall(Function, __func__, 0);
}

__export int32_t PopH264_GetTestData(const char* Name,uint8_t* Buffer,int32_t BufferSize)
{
	auto Function = [&]()
	{
		auto BufferArray = GetRemoteArray( Buffer, BufferSize );
		size_t FullSize = 0;

		//	todo: catch "name doesnt exist"
		PopH264::GetTestData( Name, GetArrayBridge(BufferArray), FullSize );
		return FullSize;
	};
	return SafeCall(Function, __func__, -1);
}



__export void UnityPluginLoad(/*IUnityInterfaces*/void*)
{
	//	if this DLL is being used in unity, we dont get any debug output in xcode, so enable NSLog
	Debug::EnablePrint_Platform = true;
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
}

__export void UnityPluginUnload()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	PopH264_Shutdown();
}


void PopH264::Shutdown(bool FromDllExit)
{
	PopH264::EncoderInstanceManager.FreeInstances();
	PopH264::DecoderInstanceManager.FreeInstances();
}



