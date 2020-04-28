#include "PopH264.h"
#include "TDecoderInstance.h"
#include "TEncoderInstance.h"
#include "SoyLib/src/SoyPixels.h"
#include "Json11/json11.hpp"
#include "TInstanceManager.h"


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
	const Soy::TVersion	Version(1,2,7);
}



namespace PopH264
{
	TInstanceManager<TEncoderInstance,std::string>	EncoderInstanceManager;
	TInstanceManager<TDecoderInstance,uint32_t>		DecoderInstanceManager;
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
	return TRUE;
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
		auto& Device = PopH264::DecoderInstanceManager.GetInstance(Instance);
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
		auto& Device = PopH264::DecoderInstanceManager.GetInstance(Instance);
		
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


__export int32_t PopH264_GetVersion()
{
	auto Function = [&]()
	{
		return PopH264::Version.GetMillion();
	};
	return SafeCall( Function, __func__, -1 );
}



__export int32_t PopH264_CreateInstance(int32_t Mode)
{
	auto Function = [&]()
	{
		auto InstanceId = PopH264::DecoderInstanceManager.CreateInstance( Mode );
		return InstanceId;
	};
	return SafeCall( Function, __func__, -1 );
}

__export void PopH264_DestroyInstance(int32_t Instance)
{
	auto Function = [&]()
	{
		PopH264::DecoderInstanceManager.FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
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
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
		return -1;
	}
	catch(...)
	{
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
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
	}
	catch(...)
	{
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferSize );
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

