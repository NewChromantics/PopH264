#include "PopH264.h"
#include <memory>
#include <mutex>
#include "SoyLib/src/HeapArray.hpp"
#include "BroadwayDecoder.h"

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


__export int32_t GetTestInteger()
{
	return 12345;
}


class TNoParams
{
	
};


using TInstanceObject = Broadway::TDecoder;
class TInstance;
using TInstanceParams = TNoParams;

namespace InstanceManager
{
	TInstanceObject&	GetInstance(uint32_t Instance);
	uint32_t			AssignInstance(std::shared_ptr<TInstanceObject> Object);
	void				FreeInstance(uint32_t Instance);
	uint32_t			CreateInstance(const TInstanceParams& Params);
	
	std::mutex			InstancesLock;
	Array<TInstance>	Instances;
	uint32_t			InstancesCounter = 1;
}


class TInstance
{
public:
	std::shared_ptr<TInstanceObject>	mObject;
	uint32_t							mInstanceId = 0;
	
	bool								operator==(const uint32_t& InstanceId) const	{	return mInstanceId == InstanceId;	}
};



uint32_t InstanceManager::CreateInstance(const TInstanceParams& Params)
{
	//	alloc device
	try
	{
		auto Object = std::make_shared<Broadway::TDecoder>();
		if ( Object )
			return InstanceManager::AssignInstance(Object);
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}
	
	
	throw Soy::AssertException("Failed to create instance");
}




TInstanceObject& InstanceManager::GetInstance(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	auto pInstance = Instances.Find(Instance);
	auto* Device = pInstance ? pInstance->mObject.get() : nullptr;
	if ( !Device )
	{
		std::stringstream Error;
		Error << "No instance/device matching " << Instance;
		throw Soy::AssertException(Error.str());
	}
	
	return *Device;
}


uint32_t InstanceManager::AssignInstance(std::shared_ptr<TInstanceObject> Object)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	
	TInstance Instance;
	Instance.mInstanceId = InstancesCounter;
	Instance.mObject = Object;
	Instances.PushBack(Instance);
	
	InstancesCounter++;
	return Instance.mInstanceId;
}


void InstanceManager::FreeInstance(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	
	auto InstanceIndex = Instances.FindIndex(Instance);
	if ( InstanceIndex < 0 )
	{
		std::Debug << "No instance " << Instance << " to free" << std::endl;
		return;
	}
	
	Instances.RemoveBlock(InstanceIndex, 1);
}



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



__export int32_t CreateInstance()
{
	auto Function = [&]()
	{
		auto InstanceId = InstanceManager::CreateInstance( TInstanceParams() );
		return InstanceId;
	};
	return SafeCall( Function, __func__, -1 );
}

__export void DestroyInstance(int32_t Instance)
{
	auto Function = [&]()
	{
		InstanceManager::FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


namespace Decoder
{
	void	PopFrame(Broadway::TDecoder& Decoder,int32_t& FrameTimeMs,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2);
}

void Decoder::PopFrame(Broadway::TDecoder& Decoder,int32_t& FrameTimeMs,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2)
{
	//Decoder.PopFrame(Plane0, Plane1, Plane2, FrameTimeMs );
	throw Soy::AssertException("Todo; Decoder.PopFrame");
}

__export int32_t PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size)
{
	auto Function = [&]()
	{
		auto& Decoder = InstanceManager::GetInstance(Instance);
		//	Decoder.PopFrame
		auto Plane0Array = GetRemoteArray(Plane0, Plane0Size);
		auto Plane1Array = GetRemoteArray(Plane1, Plane1Size);
		auto Plane2Array = GetRemoteArray(Plane2, Plane2Size);
		int32_t FrameTimeMs = -1;
		Decoder::PopFrame( Decoder, FrameTimeMs, GetArrayBridge(Plane0Array), GetArrayBridge(Plane1Array), GetArrayBridge(Plane2Array));
		return FrameTimeMs;
	};
	return SafeCall(Function, __func__, -99 );
}

__export int32_t PushData(int32_t Instance,uint8_t* Data,int32_t DataSize)
{
	auto Function = [&]()
	{
		auto& Decoder = InstanceManager::GetInstance(Instance);
		return 0;
	};
	return SafeCall(Function, __func__, 0 );
}


__export void GetMeta(int32_t Instance, int32_t* pMetaValues, int32_t MetaValuesCount)
{
	auto Function = [&]()
	{
		auto& Device = InstanceManager::GetInstance(Instance);
		/*
		auto& Meta = Device.GetMeta();
		
		size_t MetaValuesCounter = 0;
		auto MetaValues = GetRemoteArray(pMetaValues, MetaValuesCount, MetaValuesCounter);
		
		BufferArray<SoyPixelsMeta, 3> PlaneMetas;
		Meta.GetPlanes(GetArrayBridge(PlaneMetas));
		MetaValues.PushBack(PlaneMetas.GetSize());
		
		for ( auto p=0;	p<PlaneMetas.GetSize();	p++ )
		{
			auto& PlaneMeta = PlaneMetas[p];
			MetaValues.PushBack(PlaneMeta.GetWidth());
			MetaValues.PushBack(PlaneMeta.GetHeight());
			MetaValues.PushBack(PlaneMeta.GetChannels());
			MetaValues.PushBack(PlaneMeta.GetFormat());
			MetaValues.PushBack(PlaneMeta.GetDataSize());
		}
		*/
		return 0;
	};
	auto x = SafeCall(Function, __func__, 0 );
}
