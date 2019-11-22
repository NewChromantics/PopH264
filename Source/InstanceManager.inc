#include <memory>
#include <mutex>
#include "SoyLib/src/HeapArray.hpp"


class TNoParams
{
};


//	define your own type
//using TInstanceObject = Broadway::TDecoder;
//using TInstanceParams = TNoParams;
class TInstance;

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
		auto Object = std::make_shared<TInstanceObject>( Params.Mode );
		if ( Object )
			return InstanceManager::AssignInstance(Object);
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
		throw;
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



__export int32_t PopH264_CreateInstance(int32_t Mode)
{
	auto Function = [&]()
	{
		TInstanceParams Params(Mode);
		auto InstanceId = InstanceManager::CreateInstance( Params );
		return InstanceId;
	};
	return SafeCall( Function, __func__, -1 );
}

__export void PopH264_DestroyInstance(int32_t Instance)
{
	auto Function = [&]()
	{
		InstanceManager::FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export TInstanceObject* PopH264_GetInstancePtr(int32_t Instance)
{
	auto Function = [&]()
	{
		auto& Object = InstanceManager::GetInstance(Instance);
		return &Object;
	};
	return SafeCall<TInstanceObject*>( Function, __func__, nullptr );
}
