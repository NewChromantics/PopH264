#pragma once



//	INSTANCETYPE should derive from TInstance
template<typename INSTANCETYPE,typename INSTANCEPARAMS>
class TInstanceManager
{
private:
	class TInstance
	{
	public:
		std::shared_ptr<INSTANCETYPE>	mObject;
		uint32_t						mInstanceId = 0;
		
		bool							operator==(const uint32_t& InstanceId) const	{	return mInstanceId == InstanceId;	}
	};
	
public:
	INSTANCETYPE&		GetInstance(uint32_t Instance);
	uint32_t			AssignInstance(std::shared_ptr<INSTANCETYPE> Object);
	void				FreeInstance(uint32_t Instance);
	uint32_t			CreateInstance(const INSTANCEPARAMS& Params);
	
private:
	std::mutex			mInstancesLock;
	Array<TInstance>	mInstances;
	uint32_t			mInstancesCounter = 1;
};



template<typename INSTANCETYPE,typename INSTANCEPARAMS>
uint32_t TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::CreateInstance(const INSTANCEPARAMS& Params)
{
	//	alloc device
	try
	{
		auto Object = std::make_shared<INSTANCETYPE>( Params );
		if ( Object )
			return AssignInstance(Object);
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
		throw;
	}
	
	throw Soy::AssertException("Failed to create instance");
}


template<typename INSTANCETYPE,typename INSTANCEPARAMS>
INSTANCETYPE& TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::GetInstance(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(mInstancesLock);
	auto pInstance = mInstances.Find(Instance);
	auto* Device = pInstance ? pInstance->mObject.get() : nullptr;
	if ( !Device )
	{
		std::stringstream Error;
		Error << "No instance/device matching " << Instance;
		throw Soy::AssertException(Error.str());
	}
	
	return *Device;
}

template<typename INSTANCETYPE,typename INSTANCEPARAMS>
uint32_t TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::AssignInstance(std::shared_ptr<INSTANCETYPE> Object)
{
	std::lock_guard<std::mutex> Lock(mInstancesLock);
	
	TInstance Instance;
	Instance.mInstanceId = mInstancesCounter;
	Instance.mObject = Object;
	mInstances.PushBack(Instance);
	
	mInstancesCounter++;
	return Instance.mInstanceId;
}


template<typename INSTANCETYPE,typename INSTANCEPARAMS>
void TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::FreeInstance(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(mInstancesLock);
	
	auto InstanceIndex = mInstances.FindIndex(Instance);
	if ( InstanceIndex < 0 )
	{
		std::Debug << "No instance " << Instance << " to free" << std::endl;
		return;
	}
	
	mInstances.RemoveBlock(InstanceIndex, 1);
}


