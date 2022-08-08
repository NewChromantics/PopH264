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
	~TInstanceManager()				{	FreeInstances();	}
	std::shared_ptr<INSTANCETYPE>	GetInstance(uint32_t Instance);
	uint32_t						AssignInstance(std::shared_ptr<INSTANCETYPE> Object);
	void							FreeInstance(uint32_t Instance);
	uint32_t						CreateInstance(const INSTANCEPARAMS& Params);
	size_t							GetInstanceCount() const	{	return mInstances.GetSize();	}
	void							FreeInstances();
	
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
		//	if the debugger won't step into this from outside, use this in lldb
		//	breakpoint set --name PopH264::TEncoderInstance::TEncoderInstance
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
std::shared_ptr<INSTANCETYPE> TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::GetInstance(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(mInstancesLock);
	auto pInstanceEntry = mInstances.Find(Instance);
	auto pInstance = pInstanceEntry ? pInstanceEntry->mObject : nullptr;
	if ( !pInstance )
	{
		std::stringstream Error;
		Error << "No instance/device matching " << Instance;
		throw Soy::AssertException(Error.str());
	}
	
	return pInstance;
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
	//	pop from instances then free to make sure any other instance-fetching call gets nothing whilst we clean up
	std::shared_ptr<INSTANCETYPE> pInstance;
	{
		std::scoped_lock Lock(mInstancesLock);
		auto InstanceIndex = mInstances.FindIndex(Instance);
		if ( InstanceIndex < 0 )
		{
			std::Debug << "No instance " << Instance << " to free" << std::endl;
			return;
		}
		auto InstanceEntry = mInstances.PopAt(InstanceIndex);
		pInstance = InstanceEntry.mObject;
	}

	//	free outside lock (or if there's another reference, will be freed in that ptr's scope)
	pInstance.reset();
}


template<typename INSTANCETYPE,typename INSTANCEPARAMS>
void TInstanceManager<INSTANCETYPE,INSTANCEPARAMS>::FreeInstances()
{
	std::lock_guard<std::mutex> Lock(mInstancesLock);
	mInstances.Clear(true);
}
