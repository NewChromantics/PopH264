#include "PopCameraDevice.hpp"
#include <exception>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include "TStringBuffer.hpp"
#include "SoyLib\src\HeapArray.hpp"
#include "TestDevice.hpp"
#include "MfCapture.h"

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

class TDeviceInstance;

namespace PopCameraDevice
{
	TCameraDevice&	GetCameraDevice(uint32_t Instance);
	uint32_t		CreateInstance(std::shared_ptr<TCameraDevice> Device);
	void			FreeInstance(uint32_t Instance);
	bool			PopFrame(TCameraDevice& Device, ArrayBridge<uint8_t>&& Plane0, ArrayBridge<uint8_t>&& Plane1, ArrayBridge<uint8_t>&& Plane2);

	uint32_t		CreateCameraDevice(const std::string& Name);

	std::mutex				InstancesLock;
	Array<TDeviceInstance>	Instances;
	uint32_t				InstancesCounter = 1;
}


class TDeviceInstance
{
public:
	std::shared_ptr<TCameraDevice>	mDevice;
	uint32_t						mInstanceId = 0;

	bool							operator==(const uint32_t& InstanceId) const	{	return mInstanceId == InstanceId;	}
};


class TWriteContext
{
public:
	TWriteContext(uint8_t* JpegData,size_t JpegDataSize) :
		mJpegData		( JpegData ),
		mJpegDataSize	( JpegDataSize ),
		mDataWritten	( 0 )
	{
	}
	
	void		Write(const uint8_t* Data,size_t Size);

public:
	uint8_t*	mJpegData;
	size_t		mJpegDataSize;
	size_t		mDataWritten;
};


namespace PopEncodeJpeg
{
	std::shared_ptr<TStringBuffer>	gDebugStrings;
	TStringBuffer&					GetDebugStrings();
}


template<typename STRING>
void DebugLog(const STRING& String)
{
	auto& DebugStrings = PopEncodeJpeg::GetDebugStrings();
	DebugStrings.Push( String );
}


__export const char* PopDebugString()
{
	try
	{
		auto& DebugStrings = PopEncodeJpeg::GetDebugStrings();
		return DebugStrings.Pop();
	}
	catch(...)
	{
		//	bit recursive if we push one?
		return nullptr;
	}
}

__export void ReleaseDebugString(const char* String)
{
	try
	{
		auto& DebugStrings = PopEncodeJpeg::GetDebugStrings();
		DebugStrings.Release( String );
	}
	catch(...)
	{
	}
}


__export void EnumCameraDevices(char* StringBuffer,int32_t StringBufferLength)
{
	//	first char is delin
	const char PossibleDelin[] = ",;:#!@+=_-&^%*$£?|/ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopoqrstuvwzy0123456789";
	auto PossibleDelinArray = GetRemoteArray( PossibleDelin );

	Array<std::string> DeviceNames;
	auto EnumDevice = [&](const std::string& Name)
	{
		DeviceNames.PushBack(Name);
	};
	TestDevice::EnumDeviceNames(EnumDevice);
	MediaFoundation::EnumCaptureDevices(EnumDevice);


	auto IsCharUsed = [&](char Char)
	{
		for ( int d=0;	d<DeviceNames.GetSize();	d++ )
		{
			auto& DeviceName = DeviceNames[d];
			auto Index = DeviceName.find_first_of(Char);
			if ( Index != DeviceName.npos )
				return true;
		}
		return false;
	};

	char Delin;
	for ( auto pd=0;	pd<PossibleDelinArray.GetSize();	pd++ )
	{
		Delin = PossibleDelin[pd];
		bool Used = IsCharUsed(Delin);
		if ( !Used )
			break;		
	}
	//	todo! handle no unused chars!

	//	build output
	//	first char is delin, then each device seperated by that delin
	std::stringstream OutputString;
	OutputString << Delin;
	for ( int d=0;	d<DeviceNames.GetSize();	d++ )
	{
		auto& DeviceName = DeviceNames[d];
		OutputString << DeviceName << Delin;
	}

	auto OutputStringStr = OutputString.str();
	Soy::StringToBuffer( OutputStringStr, StringBuffer, StringBufferLength );
}



uint32_t PopCameraDevice::CreateCameraDevice(const std::string& Name)
{
	//	alloc device
	try
	{
		auto Device = TestDevice::CreateDevice(Name);
		if ( Device )
			return PopCameraDevice::CreateInstance(Device);
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}


	try
	{
		std::shared_ptr<TCameraDevice> Device(new MediaFoundation::TCamera(Name));
		if ( Device )
			return PopCameraDevice::CreateInstance(Device);
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}


	throw Soy::AssertException("Failed to create device");
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

__export int32_t CreateCameraDevice(const char* Name)
{
	auto Function = [&]()
	{
		auto InstanceId = PopCameraDevice::CreateCameraDevice( Name );
		return InstanceId;
	};
	return SafeCall( Function, __func__, -1 );
}



TStringBuffer& PopEncodeJpeg::GetDebugStrings()
{
	if ( !gDebugStrings )
	{
		gDebugStrings.reset( new TStringBuffer() );
	}
	return *gDebugStrings;
}



void TWriteContext::Write(const uint8_t* Data,size_t Size)
{
	//	gr: keep going so we know how much we need
	/*
	if ( mDataWritten + Size > mJpegDataSize )
	{
		std::stringstream Error;
		Error << "Jpeg buffer size not big enough, trying to write " << (mDataWritten + Size) << "/" << mJpegDataSize;
		throw std::runtime_error::runtime_error( Error.str() );
	}
	*/
	
	auto SpaceRemaining = mJpegDataSize - mDataWritten;
	auto WriteLength = std::min( SpaceRemaining, Size );
	if ( WriteLength > 0 )
	{
		auto* Dst = &mJpegData[mDataWritten];
		auto* Src = Data;
		memcpy( Dst, Src, WriteLength );
	}
	
	mDataWritten += Size;
}


TCameraDevice& PopCameraDevice::GetCameraDevice(uint32_t Instance)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	auto pInstance = Instances.Find(Instance);
	auto* Device = pInstance ? pInstance->mDevice.get() : nullptr;
	if ( !Device )
	{
		std::stringstream Error;
		Error << "No instance/device matching " << Instance;
		throw Soy::AssertException(Error.str());
	}

	return *Device;
}


uint32_t PopCameraDevice::CreateInstance(std::shared_ptr<TCameraDevice> Device)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	
	TDeviceInstance Instance;
	Instance.mInstanceId = InstancesCounter;
	Instance.mDevice = Device;
	Instances.PushBack(Instance);

	InstancesCounter++;
	return Instance.mInstanceId;
}


void PopCameraDevice::FreeInstance(uint32_t Instance)
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


__export void GetMeta(int32_t Instance, int32_t* pMetaValues, int32_t MetaValuesCount)
{
	auto Function = [&]()
	{
		auto& Device = PopCameraDevice::GetCameraDevice(Instance);
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

		return 0;
	};
	auto x = SafeCall(Function, __func__, 0 );
}

__export void FreeCameraDevice(int32_t Instance)
{
	auto Function = [&]()
	{
		PopCameraDevice::FreeInstance(Instance);
		return 0;
	};
	auto x = SafeCall(Function, __func__, 0 );
}


bool PopCameraDevice::PopFrame(TCameraDevice& Device,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2)
{
	if ( !Device.PopLastFrame(Plane0, Plane1, Plane2) )
		return false;

	return true;
}

__export int32_t PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size)
{
	auto Function = [&]()
	{
		auto& Device = PopCameraDevice::GetCameraDevice(Instance);
		auto Plane0Array = GetRemoteArray(Plane0, Plane0Size);
		auto Plane1Array = GetRemoteArray(Plane1, Plane1Size);
		auto Plane2Array = GetRemoteArray(Plane2, Plane2Size);
		auto Result = PopCameraDevice::PopFrame(Device, GetArrayBridge(Plane0Array), GetArrayBridge(Plane1Array), GetArrayBridge(Plane2Array));
		return Result ? 1 : 0;
	};
	return SafeCall(Function, __func__, 0 );
}

