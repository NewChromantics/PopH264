#include "PopCameraDevice.hpp"
#include <exception>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include "TStringBuffer.hpp"
#include "SoyLib\src\HeapArray.hpp"
#include "TestDevice.hpp"


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

	uint32_t		CreateCameraDevice(const std::string& Name);

	std::mutex								InstancesLock;
	Array<std::shared_ptr<TDeviceInstance>>	Instances;
	uint32_t								InstancesCounter = 1;
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


	throw Soy::AssertException("Failed to create device");
}


__export int32_t CreateCameraDevice(const char* Name)
{
	try
	{
		auto InstanceId = PopCameraDevice::CreateCameraDevice( Name );

		return InstanceId;
	}
	catch(std::exception& e)
	{
		std::Debug << __func__ << " exception: " << e.what() << std::endl;
		return -1;
	}
	catch(...)
	{
		std::Debug << __func__ << " unknown exception." << std::endl;
		return -1;
	}


}
__export void				FreeCameraDevice(int32_t Instance);
__export void				GetMeta(int32_t Instance,int32_t* MetaValues,int32_t MetaValuesCount);
__export int32_t			PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);



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


uint32_t PopCameraDevice::CreateInstance(std::shared_ptr<TCameraDevice> Device)
{
	std::lock_guard<std::mutex> Lock(InstancesLock);
	
	std::shared_ptr<TDeviceInstance> pInstance(new TDeviceInstance);
	auto& Instance = *pInstance;
	Instance.mInstanceId = InstancesCounter;
	Instance.mDevice = Device;
	Instances.PushBack(pInstance);
	InstancesCounter++;
	return Instance.mInstanceId;
}

