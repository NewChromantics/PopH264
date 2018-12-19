#include "PopEncodeJpeg.hpp"
#include <exception>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include "TStringBuffer.hpp"
#include "SoyLib\src\HeapArray.hpp"



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
	DeviceNames.PushBack("Test,");

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


