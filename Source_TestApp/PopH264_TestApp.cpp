#include <iostream>
#include <sstream>
#include "PopH264.h"

#if defined(_MSC_VER)
#define TARGET_WINDOWS
#endif

#if defined(TARGET_WINDOWS)
#include <Windows.h>
#endif

#include <thread>

extern void MakeGreyscalePng(const char* Filename);
extern void CompareGreyscale(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data);
extern void MakeRainbowPng(const char* Filename);
extern void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data);

void DebugPrint(const std::string& Message)
{
#if defined(TARGET_WINDOWS)
	OutputDebugStringA(Message.c_str());
	OutputDebugStringA("\n");
#endif
	std::cout << Message.c_str() << std::endl;
}

typedef void CompareFunc_t(const char* MetaJson,uint8_t* Plane0,uint8_t* Plane1,uint8_t* Plane2);


void DecoderTest(const char* TestDataName,CompareFunc_t* Compare)
{
	uint8_t TestData[7*1024];
	auto TestDataSize = PopH264_GetTestData(TestDataName,TestData,std::size(TestData));
	if ( TestDataSize > std::size(TestData) )
		throw std::runtime_error("Buffer for test data not big enough");
	
	auto Handle = PopH264_CreateInstance(0);

	auto Result = PopH264_PushData( Handle, TestData, TestDataSize, 0 );
	if ( Result < 0 )
		throw std::runtime_error("DecoderTest: PushData error");
	
	//	flush
	PopH264_PushData(Handle,nullptr,0,0);
	
	//	wait for it to decode
	for ( auto i=0;	i<100;	i++ )
	{
		char MetaJson[1000];
		PopH264_PeekFrame( Handle, MetaJson, std::size(MetaJson) );
		uint8_t Plane0[256*256];
		uint8_t Plane1[256*256];
		uint8_t Plane2[256*256];
		auto FrameTime = PopH264_PopFrame(Handle, Plane0, std::size(Plane0), Plane1, std::size(Plane1), Plane2, std::size(Plane2) );
		std::stringstream Error;
		Error << "Decoded testdata; " << MetaJson << " frame=" << FrameTime;
		DebugPrint(Error.str());
		bool IsValid = FrameTime >= 0;
		if ( !IsValid )
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}
		
		Compare( MetaJson, Plane0, Plane1, Plane2 );
		break;
	}
	
	PopH264_DestroyInstance(Handle);
}



int main()
{
	MakeGreyscalePng("PopH264Test_GreyscaleGradient.png");
	MakeRainbowPng("PopH264Test_RainbowGradient.png");

	DebugPrint("PopH264_UnitTests");
	//PopH264_UnitTests();

	try
	{
		DecoderTest("RainbowGradient.h264",CompareRainbow);
	}
	catch (std::exception& e)
	{
		DebugPrint(e.what());
	}

	try
	{
		DecoderTest("GreyscaleGradient.h264",CompareGreyscale);
	}
	catch (std::exception& e)
	{
		DebugPrint(e.what());
	}

	const char* EncoderOptionsJson =
	R"V0G0N(
	{
	}
	)V0G0N";

	//	testing the apple encoder
	char ErrorBuffer[1000] = {0};
	auto Handle = PopH264_CreateEncoder(EncoderOptionsJson, ErrorBuffer, std::size(ErrorBuffer) );
	std::stringstream Debug;
	Debug << "PopH264_CreateEncoder handle=" << Handle << " error=" << ErrorBuffer;
	DebugPrint(Debug.str());
	
	//	encode a test image
	const uint8_t TestImage[128*128]={128};
	const char* TestMetaJson =
	R"V0G0N(
	{
		"Width":128,
		"Height":128,
		"LumaSize":16384
	}
	)V0G0N";
	PopH264_EncoderPushFrame( Handle, TestMetaJson, TestImage, nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer) );
	Debug << "PopH264_EncoderPushFrame error=" << ErrorBuffer;
	DebugPrint(Debug.str());

	PopH264_DestroyEncoder(Handle);
	
	return 0;
}

