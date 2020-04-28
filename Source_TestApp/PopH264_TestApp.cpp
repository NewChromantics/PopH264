#include <iostream>
#include <sstream>
#include "PopH264.h"

#if defined(_MSC_VER)
#define TARGET_WINDOWS
#endif

#if defined(TARGET_WINDOWS)
#include <Windows.h>
#endif

void DebugPrint(const std::string& Message)
{
#if defined(TARGET_WINDOWS)
	OutputDebugStringA(Message.c_str());
	OutputDebugStringA("\n");
#endif
	std::cout << Message.c_str() << std::endl;
}

void DecoderTest()
{
	auto Handle = PopH264_CreateInstance(0);
	PopH264_DestroyInstance(Handle);
}

int main()
{
	DebugPrint("PopH264_UnitTests");
	//PopH264_UnitTests();

	try
	{
		DecoderTest();
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

