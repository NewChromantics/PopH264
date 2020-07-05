#include <iostream>
#include <sstream>
#include "PopH264.h"
#include "SoyPixels.h"
#include "SoyH264.h"

#if !defined(TARGET_WINDOWS) && defined(_MSC_VER)
#define TARGET_WINDOWS
#endif

#if !defined(TARGET_WINDOWS)
#define TEST_ASSETS
#endif

#if defined(TARGET_WINDOWS)
//#include <Windows.h>
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

int GetFirstNonHeaderH264PacketOffset(const uint8_t* Data, size_t DataSize)
{
	int Start = 0;
	while( Start < DataSize )
	{
		auto RemainingData = GetRemoteArray(&Data[Start], DataSize - Start);
		size_t NaluSize;
		size_t HeaderSize;
		auto PacketStart = FindNaluStartIndex(GetArrayBridge(RemainingData), NaluSize, HeaderSize);
		auto PacketDataStart = PacketStart + NaluSize;
		H264::DecodeNaluByte(SoyMediaFormat::Type Format, const ArrayBridge<uint8>&& Data, H264NaluContent::Type& Content, H264NaluPriority::Type& Priority);	//	throws on error (eg. reservered-zero not zero)



	//	
}

void DecoderTest(const char* TestDataName,CompareFunc_t* Compare,int DecoderMode)
{
	uint8_t TestData[7*1024];
	auto TestDataSize = PopH264_GetTestData(TestDataName,TestData,std::size(TestData));
	if ( TestDataSize > std::size(TestData) )
		throw std::runtime_error("Buffer for test data not big enough");
	
	auto Handle = PopH264_CreateInstance(DecoderMode);

	//	gr; to test robusness, cut out SPS & PPS and send data, then send the complete data
	//		we should still decode.
	{
		auto NonSpsPacketOffset = GetFirstNonHeaderH264PacketOffset(TestData, TestDataSize);
		auto Result = PopH264_PushData(Handle, TestData, TestDataSize, 0);
		if (Result < 0)
			throw std::runtime_error("DecoderTest: PushData without header error");
	}

	auto Result = PopH264_PushData( Handle, TestData, TestDataSize, 0 );
	if ( Result < 0 )
		throw std::runtime_error("DecoderTest: PushData error");
	
	//	gr: did we need to push twice to catch a bug in broadway?
	//PopH264_PushData(Handle, TestData, TestDataSize, 0);
	
	//	flush
	PopH264_PushData(Handle, nullptr, 0, 0);
//	PopH264_PushData(Handle, nullptr, 0, 0);
	
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

void EncoderGreyscaleTest()
{
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
	
	//	todo: decode it again
	
	PopH264_DestroyEncoder(Handle);
}


void EncoderYuv8_88Test()
{
	const char* EncoderOptionsJson =
	R"V0G0N(
	{
	}
	)V0G0N";
	
	char ErrorBuffer[1000] = {0};
	auto Handle = PopH264_CreateEncoder(EncoderOptionsJson, ErrorBuffer, std::size(ErrorBuffer) );
	std::stringstream Debug;
	Debug << "PopH264_CreateEncoder handle=" << Handle << " error=" << ErrorBuffer;
	DebugPrint(Debug.str());

	SoyPixels Yuv( SoyPixelsMeta(640,480,SoyPixelsFormat::Yuv_8_88));
	auto Size = Yuv.GetPixelsArray().GetDataSize();
	const char* TestMetaJson =
	R"V0G0N(
	{
		"Width":640,
		"Height":480,
		"LumaSize":460800,
		"Format":"Yuv_8_88"
	}
	)V0G0N";
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame( Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer) );
	Debug << "PopH264_EncoderPushFrame error=" << ErrorBuffer;
	DebugPrint(Debug.str());
	
	PopH264_EncoderEndOfStream(Handle);
	
	//	todo: decode it again
	while(true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		uint8_t PacketBuffer[1024*50];
		auto FrameSize = PopH264_EncoderPopData(Handle, PacketBuffer, std::size(PacketBuffer) );
		if ( FrameSize < 0 )
			break;
		std::Debug << "Encoder packet: x" << FrameSize << std::endl;
	}
	
	PopH264_DestroyEncoder(Handle);
}
	
int main()
{
	EncoderYuv8_88Test();

#if defined(TEST_ASSETS)
	MakeGreyscalePng("PopH264Test_GreyscaleGradient.png");
	MakeRainbowPng("PopH264Test_RainbowGradient.png");
#endif

	DebugPrint("PopH264_UnitTests");
	//PopH264_UnitTests();

	try
	{
		DecoderTest("RainbowGradient.h264", CompareRainbow, POPH264_DECODERMODE_SOFTWARE);
		DecoderTest("RainbowGradient.h264", CompareRainbow, POPH264_DECODERMODE_HARDWARE);
	}
	catch (std::exception& e)
	{
		DebugPrint(e.what());
	}

	return 0;

	try
	{
#if defined(TEST_ASSETS)
		DecoderTest("GreyscaleGradient.h264", CompareGreyscale, POPH264_DECODERMODE_SOFTWARE);
		DecoderTest("GreyscaleGradient.h264", CompareGreyscale, POPH264_DECODERMODE_HARDWARE);
#endif
	}
	catch (std::exception& e)
	{
		DebugPrint(e.what());
	}

	EncoderGreyscaleTest();
	
	return 0;
}

#if !defined(TEST_ASSETS)
void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data)
{
}
#endif
