#include <iostream>
#include <sstream>
#include "PopH264.h"
#include "SoyPixels.h"

#if !defined(TARGET_WINDOWS) && defined(_MSC_VER)
#define TARGET_WINDOWS
#endif

#if !defined(TARGET_WINDOWS) && !defined(TARGET_ANDROID) && !defined(TARGET_IOS)
#define TEST_ASSETS
#endif

#if defined(TARGET_WINDOWS)
//#include <Windows.h>
#endif


#if defined(TARGET_WINDOWS)//||defined(TARGET_LINUX)||defined(TARGET_ANDROID)
//	instead of building SoyFilesystem.cpp
namespace Platform
{
	std::string	GetAppResourcesDirectory();
}
std::string Platform::GetAppResourcesDirectory()
{
	return "";
}
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


//	fopen_s is a ms specific "safe" func, so provide an alternative
#if !defined(TARGET_WINDOWS)
int fopen_s(FILE **f, const char *name, const char *mode) 
{
	assert(f);
	*f = fopen(name, mode);
	//	Can't be sure about 1-to-1 mapping of errno and MS' errno_t
	if (!*f)
		return errno;
	return 0;
}
#endif

#include "SoyFilesystem.h"

bool LoadDataFromFilename(const char* DataFilename,ArrayBridge<uint8_t>&& Data)
{
	std::stringstream FilePath;
	FilePath << Platform::GetAppResourcesDirectory() << DataFilename;

	auto Filename = FilePath.str();

	FILE* File = nullptr;
	auto Error = fopen_s(&File,Filename.c_str(), "rb");
	if (!File)
		return false;
	fseek(File, 0, SEEK_SET);
	while (!feof(File))
	{
		uint8_t Buffer[1024 * 100];
		auto BytesRead = fread(Buffer, 1, sizeof(Buffer), File);
		auto BufferArray = GetRemoteArray(Buffer, BytesRead, BytesRead);
		Data.PushBackArray(BufferArray);
	}
	fclose(File);
	return true;
}

//	gr: 1mb too big for windows on stack
uint8_t TestDataBuffer[1 * 1024 * 1024];

void DecoderTest(const char* TestDataName,CompareFunc_t* Compare,const char* DecoderName,size_t DataRepeat=1)
{
	std::Debug << "DecoderTest(" << (TestDataName?TestDataName:"<null>") << "," << (DecoderName?DecoderName:"<null>") << ")" << std::endl;
	Array<uint8_t> TestData;

	if (!LoadDataFromFilename(TestDataName, GetArrayBridge(TestData)))
	{
		//	gr: using int (auto) here, causes some resolve problem with GetRemoteArray below
		size_t TestDataSize = PopH264_GetTestData(TestDataName, TestDataBuffer, std::size(TestDataBuffer));
		if ( TestDataSize < 0 )
			throw std::runtime_error("Missing test data");
		if ( TestDataSize == 0 )
			throw std::runtime_error("PopH264_GetTestData unexpectedly returned zero-length test data");
		if (TestDataSize > std::size(TestDataBuffer))
		{
			std::stringstream Debug;
			Debug << "Buffer for test data (" << TestDataSize << ") not big enough";
			throw std::runtime_error(Debug.str());
		}
		//	gr: debug here as on Android GetRemoteArray with TestDataSize=auto was making a remote array of zero bytes
		//std::Debug << "making TestDataArray..." << std::endl;
		auto TestDataArray = GetRemoteArray(TestDataBuffer, TestDataSize, TestDataSize);
		//std::Debug << "TestDataSize=" << TestDataSize << " TestDataArray.GetSize=" << TestDataArray.GetDataSize() << std::endl;
		TestData.PushBackArray(TestDataArray);
		//std::Debug << "TestData.PushBackArray() " << TestData.GetDataSize() << std::endl;
	}

	std::stringstream OptionsStr;
	OptionsStr << "{";
	if ( DecoderName )
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":true";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options,ErrorBuffer,std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName?TestDataName:"<null>") << ") Size: " << TestData.GetDataSize() << std::endl;
	
	int FirstFrameNumber = 9999 - 100;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.GetArray(), TestData.GetDataSize(), FirstFrameNumber);
		if (Result < 0)
			throw std::runtime_error("DecoderTest: PushData error");

		//	gr: did we need to push twice to catch a bug in broadway?
		//PopH264_PushData(Handle, TestData, TestDataSize, 0);

		//	flush
		if (LastIteration)
			PopH264_PushEndOfStream(Handle);

		//	wait for it to decode
		for (auto i = 0; i < 100; i++)
		{
			char MetaJson[1000];
			PopH264_PeekFrame(Handle, MetaJson, std::size(MetaJson));
			static uint8_t Plane0[1024 * 1024];
			static uint8_t Plane1[1024 * 1024];
			static uint8_t Plane2[1024 * 1024];
			auto FrameNumber = PopH264_PopFrame(Handle, Plane0, std::size(Plane0), Plane1, std::size(Plane1), Plane2, std::size(Plane2));
			std::stringstream Error;
			Error << "Decoded testdata; " << MetaJson << " FrameNumber=" << FrameNumber << " Should be " << FirstFrameNumber;
			DebugPrint(Error.str());
			bool IsValid = FrameNumber >= 0;
			if (!IsValid)
			{
				//std::this_thread::sleep_for(std::chrono::milliseconds(500));
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				continue;
			}

			if (FrameNumber != FirstFrameNumber)
				throw std::runtime_error("Wrong frame number from decoder");

			if (Compare)
				Compare(MetaJson, Plane0, Plane1, Plane2);
			break;
		}
	}

	PopH264_DestroyInstance(Handle);
}



void LoadTestData(Array<uint8_t>& TestData,const char* TestDataName)
{
	if ( LoadDataFromFilename(TestDataName, GetArrayBridge(TestData)) )
		return;
		
	//	gr: using int (auto) here, causes some resolve problem with GetRemoteArray below
	size_t TestDataSize = PopH264_GetTestData(TestDataName, TestDataBuffer, std::size(TestDataBuffer));
	if ( TestDataSize < 0 )
		throw std::runtime_error("Missing test data");
	if ( TestDataSize == 0 )
		throw std::runtime_error("PopH264_GetTestData unexpectedly returned zero-length test data");
	if (TestDataSize > std::size(TestDataBuffer))
	{
		std::stringstream Debug;
		Debug << "Buffer for test data (" << TestDataSize << ") not big enough";
		throw std::runtime_error(Debug.str());
	}
	//	gr: debug here as on Android GetRemoteArray with TestDataSize=auto was making a remote array of zero bytes
	//std::Debug << "making TestDataArray..." << std::endl;
	auto TestDataArray = GetRemoteArray(TestDataBuffer, TestDataSize, TestDataSize);
	//std::Debug << "TestDataSize=" << TestDataSize << " TestDataArray.GetSize=" << TestDataArray.GetDataSize() << std::endl;
	TestData.PushBackArray(TestDataArray);
	//std::Debug << "TestData.PushBackArray() " << TestData.GetDataSize() << std::endl;
}

//	give the decoder loads of data to decode, then try and destroy the decoder
//	whilst its still decoding (to crash android)
void DestroyMidDecodeTest(const char* TestDataName,CompareFunc_t* Compare,const char* DecoderName)
{
	std::Debug << __PRETTY_FUNCTION__ << "(" << (TestDataName?TestDataName:"<null>") << "," << (DecoderName?DecoderName:"<null>") << ")" << std::endl;
	Array<uint8_t> TestData;
	LoadTestData(TestData,TestDataName);

	std::stringstream OptionsStr;
	OptionsStr << "{";
	if ( DecoderName )
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":true";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options,ErrorBuffer,std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName?TestDataName:"<null>") << ") Size: " << TestData.GetDataSize() << std::endl;
	
	size_t DataRepeat = 500;
	int FirstFrameNumber = 9999 - 100;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.GetArray(), TestData.GetDataSize(), FirstFrameNumber);
		if (Result < 0)
			throw std::runtime_error("DecoderTest: PushData error");

		if ( Iteration == 100 )
		{
		/*
			std::Debug << __PRETTY_FUNCTION__ << " PopH264_DestroyInstance(" << Handle << ")" << std::endl;
			PopH264_DestroyInstance(Handle);
			*/
		}
/*
		if (LastIteration)
			PopH264_PushEndOfStream(Handle);
			*/
	}

	//	hoping we push so fast that its still decoding here
	std::Debug << __PRETTY_FUNCTION__ << " PopH264_DestroyInstance(" << Handle << ")" << std::endl;
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


void EncoderYuv8_88Test(int Width,int Height,const char* EncoderName="")
{
	std::stringstream EncoderOptionsJson;
	EncoderOptionsJson << "{\n";
	EncoderOptionsJson << "	\"Encoder\":\"" << EncoderName << "\"	";
	EncoderOptionsJson << "}";
	DebugPrint(std::string("Encoder options: ") + EncoderOptionsJson.str());
	
	char ErrorBuffer[1000] = {0};
	auto Handle = PopH264_CreateEncoder(EncoderOptionsJson.str().c_str(), ErrorBuffer, std::size(ErrorBuffer) );
	std::stringstream Debug;
	Debug << "PopH264_CreateEncoder EncoderName=" << EncoderName << " handle=" << Handle << " error=" << ErrorBuffer;
	DebugPrint(Debug.str());

	SoyPixels Yuv( SoyPixelsMeta(Width,Height,SoyPixelsFormat::Yuv_8_88));
	auto Size = Yuv.GetPixelsArray().GetDataSize();
	std::stringstream TestMetaJsonStr;
	TestMetaJsonStr << "{";
	TestMetaJsonStr << "\"Width\":" << Width << ",";
	TestMetaJsonStr << "\"Height\":" << Height << ",";
	TestMetaJsonStr << "\"LumaSize\":" << Yuv.GetMeta().GetDataSize() << ",";
	TestMetaJsonStr << "\"Format\":\"" << SoyPixelsFormat::Yuv_8_88 << "\",";
	TestMetaJsonStr << "\"TestMeta\":\"PurpleMonkeyDishwasher\"";
	TestMetaJsonStr << "}";
	auto TestMetaJsons = TestMetaJsonStr.str();
	const char* TestMetaJson = TestMetaJsons.c_str();//	unsafe!

	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer));
	PopH264_EncoderPushFrame( Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, std::size(ErrorBuffer) );
	
	if ( strlen(ErrorBuffer) )
	{
		Debug << "PopH264_EncoderPushFrame error=" << ErrorBuffer;
		DebugPrint(Debug.str());
	}

	PopH264_EncoderEndOfStream(Handle);
	
	//	todo: decode it again
	int MaxErrors = 100;
	while(true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	
		//	read meta first to make sure test data is propogated out again
		char FrameMetaJson[1024] = { 0 };
		PopH264_EncoderPeekData(Handle, FrameMetaJson, std::size(FrameMetaJson));
		std::Debug << "PopH264_EncoderPeekData meta: " << FrameMetaJson << std::endl;
		//	check for test data
		{
			auto TestString = "PurpleMonkeyDishwasher";
			auto FoundPos = std::string(FrameMetaJson).find(TestString);
			if (FoundPos == std::string::npos)
			{
				std::Debug << "Test string missing from meta " << TestString << std::endl;
			}
		}

		uint8_t PacketBuffer[1024*50];
		auto FrameSize = PopH264_EncoderPopData(Handle, PacketBuffer, std::size(PacketBuffer) );
		if ( FrameSize < 0 )
		{
			//	gr: try a few times in case data isnt ready yet.
			if ( MaxErrors-- < 0 )
			{
				DebugPrint("Re-decode, too many errors");
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		std::Debug << "Encoder packet: x" << FrameSize << std::endl;
	}
	
	PopH264_DestroyEncoder(Handle);
}

void SafeDecoderTest(const char* TestDataName,CompareFunc_t* Compare,const char* DeocoderName=nullptr)
{
	try
	{
		DecoderTest(TestDataName, Compare, DeocoderName );
	}
	catch (std::exception& e)
	{
		DebugPrint(e.what());
	}
}


int main()
{
	//if ( false )
	{
		EncoderGreyscaleTest();
	}

	if ( false )
	{
		EncoderYuv8_88Test(1280,480);
	}

	if ( false )
	{
		//	trying to crash android
		for ( auto d=0;	d<300;	d++)
		{
			std::Debug << "DestroyMidDecodeTest #" << d << std::endl;
			DestroyMidDecodeTest("RainbowGradient.h264", nullptr, nullptr);
		}
	}
	
	if ( false )
	{
		// heavy duty test to find leaks
		for ( auto d=0;	d<10;	d++)
			DecoderTest("RainbowGradient.h264", nullptr, nullptr, 500);
	}

	std::cout << "main" << std::endl;
	
	//EncoderYuv8_88Test("");

#if defined(TEST_ASSETS)
	MakeGreyscalePng("PopH264Test_GreyscaleGradient.png");
	MakeRainbowPng("PopH264Test_RainbowGradient.png");
#endif

	DebugPrint("PopH264_UnitTests");
	PopH264_UnitTest(nullptr);
	
	//	depth data has iframe, pps, sps order
	SafeDecoderTest("TestData/Main5.h264", nullptr, nullptr );
	SafeDecoderTest("TestData/Colour.h264", nullptr, nullptr );
	SafeDecoderTest("TestData/Depth.h264", nullptr, nullptr );
	SafeDecoderTest("TestData/Depth.h264", nullptr, "Broadway" );
	SafeDecoderTest("RainbowGradient.h264", CompareRainbow, nullptr );
	SafeDecoderTest("RainbowGradient.h264", CompareRainbow, "Broadway" );
	SafeDecoderTest("../TestData/Colour.h264", nullptr, nullptr );
	SafeDecoderTest("../TestData/Colour.h264", nullptr, "Broadway" );
	//SafeDecoderTest("RainbowGradient.h264", CompareRainbow);
	//SafeDecoderTest("RainbowGradient.h264",CompareRainbow);
	
	return 0;
	
#if defined(TEST_ASSETS)
	SafeDecoderTest("GreyscaleGradient.h264",CompareGreyscale);
#endif

	EncoderGreyscaleTest();
	
	return 0;
}

#if !defined(TEST_ASSETS)
void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data)
{
}
#endif

