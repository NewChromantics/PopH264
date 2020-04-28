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
	uint8_t TestData[1000];
	auto TestDataSize = PopH264_GetTestData("GreyscaleGradient.h264",TestData,std::size(TestData));
	
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
		uint8_t Plane0[256*100];
		auto FrameTime = PopH264_PopFrame(Handle, Plane0, std::size(Plane0), nullptr, 0, nullptr, 0 );
		std::stringstream Error;
		Error << "Decoded testdata; " << MetaJson << " frame=" << FrameTime;
		DebugPrint(Error.str());
		bool IsValid = FrameTime >= 0;
		if ( !IsValid )
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}
		
		//	debug first column (for the greyscale test image)
		{
			std::stringstream Debug;
			auto Width = 16;
			for ( auto y=0;	y<256;	y++ )
			{
				auto i = Width * y;
				auto l = Plane0[i];
				Debug << static_cast<int>(l) << ' ';
			}
			DebugPrint(Debug.str());
		}
		
		break;
	}
	
	PopH264_DestroyInstance(Handle);
}

#include "SoyPixels.h"
#include "SoyPng.h"
#include "SoyFileSystem.h"
void MakeGreyscalePng(const char* Filename)
{
	SoyPixels Pixels(SoyPixelsMeta(10,256,SoyPixelsFormat::RGB));
	auto Components = Pixels.GetMeta().GetChannels();
	auto& PixelsArray = Pixels.GetPixelsArray();
	for ( auto y=0;	y<Pixels.GetHeight();	y++ )
	{
		for ( auto x=0;	x<Pixels.GetWidth();	x++ )
		{
			auto i = y*Pixels.GetWidth();
			i += x;
			i *= Components;
			PixelsArray[i+0] = y;
			PixelsArray[i+1] = y;
			PixelsArray[i+2] = y;
		}
	}
	Array<uint8_t> PngData;
	TPng::GetPng( Pixels, GetArrayBridge(PngData), 0 );
	Soy::ArrayToFile( GetArrayBridge(PngData), Filename);
	Platform::ShowFileExplorer(Filename);
}


int main()
{
	MakeGreyscalePng("PopH264Test_GreyscaleGradient.png");
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

