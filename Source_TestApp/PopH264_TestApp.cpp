#include <iostream>
#include <sstream>
#include "PopH264.h"
#include "SoyPixels.h"
#include "SoyH264.h"
#include <thread>

#include "gtest/gtest.h"
#if defined(TARGET_OSX)
#if GTEST_HAS_FILE_SYSTEM
#error This build will error in sandbox mac apps
#endif
#endif

#include "PopJsonCpp/PopJson.hpp"

#if !defined(TARGET_WINDOWS) && defined(_MSC_VER)
#define TARGET_WINDOWS
#endif

#if !defined(TARGET_WINDOWS) && !defined(TARGET_ANDROID) && !defined(TARGET_IOS)
//#define TEST_ASSETS
#endif

#if defined(TARGET_WINDOWS)
//#include <Windows.h>
#endif

namespace Platform
{
	std::string	GetAppResourcesDirectory();

	void		CaptureStdErr();
	void		DebugLog(const char* text);
}


#if defined(TARGET_WINDOWS)//||defined(TARGET_LINUX)||defined(TARGET_ANDROID)
//	instead of building SoyFilesystem.cpp
std::string Platform::GetAppResourcesDirectory()
{
	return "";
}
#endif

template<typename ARRAY,typename MATCHTYPE>
int FindIndex(const ARRAY& Array, const MATCHTYPE& Match)
{
	for ( auto i=0;	i<Array.size();	i++ )
	{
		auto& Element = Array[i];
		if ( Element != Match )
			continue;
		return i;
	}
	return -1;
}

extern void MakeGreyscalePng(const char* Filename);
extern void CompareGreyscale(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data);
extern void MakeRainbowPng(const char* Filename);
extern void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data);



template <class CharT, class TraitsT = std::char_traits<CharT> >
class basic_debugbuf :
	public std::basic_stringbuf<CharT, TraitsT>
{
public:

	virtual ~basic_debugbuf()
	{
		//	not on all platforms
		//sync();
	}

protected:

	std::mutex	mSyncLock;
	std::string	mBuffer;


	int overflow(int c) override
	{
		std::lock_guard<std::mutex> Lock(mSyncLock);
		mBuffer += (char)c;

		if (c == '\n')
		{
			//flush();
			Platform::DebugLog(mBuffer.c_str());
			//mBuffer = std::string();
			mBuffer.clear();
		}
		//	gr: what is -1? std::eof?
		return c == -1 ? -1 : ' ';
	}
};


basic_debugbuf<char> OutputBuf;


void Platform::CaptureStdErr()
{
	//if (!IsDebuggerPresent())
	//	return;
	std::cerr.rdbuf(&OutputBuf);
}


#if defined(TARGET_WINDOWS)
void Platform::DebugLog(const char* text)
{
	::OutputDebugStringA(text);
}
#else
void Platform::DebugLog(const char* text)
{
	printf("%s\n",text);
}
#endif

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

std::vector<uint8_t> LoadFile(const std::string& Filename)
{
	FILE* File = nullptr;
	auto Error = fopen_s(&File,Filename.c_str(), "rb");
	if ( !File )
		throw std::runtime_error( std::string("Failed to open ") + Filename );

	std::vector<uint8_t> FileContents;
	std::vector<uint8_t> Buffer(1*1024*1024);
	fseek(File, 0, SEEK_SET);
	while (!feof(File))
	{
		auto BytesRead = fread( Buffer.data(), 1, Buffer.size(), File );
		auto DataRead = std::span( Buffer.data(), BytesRead );
		std::copy( DataRead.begin(), DataRead.end(), std::back_inserter(FileContents ) );
	}
	fclose(File);
	return FileContents;
}

std::vector<uint8_t> LoadDataFromFilename(std::string_view DataFilename)
{
	//	change this to detect absolute paths rather than just trying random combinations
	std::vector<std::string> TryFilenames;
	
	{
		std::stringstream FilePath;
		FilePath << Platform::GetAppResourcesDirectory() << DataFilename;
		TryFilenames.push_back(FilePath.str());
	}
	//	in case absolute path
	TryFilenames.push_back( std::string(DataFilename) );

	for ( auto& Filename : TryFilenames )
	{
		try
		{
			auto Data = LoadFile( Filename );
			return Data;
		}
		catch(std::exception& e)
		{
		}
	}
	
	//	load from api
	std::vector<uint8_t> Buffer;
	Buffer.resize( 1 * 1024 * 1024 );
	std::string DataFilenameWithTerminator(DataFilename);
	
	auto TestDataSize = PopH264_GetTestData( DataFilenameWithTerminator.c_str(), Buffer.data(), Buffer.size() );
	if ( TestDataSize < 0 )
		throw std::runtime_error("Missing test data");
	if ( TestDataSize == 0 )
		throw std::runtime_error("PopH264_GetTestData unexpectedly returned zero-length test data");
	if ( TestDataSize > Buffer.size() )
	{
		std::stringstream Debug;
		Debug << "Buffer (" << Buffer.size() << " bytes) for test data (" << TestDataSize << ") not big enough";
		throw std::runtime_error(Debug.str());
	}

	Buffer.resize(TestDataSize);
	return Buffer;
}


void DecoderTest(const char* TestDataName,CompareFunc_t* Compare,const char* DecoderName,size_t DataRepeat=1)
{
	std::Debug << "DecoderTest(" << (TestDataName?TestDataName:"<null>") << "," << (DecoderName?DecoderName:"<null>") << ")" << std::endl;

	auto TestData = LoadDataFromFilename( TestDataName );

	std::stringstream OptionsStr;
	OptionsStr << "{";
	if ( DecoderName )
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":false";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options,ErrorBuffer,std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName?TestDataName:"<null>") << ") Size: " << TestData.size() << std::endl;
	
	int FirstFrameNumber = 9999 - 100;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.data(), TestData.size(), FirstFrameNumber);
		if (Result < 0)
			throw std::runtime_error("DecoderTest: PushData error");

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
			std::cerr  << "Decoded testdata; " << MetaJson << " FrameNumber=" << FrameNumber << " Should be " << FirstFrameNumber << std::endl;
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


//	give the decoder loads of data to decode, then try and destroy the decoder
//	whilst its still decoding (to crash android)
void DestroyMidDecodeTest(const char* TestDataName,CompareFunc_t* Compare,const char* DecoderName)
{
	std::Debug << __PRETTY_FUNCTION__ << "(" << (TestDataName?TestDataName:"<null>") << "," << (DecoderName?DecoderName:"<null>") << ")" << std::endl;
	auto TestData = LoadDataFromFilename(TestDataName);

	std::stringstream OptionsStr;
	OptionsStr << "{";
	if ( DecoderName )
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":false";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options,ErrorBuffer,std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName?TestDataName:"<null>") << ") Size: " << TestData.size() << std::endl;
	
	size_t DataRepeat = 500;
	int FirstFrameNumber = 9999 - 100;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.data(), TestData.size(), FirstFrameNumber);
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
	std::array<char,100> ErrorBuffer={0};
	auto Handle = PopH264_CreateEncoder(EncoderOptionsJson, ErrorBuffer.data(), ErrorBuffer.size() );
	std::cerr << "PopH264_CreateEncoder handle=" << Handle << " error=" << std::string(ErrorBuffer.data()) << std::endl;
	
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
	PopH264_EncoderPushFrame( Handle, TestMetaJson, TestImage, nullptr, nullptr, ErrorBuffer.data(), ErrorBuffer.size() );
	std::cerr  << "PopH264_EncoderPushFrame error=" << std::string(ErrorBuffer.data()) << std::endl;
	
	//	todo: decode it again
	
	PopH264_DestroyEncoder(Handle);
}


void EncoderYuv8_88Test(int Width,int Height,const char* EncoderName="")
{
	std::stringstream EncoderOptionsJson;
	EncoderOptionsJson << "{\n";
	EncoderOptionsJson << "	\"Encoder\":\"" << EncoderName << "\"	";
	EncoderOptionsJson << "}";
	std::cerr << "Encoder options: " << EncoderOptionsJson.str() << std::endl;
	
	char ErrorBuffer[1000] = {0};
	auto Handle = PopH264_CreateEncoder(EncoderOptionsJson.str().c_str(), ErrorBuffer, std::size(ErrorBuffer) );
	std::cerr << "PopH264_CreateEncoder EncoderName=" << EncoderName << " handle=" << Handle << " error=" << ErrorBuffer << std::endl;

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

	auto ErrorBufferSize = static_cast<int>( std::size(ErrorBuffer) );
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, ErrorBufferSize );
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, ErrorBufferSize);
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, ErrorBufferSize);
	PopH264_EncoderPushFrame(Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, ErrorBufferSize);
	PopH264_EncoderPushFrame( Handle, TestMetaJson, Yuv.GetPixelsArray().GetArray(), nullptr, nullptr, ErrorBuffer, ErrorBufferSize );
	
	if ( strlen(ErrorBuffer) )
	{
		std::cerr << "PopH264_EncoderPushFrame error=" << ErrorBuffer << std::endl;
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
				std::cerr << "Re-decode, too many errors" << std::endl;
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		std::Debug << "Encoder got encoded packet: x" << FrameSize << std::endl;
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
		std::cerr << e.what() << std::endl;
	}
}



int main()
{
#if defined(TARGET_WINDOWS)
	if ( Platform::IsDebuggerAttached() )
		Platform::CaptureStdErr();
#endif
	
	
	// Must be called prior to running any tests
	testing::InitGoogleTest();
	
	std::string_view GTestFilter = "";
	//GTestFilter = "**DecodeFileFirstFrame**";
	//GTestFilter = "**DecodeFileFirstFrame_DripFeedData**";
	//GTestFilter = "**DecodeFile**";

	if ( !GTestFilter.empty() )
	{
		using namespace testing;
		GTEST_FLAG(filter) = std::string(GTestFilter);
	}
	
	static bool BreakOnTestError = false;
	if ( BreakOnTestError )
		GTEST_FLAG_SET(break_on_failure,true);
	
	const auto ReturnCode = RUN_ALL_TESTS();
	
	if ( ReturnCode != 0 )
		return ReturnCode;
	
	std::cerr << "Integration tests succeeded!" << std::endl;
	return 0;
}



class DecodedImage_t
{
public:
	DecodedImage_t()	{};
	DecodedImage_t(SoyPixelsImpl& Pixels);
	
	SoyPixelsFormat::Type	GetFormat();

	
	std::vector<uint8_t>	mPlane0;
	std::vector<uint8_t>	mPlane1;
	std::vector<uint8_t>	mPlane2;
	std::string				mMetaJson;
	int						mWidth = 0;
	int						mHeight = 0;
	std::vector<SoyPixelsFormat::Type>	mPlaneFormats;
};

DecodedImage_t::DecodedImage_t(SoyPixelsImpl& Pixels)
{
	std::span<uint8_t> PixelsData( Pixels.GetPixelsArray().GetArray(), Pixels.GetPixelsArray().GetDataSize() );
	std::copy( PixelsData.begin(), PixelsData.end(), std::back_inserter(mPlane0) );
	mPlaneFormats.push_back(Pixels.GetFormat() );
	mWidth = Pixels.GetWidth();
	mHeight = Pixels.GetHeight();
}

SoyPixelsFormat::Type DecodedImage_t::GetFormat()
{
	switch ( mPlaneFormats.size() )
	{
		case 0:	return SoyPixelsFormat::Invalid;
		case 1:	return mPlaneFormats[0];
		case 2: return SoyPixelsFormat::GetMergedFormat( mPlaneFormats[0], mPlaneFormats[1] );
		case 3: return SoyPixelsFormat::GetMergedFormat( mPlaneFormats[0], mPlaneFormats[1], mPlaneFormats[2] );
		default: break;
	}
	std::stringstream Error;
	Error << "Don't know how to combine " << mPlaneFormats.size() << " pixel formats together";
	throw std::runtime_error(Error.str());
}

class DecodeResults_t
{
public:
	int					FrameCount = 0;
	int					Width = 0;
	int					Height = 0;
	H264Profile::Type	Profile = H264Profile::Invalid;
	
	friend std::ostream& operator<<(std::ostream& os, const DecodeResults_t& Params)
	{
		os << "DecodeResults_t-->";
		os << " FrameCount=" << Params.FrameCount;
		os << " Width=" << Params.Width;
		os << " Height=" << Params.Height;
		os << " Profile=" << Params.Profile;
		return os;
	}
};

class DecodeTestParams_t
{
public:
	std::string		Filename;
	std::string		DecoderName;
	DecodeResults_t	ExpectedResults;
	
	friend std::ostream& operator<<(std::ostream& os, const DecodeTestParams_t& Params)
	{
		os << "DecodeTestParams_t-->";
		os << " Filename=" << Params.Filename;
		os << " DecoderName=" << Params.DecoderName;
		os << " ExpectedResults=" << Params.ExpectedResults;
		return os;
	}
};


//	params for DecodeFileFrames- the process, not the test
class DecodeFileFramesParams_t
{
public:
	std::string		mFilename;
	std::string		mDecoderName;
	size_t			mPushDataBlockSize = 0;	//	if 0, send everything. Used to drip feed data for testing
};

class PopH264_Decode_Tests : public testing::TestWithParam<DecodeTestParams_t>
{
};

auto DecodeTestValues = ::testing::Values
(
	//	hevc
 //	gr: this is supposed to be 68 frames, but 62 come out on apple atm
	//DecodeTestParams_t{.Filename="TestData/AppleSpatialRobotNutcracker.h265", .ExpectedResults{.FrameCount=68,.Width=1920,.Height=1080} }
	DecodeTestParams_t{.Filename="TestData/AppleSpatialRobotNutcracker.h265", .ExpectedResults{.FrameCount=62,.Width=1920,.Height=1080} },

 
	DecodeTestParams_t{.Filename="TestData/AppleSpatialRobotNutcracker.h264", .ExpectedResults{.FrameCount=68,.Width=1920,.Height=1080,.Profile=H264Profile::High4} },
	DecodeTestParams_t{.Filename="TestData/Issue83.h264", .ExpectedResults{.FrameCount=563,.Width=1920,.Height=1080,.Profile=H264Profile::High4} },

	//	data from ffmpeg's udp:// streaming protocol
	//	ffmpeg -filter_complex ddagrab=output_idx=0:framerate=60:draw_mouse=0,hwdownload,format=bgra  -c:v libx264 -tune zerolatency -preset ultrafast -s 512x512 -r 60 -f h264 udp://127.0.0.1:10000
	//	https://github.com/NewChromantics/PopH264/issues/80
	DecodeTestParams_t{.Filename="TestData/FfmpegUdpStream_yuv420p.h264", .ExpectedResults{.FrameCount=563,.Width=960,.Height=540,.Profile=H264Profile::High4} },
	//	this fails on apple as its yuv_444
	//DecodeTestParams_t{.Filename="TestData/FfmpegUdpStream_yuv444.h264", .ExpectedResults{.FrameCount=563,.Width=960,.Height=540,.Profile=H264Profile::High4} },



	// //	depth.h264 has IDRs before SPS/PPS
	//DecodeTestParams_t{.Filename="TestData/Depth.h264", .ExpectedResults{.FrameCount=1} }
	//DecodeTestParams_t{.Filename="TestData/Colour.h264", .ExpectedResults{.FrameCount=1,.Width=640,.Height=480,.Profile=H264Profile::Baseline} }
	//DecodeTestParams_t{.Filename="TestData/Main5.h264", .ExpectedResults{.FrameCount=15,.Width=1280,.Height=2708,.Profile=H264Profile::Main} },

	//	windows width 96
	//	apple width 128 (padded?)
#if defined(TARGET_WINDOWS)
 DecodeTestParams_t{.Filename="RainbowGradient.h264", .ExpectedResults{.FrameCount=1,.Width=96,.Height=256,.Profile=H264Profile::Baseline} },
#else
 DecodeTestParams_t{.Filename="RainbowGradient.h264", .ExpectedResults{.FrameCount=1,.Width=128,.Height=256,.Profile=H264Profile::Baseline} },
#endif
	DecodeTestParams_t{.Filename="Condense.h264", .ExpectedResults{.FrameCount=1,.Width=2560,.Height=2000,.Profile=H264Profile::Baseline} },

	//	greyscale isnt decoding on apple or windows11
	//	mediafoundation always has MF_E_TRANSFORM_NEED_MORE_INPUT, so possibly the data is incomplete (ffmpeg I believe will decode it though)
	//DecodeTestParams_t{.Filename="GreyscaleGradient.h264", .ExpectedResults{.FrameCount=1,.Width=10,.Height=256,.Profile=H264Profile::Baseline} }

	DecodeTestParams_t{.Filename="Cat.jpg", .ExpectedResults{.FrameCount=1,.Width=64,.Height=20} }
);
	
INSTANTIATE_TEST_SUITE_P( PopH264_Decode_Tests, PopH264_Decode_Tests, DecodeTestValues );



void GenerateFakeImages(std::string_view Filename,std::function<bool(DecodedImage_t)> OnDecodedImage)
{
	if ( Filename == "128x128_Greyscale" )
	{
		SoyPixels Pixels( SoyPixelsMeta(128, 128, SoyPixelsFormat::Greyscale) );
		DecodedImage_t Image( Pixels );
		OnDecodedImage( Image );
		return;
	}
	
	if ( Filename == "128x128_Yuv_8_8_8" )
	{
		SoyPixels Pixels( SoyPixelsMeta(128, 128, SoyPixelsFormat::Yuv_8_8_8) );
		DecodedImage_t Image( Pixels );
		OnDecodedImage( Image );
		return;
	}
	
	if ( Filename == "128x128_Yuv_8_88" )
	{
		SoyPixels Pixels( SoyPixelsMeta(128, 128, SoyPixelsFormat::Yuv_8_88) );
		DecodedImage_t Image( Pixels );
		OnDecodedImage( Image );
		return;
	}
	
	std::stringstream Error;
	Error << "No fake image named " << Filename;
	throw std::runtime_error(Error.str());
}


//	throws when we hit a decode error
//	exits cleanly only if we get an EOF
void DecodeFileFrames(DecodeFileFramesParams_t Params,std::function<bool(DecodedImage_t)> OnDecodedImage)
{
	auto Filename = Params.mFilename;
	auto DecoderName = Params.mDecoderName;
	
	//	fake images
	try
	{
		GenerateFakeImages( Filename, OnDecodedImage );
		return;
	}
	catch(std::exception& e)
	{
		//	not fake image
	}
	
	auto TestData = LoadDataFromFilename(Filename);
	
	std::stringstream OptionsJson;
	OptionsJson << "{";
	if ( !DecoderName.empty() )
		OptionsJson << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsJson << "\"VerboseDebug\":false";
	OptionsJson << "}";
	
	std::array<char,1024> ErrorBuffer = {0};
	auto Decoder = PopH264_CreateDecoder( OptionsJson.str().c_str(), ErrorBuffer.data(), ErrorBuffer.size() );
	if ( Decoder == PopH264_NullInstance )
	{
		std::stringstream Error;
		Error << "Failed to allocate decoder with filename=" << Filename << ", decoder=" << DecoderName << ". Error=" << ErrorBuffer.data();
		throw std::runtime_error(Error.str());
	}
	
	//	push input data
	//	when we do https://github.com/NewChromantics/PopH264/issues/87
	//	we should require frame number 0 when we dont know the packet breaks
	int FirstFrameNumber = 9999;
	{
		//	drip feed data
		auto BlockSize = Params.mPushDataBlockSize;
		if ( BlockSize == 0 )
			BlockSize = TestData.size();
		
		for ( auto i=0;	i<TestData.size();	i+=BlockSize )
		{
			auto NextBlock = std::span(TestData);
			NextBlock = NextBlock.subspan( i );
			auto NextBlockSize = std::min( BlockSize, NextBlock.size() );
			NextBlock = NextBlock.subspan( 0, NextBlockSize );
			
			auto Result = PopH264_PushData( Decoder, NextBlock.data(), NextBlock.size(), FirstFrameNumber );
			if (Result < 0)
				throw std::runtime_error("DecoderTest: PushData error");
		}
		PopH264_PushEndOfStream( Decoder );
	}
	
	//	pop frames
	//	todo: add a proper timeout instead of loop*pause
	bool HadEndOfStream = false;
	
	for ( auto i=0;	i<1000;	i++ )
	{
		std::this_thread::sleep_for( std::chrono::milliseconds(100) );
		
		std::array<char,1000> MetaJsonBuffer = {0};
		PopH264_PeekFrame( Decoder, MetaJsonBuffer.data(), MetaJsonBuffer.size() );
		std::string MetaJson(MetaJsonBuffer.data());
		PopJson::Value_t Meta(MetaJson);
		
		if ( Meta.HasKey("Error",MetaJson) )
		{
			auto Error = Meta.GetValue("Error",MetaJson).GetString(MetaJson);
			//	deallocate
			PopH264_DestroyDecoder( Decoder );
			throw std::runtime_error( Error );
		}
		
		//	gr: can we have EOF and a frame in the same packet?
		//	gr: should we? (no!)
		if ( Meta.HasKey("EndOfStream",MetaJson) )
		{
			HadEndOfStream = true;
			break;
		}

		DecodedImage_t Image;
		Image.mMetaJson = MetaJson;
		
		//	pull out image meta
		auto PlaneMetas = Meta.GetValue("Planes", MetaJson);
		for ( auto Plane=0;	Plane<PlaneMetas.GetChildCount();	Plane++ )
		{
			auto Plane0Meta = PlaneMetas.GetValue(Plane, MetaJson);
			if ( Plane == 0 )
			{
				Image.mWidth = Plane0Meta.GetValue("Width",MetaJson).GetInteger(MetaJson);
				Image.mHeight = Plane0Meta.GetValue("Height",MetaJson).GetInteger(MetaJson);
			}
			auto FormatName = Plane0Meta.GetValue("Format",MetaJson).GetString(MetaJson);
			auto Format = SoyPixelsFormat::ToType(FormatName);
			Image.mPlaneFormats.push_back(Format);

			auto DataSize = Plane0Meta.GetValue("DataSize",MetaJson).GetInteger(MetaJson);
			if ( Plane == 0 )
				Image.mPlane0.resize( DataSize );
			if ( Plane == 1 )
				Image.mPlane1.resize( DataSize );
			if ( Plane == 2 )
				Image.mPlane2.resize( DataSize );
		}

		auto FrameNumber = PopH264_PopFrame( Decoder, Image.mPlane0.data(), Image.mPlane0.size(), Image.mPlane1.data(), Image.mPlane1.size(), Image.mPlane2.data(), Image.mPlane2.size() );
		EXPECT_EQ( FrameNumber, FirstFrameNumber )<< "Decoded testdata; " << MetaJson << " FrameNumber=" << FrameNumber << " Should be " << FirstFrameNumber << std::endl;
		bool IsValid = FrameNumber >= 0;
		if ( !IsValid )
			continue;
		
		bool Continue = OnDecodedImage( Image );
		
		//	caller wants to break early
		if ( !Continue )
		{
			PopH264_DestroyDecoder( Decoder );
			return;
		}
	}
	
	//	deallocate
	PopH264_DestroyDecoder( Decoder );

	if ( !HadEndOfStream )
	{
		throw std::runtime_error("Decoding timed out (Never got EndOfStream)");
	}
}


DecodedImage_t DecodeFileFirstFrame(DecodeFileFramesParams_t Params)
{
	DecodedImage_t FirstImage;
	bool HadImage = false;
	
	auto OnDecodedImage = [&](DecodedImage_t Image)
	{
		//	only need first frame
		FirstImage = Image;
		HadImage = true;
		return false;
	};
	//DecodeResults_t Results;

	DecodeFileFrames( Params, OnDecodedImage );
	
	//EXPECT_EQ( HadImage, true ) << Filename << " decoded no frames";
	if ( !HadImage )
		throw std::runtime_error( Params.mFilename + " decoded no frames");
	
	return FirstImage;
}


void PopH264_Decode_Tests_DecodeFileFirstFrame(const DecodeTestParams_t& Params,size_t DecodeFileBlockSize)
{
	std::cerr << "Decode first frame of " << Params.Filename << std::endl;
	
	//	throws if failed
	DecodeFileFramesParams_t DecodeParams;
	DecodeParams.mFilename = Params.Filename;
	DecodeParams.mDecoderName = Params.DecoderName;
	DecodeParams.mPushDataBlockSize = DecodeFileBlockSize;

	auto Image = DecodeFileFirstFrame( DecodeParams );
	
	//EXPECT_EQ( Results.FrameCount, Params.ExpectedResults.FrameCount );
	EXPECT_EQ( Image.mWidth, Params.ExpectedResults.Width );
	EXPECT_EQ( Image.mHeight, Params.ExpectedResults.Height );
	//EXPECT_EQ( Results.Profile, Params.ExpectedResults.Profile );
}

//	this is essentially the same as DecodeFile but using this utility function
TEST_P(PopH264_Decode_Tests,DecodeFileFirstFrame)
{
	auto DecodeFileBlockSize = 0;
	PopH264_Decode_Tests_DecodeFileFirstFrame( GetParam(), DecodeFileBlockSize );
}

TEST_P(PopH264_Decode_Tests,DecodeFileFirstFrame_DripFeedData)
{
	//	https://github.com/NewChromantics/PopH264/issues/89
	GTEST_SKIP() << "todo: issue 89; drip-feeding data doesnt work";

	auto DecodeFileBlockSize = 1024;
	PopH264_Decode_Tests_DecodeFileFirstFrame( GetParam(), DecodeFileBlockSize );
}


void PopH264_Decode_Tests_DecodeFile(const DecodeTestParams_t& Params,size_t DecodeFileBlockSize)
{
	std::cerr << "DecodeFile " << Params.Filename << std::endl;

	DecodeResults_t Results;
	
	auto OnDecodedImage = [&](DecodedImage_t Image)
	{
		std::cerr << "Got frame #" << Results.FrameCount << " meta=" << Image.mMetaJson << std::endl;
		Results.Width = Image.mWidth;
		Results.Height = Image.mHeight;
		//	get h264 meta
		//Results.Profile = Image.
		Results.FrameCount++;
		return true;
	};
	
	DecodeFileFramesParams_t DecodeParams;
	DecodeParams.mFilename = Params.Filename;
	DecodeParams.mDecoderName = Params.DecoderName;
	DecodeParams.mPushDataBlockSize = DecodeFileBlockSize;

	DecodeFileFrames( DecodeParams, OnDecodedImage );
	
	//	compare with expected results
	EXPECT_EQ( Results.FrameCount, Params.ExpectedResults.FrameCount );
	EXPECT_EQ( Results.Width, Params.ExpectedResults.Width );
	EXPECT_EQ( Results.Height, Params.ExpectedResults.Height );
	//EXPECT_EQ( Results.Profile, Params.ExpectedResults.Profile );
}

TEST_P(PopH264_Decode_Tests,DecodeFile)
{
	auto DecodeFileBlockSize = 0;
	PopH264_Decode_Tests_DecodeFile( GetParam(), DecodeFileBlockSize );
}

TEST_P(PopH264_Decode_Tests,DecodeFile_DripFeedData)
{
	//	https://github.com/NewChromantics/PopH264/issues/89
	GTEST_SKIP() << "todo: issue 89; drip-feeding data doesnt work";

	auto DecodeFileBlockSize = 1024;
	PopH264_Decode_Tests_DecodeFile( GetParam(), DecodeFileBlockSize );
}











class EncodeTestParams_t
{
public:
	std::string		InputImageFilename;
	int				EncodeFrameCount = 10;
	
	friend std::ostream& operator<<(std::ostream& os, const EncodeTestParams_t& Params)
	{
		os << "EncodeTestParams_t-->";
		os << " InputImageFilename=" << Params.InputImageFilename;
		return os;
	}
	
};

class PopH264_Encode_Tests : public testing::TestWithParam<EncodeTestParams_t>
{
};

auto EncodeTestValues = ::testing::Values
(
 EncodeTestParams_t{.InputImageFilename="RainbowGradient.h264"},
 //EncodeTestParams_t{.InputImageFilename="GreyscaleGradient.h264"},
 //EncodeTestParams_t{.InputImageFilename="Cat.jpg"},
 //EncodeTestParams_t{.InputImageFilename="128x128_Greyscale"},
 EncodeTestParams_t{.InputImageFilename="128x128_Yuv_8_8_8"},
 EncodeTestParams_t{.InputImageFilename="128x128_Yuv_8_88"}
);
	
INSTANTIATE_TEST_SUITE_P( PopH264_Encode_Tests, PopH264_Encode_Tests, EncodeTestValues );



TEST_P(PopH264_Encode_Tests,EncodeFile)
{
	auto Params = GetParam();
	
	DecodeFileFramesParams_t DecodeParams;
	DecodeParams.mFilename = Params.InputImageFilename;
	auto InputImage = DecodeFileFirstFrame( DecodeParams );
	
	std::stringstream OptionsJson;
	OptionsJson << "{";
	//OptionsJson << "\"VerboseDebug\":true";
	OptionsJson << "}";
	
	std::array<char,1024> ErrorBuffer = {0};
	auto Encoder = PopH264_CreateEncoder( OptionsJson.str().c_str(), ErrorBuffer.data(), ErrorBuffer.size() );
	if ( Encoder == PopH264_NullInstance )
	{
		std::stringstream Error;
		Error << "Failed to allocate Encoder with filename=" << Params.InputImageFilename << ". Error=" << ErrorBuffer.data();
		throw std::runtime_error(Error.str());
	}
	
	//	write image
	{
		//	gr: would be nice if encode image meta is same as decoded image meta
		std::stringstream TestMetaJson;
		TestMetaJson << "{";
		TestMetaJson << "\"Width\":" << InputImage.mWidth << ",";
		TestMetaJson << "\"Height\":" << InputImage.mHeight << ",";
		if ( !InputImage.mPlane0.empty() )
			TestMetaJson << "\"LumaSize\":" << InputImage.mPlane0.size() << ",";
		if ( !InputImage.mPlane1.empty() )
			TestMetaJson << "\"ChromaUSize\":" << InputImage.mPlane1.size() << ",";
		if ( !InputImage.mPlane2.empty() )
			TestMetaJson << "\"ChromaVSize\":" << InputImage.mPlane2.size() << ",";
		auto Format = InputImage.GetFormat();
		TestMetaJson << "\"Format\":\"" << Format << "\",";
		TestMetaJson << "\"TestMeta\":\"PurpleMonkeyDishwasher\"";
		//TestMetaJson << ",\"Keyframe\":true";
		
		TestMetaJson << "}";

		for ( auto PushCount=0;	PushCount<Params.EncodeFrameCount;	PushCount++ )
		{
			PopH264_EncoderPushFrame( Encoder, TestMetaJson.str().c_str(), InputImage.mPlane0.data(), InputImage.mPlane1.data(), InputImage.mPlane2.data(), ErrorBuffer.data(), ErrorBuffer.size() );
			std::string Error( ErrorBuffer.data() );
			if ( !Error.empty() )
				FAIL() << "PopH264_EncoderPushFrame() error; " << Error;
		}
		PopH264_EncoderEndOfStream(Encoder);
	}

	bool HadEndOfStream = false;
	//	todo: re-decode packets to make sure we can decode what we encode
	//std::vector<std::vector<uint8_t>> Packets;
	std::vector<H264NaluContent::Type> H264PacketTypes;

	//	read out encoded packets
	for ( auto i=0;	i<200;	i++ )
	{
		std::this_thread::sleep_for( std::chrono::milliseconds(100) );
		
		//	read meta first to make sure test data is propogated out again
		std::array<char,1000> MetaJsonBuffer = {0};
		PopH264_EncoderPeekData(Encoder, MetaJsonBuffer.data(), MetaJsonBuffer.size() );
		std::string MetaJson(MetaJsonBuffer.data());
		std::Debug << "PopH264_EncoderPeekData meta: " << MetaJson << std::endl;
		PopJson::Value_t Meta(MetaJson);
		
		if ( Meta.HasKey("Error",MetaJson) )
		{
			auto Error = Meta.GetValue("Error",MetaJson).GetString(MetaJson);
			ADD_FAILURE() << "Error found in peek meta; " << Error;
			break;
		}
		
		//	any packets ready to pop?
		auto OutputQueueCount = Meta.GetValue("OutputQueueCount",MetaJson).GetInteger( MetaJson );
		
		//	gr: can we have EOF and a frame in the same packet?
		//	gr: should we? (no!)
		if ( Meta.HasKey("EndOfStream",MetaJson) )
		{
			HadEndOfStream = true;
			break;
		}
		
		if ( OutputQueueCount == 0 )
			continue;
		std::array<uint8_t,1024*100> PacketBuffer;
		auto FrameSize = PopH264_EncoderPopData( Encoder, PacketBuffer.data(), PacketBuffer.size() );
		if ( FrameSize < 0 )
			continue;
		
		//	check for test data presence
		{
			auto TestString = "PurpleMonkeyDishwasher";
			auto FoundPos = std::string(MetaJson).find(TestString);
			if (FoundPos == std::string::npos)
			{
				std::Debug << "Test string missing from meta " << TestString << std::endl;
			}
		}
		
		std::span PacketData( PacketBuffer.data(), PacketBuffer.size() );
		auto PacketType = H264::GetPacketType(PacketData);

		std::Debug << "Encoder packet: " << PacketType << " x" << FrameSize << std::endl;
		H264PacketTypes.push_back(PacketType);
	}
	PopH264_DestroyEncoder(Encoder);

	//	the encoder should output at least
	//	SPS (before first frame)
	//	PPS
	//	Keyframe
	//	EOS (last)
	if ( H264PacketTypes.empty() )
		FAIL() << "didn't encode any packets";

	EXPECT_EQ( HadEndOfStream, true ) << "Didn't get an EndOfStream";

	auto IsFrameContentType = [](H264NaluContent::Type Type)
	{
		switch( Type )
		{
		case H264NaluContent::Slice_CodedIDRPicture:
		case H264NaluContent::Slice_NonIDRPicture:
			return true;
		default:
			return false;
		}
	};

	auto FramePacketCount = std::count_if( H264PacketTypes.begin(), H264PacketTypes.end(), IsFrameContentType );

	EXPECT_EQ( FramePacketCount, Params.EncodeFrameCount );

	if ( !H264PacketTypes.empty() )
	{
		auto SpsIndex = FindIndex( H264PacketTypes, H264NaluContent::SequenceParameterSet );
		auto PpsIndex = FindIndex( H264PacketTypes, H264NaluContent::PictureParameterSet );
		auto FirstKeyframe = FindIndex( H264PacketTypes, H264NaluContent::Slice_CodedIDRPicture );
		//EXPECT_EQ( H264PacketTypes.back(), H264NaluContent::EndOfStream ) << "last packet decoded is not EOS";
		EXPECT_NE( SpsIndex, -1 ) << "missing SPS packet";
		EXPECT_NE( PpsIndex, -1 ) << "missing PPS packet";
		EXPECT_NE( FirstKeyframe, -1 ) << "missing a keyframe packet";
		
		EXPECT_LE( SpsIndex, FirstKeyframe ) << "keyframe before SPS";
		EXPECT_LE( SpsIndex, FirstKeyframe ) << "keyframe before PPS";
	}
}











class PopH264_General_Tests : public testing::TestWithParam<bool>
{
protected:
};


TEST(PopH264_General_Tests, PopJsonUnitTest )
{
	EXPECT_NO_THROW( PopJson::UnitTest() );
}

TEST(PopH264_General_Tests, PopH264UnitTest )
{
	//PopH264_UnitTest(nullptr);
	EXPECT_NO_THROW( PopH264_UnitTestThrows() );
}



TEST(PopH264_General_Tests, EncoderPopNull )
{
	std::array<char,100> ErrorBuffer={0};
	auto EncoderOptionsJson = "{}";
	auto Encoder = PopH264_CreateEncoder(EncoderOptionsJson, ErrorBuffer.data(), ErrorBuffer.size() );
	std::cerr << "PopH264_CreateEncoder Encoder=" << Encoder << " error=" << std::string(ErrorBuffer.data()) << std::endl;
	
	//	gr: this was crashing when passing for popping null on apple. It should not crash
	//
	//	gr: seems like it was only when there was a frame or EOF to pop?
	PopH264_EncoderEndOfStream(Encoder);
	auto Bytes = PopH264_EncoderPopData( Encoder, nullptr, 0 );
	PopH264_DestroyEncoder(Encoder);
}

TEST(PopH264_General_Tests,OldMain)
{
	GTEST_SKIP();
	
	SafeDecoderTest("RainbowGradient.h264", nullptr, nullptr );
	SafeDecoderTest("Cat.jpg", nullptr, nullptr );

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
	
	//	depth data has iframe, pps, sps order
	SafeDecoderTest("TestData/Main5.h264", nullptr, nullptr );
	SafeDecoderTest("TestData/Colour.h264", nullptr, nullptr );
	SafeDecoderTest("TestData/Depth.h264", nullptr, nullptr );
	SafeDecoderTest("../TestData/Colour.h264", nullptr, nullptr );
	//SafeDecoderTest("RainbowGradient.h264", CompareRainbow);
	//SafeDecoderTest("RainbowGradient.h264",CompareRainbow);
	
	
#if defined(TEST_ASSETS)
	SafeDecoderTest("GreyscaleGradient.h264",CompareGreyscale);
#endif

	EncoderGreyscaleTest();
}

#if !defined(TEST_ASSETS)
void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data)
{
}
#endif

