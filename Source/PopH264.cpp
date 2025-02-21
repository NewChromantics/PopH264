#include "PopH264.h"
#include "TDecoderInstance.h"
#include "TEncoderInstance.h"
#include "SoyLib/src/SoyPixels.h"
#include "Json11/json11.hpp"
#include "TInstanceManager.h"
#include "PopH264TestData.h"
#include <iostream>
#include "SoyLib/src/SoyH264.h"



namespace PopH264
{
	TInstanceManager<TEncoderInstance,std::string>		EncoderInstanceManager;
	TInstanceManager<TDecoderInstance,json11::Json&>	DecoderInstanceManager;
	
	void		Shutdown(bool DllExit);
}




#if defined(TARGET_LUMIN) || defined(TARGET_ANDROID)
const char* Platform::LogIdentifer = "PopH264";
#endif


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
	
	//	hail mary cleanup for globals
	if ( ul_reason_for_call == DLL_PROCESS_DETACH)
	{
		PopH264::Shutdown(true);
	}
	
	return TRUE;
}
#endif

#if !defined(TARGET_WINDOWS)
void __attribute__((destructor)) DllExit() 
{
	PopH264::Shutdown(true);
}
#endif

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

	

__export int32_t PopH264_PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size)
{
	auto Function = [&]()
	{
		auto Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
		//	Decoder.PopFrame
		auto Plane0Array = GetRemoteArray(Plane0, Plane0Size);
		auto Plane1Array = GetRemoteArray(Plane1, Plane1Size);
		auto Plane2Array = GetRemoteArray(Plane2, Plane2Size);
		int32_t FrameTimeMs = -1;
		Decoder->PopFrame( FrameTimeMs, GetArrayBridge(Plane0Array), GetArrayBridge(Plane1Array), GetArrayBridge(Plane2Array));
		return FrameTimeMs;
	};
	return SafeCall(Function, __func__, -99 );
}

__export int32_t PopH264_PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber)
{
	auto Function = [&]()
	{
		auto Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
		if ( Data == nullptr )
		{
			Decoder->PushEndOfStream();
		}
		else
		{
			std::span<uint8_t> DataArray( Data, DataSize );
			PopH264::FrameNumber_t FrameNum( FrameNumber );
			Decoder->PushData( DataArray, FrameNum );
		}
		return 0;
	};
	return SafeCall(Function, __func__, -1 );
}

__export int32_t PopH264_PushEndOfStream(int32_t Instance)
{
	return PopH264_PushData(Instance, nullptr, 0, 0);
}

__export int32_t PopH264_CheckDecoderUpdates(int32_t Instance)
{
	return PopH264_PushData(Instance, nullptr, 0, 1);
}


json11::Json::object GetMetaJson(json11::Json::object& MetaJson,const SoyPixelsMeta& Meta)
{
	using namespace json11;
	Json::array PlaneArray;
	
	auto AddPlaneJson = [&](SoyPixelsMeta& Plane)
	{
		auto Width = static_cast<int>(Plane.GetWidth());
		auto Height = static_cast<int>(Plane.GetHeight());
		auto Channels = static_cast<int>(Plane.GetChannels());
		auto DataSize = static_cast<int>(Plane.GetDataSize());
		auto FormatStr = SoyPixelsFormat::ToString(Plane.GetFormat());
		Json PlaneMeta = Json::object{
			{ "Width",Width },
			{ "Height",Height },
			{ "Channels",Channels },
			{ "DataSize",DataSize },
			{ "Format",FormatStr }
		};
		PlaneArray.push_back(PlaneMeta);
	};
	
	BufferArray<SoyPixelsMeta, 4> PlaneMetas;
	Meta.GetPlanes(GetArrayBridge(PlaneMetas));
	for (auto p = 0; p < PlaneMetas.GetSize(); p++)
	{
		auto& PlaneMeta = PlaneMetas[p];
		AddPlaneJson(PlaneMeta);
	}
	
	//	make final object
	//	todo: add frame numbers etc here
	MetaJson["Planes"] = PlaneArray;

	return MetaJson;
}

std::string GetMetaJson(const PopH264::TDecoderFrameMeta& Meta)
{
	auto Json = Meta.mMeta;
	GetMetaJson( Json, Meta.mPixelsMeta);
	
	if ( Meta.mEndOfStream )
		Json["EndOfStream"] = true;
	
	Json["FrameNumber"] = Meta.mFrameNumber;
	Json["QueuedFrames"] = static_cast<int>(Meta.mFramesQueued);
	
	//	coping with int32 by subtracting a good portion of the 80's and 90's
	//	need a time well within 2147483647
	static uint64_t BaseTime = 1669100000000;
	auto Now64 = SoyTime::Now().mTime;
	auto Decoded64 = Meta.mDecodedTime.mTime;
	int32_t Now32 = static_cast<int32_t>(Now64 - BaseTime);
	int32_t Decoded32 = ( Decoded64 > 0 ) ? static_cast<int32_t>(Decoded64 - BaseTime) : 0;
	Json["DecodedTimeMs"] = Decoded32;
	//	send the current clock time along with Decoded for a frame of reference
	Json["NowTimeMs"] = Now32;

	json11::Json TheJson = Json;
	std::string MetaJsonString = TheJson.dump();
	return MetaJsonString;
}


__export void PopH264_PeekFrame(int32_t Instance, char* JsonBuffer, int32_t JsonBufferSize)
{
	try
	{
		auto Device = PopH264::DecoderInstanceManager.GetInstance(Instance);
		auto Meta = Device->GetMeta();

		auto Json = GetMetaJson(Meta);
		Soy::StringToBuffer(Json, JsonBuffer, JsonBufferSize);
	}
	catch(std::exception& e)
	{
		json11::Json::object JsonObject;
		JsonObject["Error"] = e.what();
		auto Json = json11::Json(JsonObject).dump();
		Soy::StringToBuffer(Json, JsonBuffer, JsonBufferSize);
	}
}



__export void PopH264_Shutdown()
{
	auto Function = [&]()
	{
		PopH264::Shutdown(false);
		return 0;
	};
	SafeCall( Function, __func__, 0 );
}



__export void PopH264_EnumDecoders(char* DecodersJsonBuffer, int32_t DecodersJsonBufferLength)
{
	try
	{
		json11::Json::array DecoderNames;
		auto EnumDecoder = [&](const std::string& DecoderName)
		{
			DecoderNames.push_back(DecoderName);
		};
		PopH264::EnumDecoderNames(EnumDecoder);

		json11::Json::object Meta;
		Meta["DecoderNames"] = DecoderNames;
		
		auto MetaString = json11::Json(Meta).dump();
		Soy::StringToBuffer(MetaString, DecodersJsonBuffer, DecodersJsonBufferLength);
	}
	catch (std::exception& e)
	{
		Soy::StringToBuffer(e.what(), DecodersJsonBuffer, DecodersJsonBufferLength);
	}
	catch (...)
	{
		Soy::StringToBuffer("Unknown exception", DecodersJsonBuffer, DecodersJsonBufferLength);
	}
}


__export int32_t PopH264_CreateDecoder(const char* OptionsJson, char* ErrorBuffer, int32_t ErrorBufferLength)
{
	try
	{
		std::string ParseError;
		json11::Json Options = json11::Json::parse(OptionsJson ? OptionsJson : "{}", ParseError);
		if (!ParseError.empty())
		{
			ParseError = std::string("PopCameraDevice_CreateCameraDevice parse json error; ") + ParseError;
			throw Soy::AssertException(ParseError);
		}
		auto InstanceId = PopH264::DecoderInstanceManager.CreateInstance(Options);
		return InstanceId;
	}
	catch (std::exception& e)
	{
		Soy::StringToBuffer(e.what(), ErrorBuffer, ErrorBufferLength);
		return PopH264_NullInstance;
	}
	catch (...)
	{
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferLength);
		return PopH264_NullInstance;
	}
}

__export void PopH264_DestroyDecoder(int32_t Instance)
{
	auto Function = [&]()
	{
		PopH264::DecoderInstanceManager.FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_DestroyInstance(int32_t Instance)
{
	PopH264_DestroyDecoder(Instance);
}




__export int32_t PopH264_CreateEncoder(const char* OptionsJson,char* ErrorBuffer,int32_t ErrorBufferSize)
{
	try
	{
		std::string ParamsJson(OptionsJson ? OptionsJson : "{}");
		auto InstanceId = PopH264::EncoderInstanceManager.CreateInstance( ParamsJson );
		return InstanceId;
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
		return PopH264_NullInstance;
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferSize );
		return PopH264_NullInstance;
	}
}

__export void PopH264_DestroyEncoder(int32_t Instance)
{
	auto Function = [&]()
	{
		PopH264::EncoderInstanceManager.FreeInstance(Instance);
		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_EncoderPushFrame(int32_t Instance,const char* MetaJson,const uint8_t* LumaData,const uint8_t* ChromaUData,const uint8_t* ChromaVData,char* ErrorBuffer,int32_t ErrorBufferSize)
{
	try
	{
		auto Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		std::string Meta( MetaJson ? MetaJson : "" );
		Encoder->PushFrame( Meta, LumaData, ChromaUData, ChromaVData );
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
		Soy::StringToBuffer( e.what(), ErrorBuffer, ErrorBufferSize );
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
		Soy::StringToBuffer("Unknown exception", ErrorBuffer, ErrorBufferSize );
	}
}


__export void PopH264_EncoderEndOfStream(int32_t Instance)
{
	try
	{
		auto Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		Encoder->EndOfStream();
	}
	catch(std::exception& e)
	{
		std::Debug << __PRETTY_FUNCTION__ << " exception " << e.what() << std::endl;
	}
	catch(...)
	{
		std::Debug << __PRETTY_FUNCTION__ << " unknown exception" << std::endl;
	}
}


__export int32_t PopH264_EncoderPopData(int32_t Instance,uint8_t* DataBuffer,int32_t DataBufferSize)
{
	auto Function = [&]() -> int32_t
	{
		auto Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);

		//	sanitise input, but this should error!
		if ( !DataBuffer || DataBufferSize <= 0 )
		{
			DataBuffer = nullptr;
			DataBufferSize = 0;
		}
		
		auto Buffer = std::span( DataBuffer, DataBufferSize );

		//	check buffer size before we pop.
		//	allow pop of 0 to pop without copy & discard.
		auto NextPacketSize = Encoder->PeekNextFrameSize();
		bool BufferTooSmall = NextPacketSize > Buffer.size();
		if ( !Buffer.empty() && BufferTooSmall )
		{
			std::stringstream Error;
			Error << "Popping packet x" << NextPacketSize << " but buffer is only x" << Buffer.size() << " bytes";
			throw std::runtime_error(Error.str());
		}
		
		auto Packet = Encoder->PopPacket();
		auto PacketData = Packet.GetData();
		std::copy( PacketData.begin(), PacketData.end(), Buffer.begin() );
		
		return PacketData.size();
	};
	return SafeCall(Function, __func__, -1 );
}

__export void PopH264_EncoderPeekData(int32_t Instance,char* MetaJsonBuffer,int32_t MetaJsonBufferSize)
{
	//	wrapped in a safe call in case the json generation fails in some way
	auto Function = [&]()
	{
		using namespace json11;
		Json::object MetaJson;
		try
		{
			//	get next frame's meta
			auto Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);

			//	add generic meta
			MetaJson["OutputQueueCount"] = static_cast<int32_t>(Encoder->GetPacketQueueCount());

			auto EncoderName = Encoder->GetEncoderName();
			if ( !EncoderName.empty() )
				MetaJson["EncoderName"] = EncoderName;

			Encoder->PeekPacket(MetaJson);
		}
		catch(std::exception& e)
		{
			MetaJson["Error"] = e.what();
		}
		catch(...)
		{
			MetaJson["Error"] = "Unknown exception";
		}
		Json FinalJson(MetaJson);
		auto MetaJsonString = FinalJson.dump();
		Soy::StringToBuffer( MetaJsonString, MetaJsonBuffer, MetaJsonBufferSize );

		return 0;
	};
	SafeCall(Function, __func__, 0 );
}


__export void PopH264_EncoderAddOnNewPacketCallback(int32_t Instance,PopH264_Callback* Callback, void* Meta)
{
	auto Function = [&]()
	{
		if (!Callback)
			throw Soy::AssertException("PopH264_EncoderAddOnNewPacketCallback callback is null");
		
		auto Lambda = [Callback, Meta]()
		{
			Callback(Meta);
		};
		auto Encoder = PopH264::EncoderInstanceManager.GetInstance(Instance);
		Encoder->AddOnNewFrameCallback(Lambda);
		return 0;
	};
	SafeCall(Function, __func__, 0);
}


__export void PopH264_DecoderAddOnNewFrameCallback(int32_t Instance,PopH264_Callback* Callback, void* Meta)
{
	auto Function = [&]()
	{
		if (!Callback)
			throw Soy::AssertException("PopH264_DecoderAddOnNewFrameCallback callback is null");
		
		auto Lambda = [Callback, Meta]()
		{
			Callback(Meta);
		};
		auto Decoder = PopH264::DecoderInstanceManager.GetInstance(Instance);
		Decoder->AddOnNewFrameCallback(Lambda);
		return 0;
	};
	SafeCall(Function, __func__, 0);
}

__export int32_t PopH264_GetTestData(const char* Name,uint8_t* Buffer,int32_t BufferSize)
{
	auto Function = [&]()
	{
		auto BufferArray = GetRemoteArray( Buffer, BufferSize );
		size_t FullSize = 0;

		//	todo: catch "name doesnt exist"
		PopH264::GetTestData( Name, GetArrayBridge(BufferArray), FullSize );
		return size_cast<int32_t>(FullSize);
	};
	return SafeCall(Function, __func__, -1);
}


__export void UnityPluginLoad(/*IUnityInterfaces*/void*)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
}

__export void UnityPluginUnload()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	PopH264_Shutdown();
}


void PopH264::Shutdown(bool FromDllExit)
{
	PopH264::EncoderInstanceManager.FreeInstances();
	PopH264::DecoderInstanceManager.FreeInstances();
}


__export void PopH264_GetDebugStatsJson(char* JsonBuffer,int32_t JsonBufferSize)
{
	try
	{
		json11::Json::object Meta;
		Meta["DecoderInstanceCount"] = static_cast<int>(PopH264::DecoderInstanceManager.GetInstanceCount());
		Meta["EncoderInstanceCount"] = static_cast<int>(PopH264::EncoderInstanceManager.GetInstanceCount());
		
		auto Json = json11::Json(Meta).dump();
		Soy::StringToBuffer(Json, JsonBuffer, JsonBufferSize);
	}
	catch (std::exception& e)
	{
		json11::Json::object Meta;
		Meta["Error"] = e.what();

		auto Json = json11::Json(Meta).dump();
		Soy::StringToBuffer(Json, JsonBuffer, JsonBufferSize);
	}
}


static void RunTest(const std::string& TestName,std::function<void()> Test, std::function<PopH264_UnitTestCallback> Report)
{
	try
	{
		Test();
		const char* NoError = nullptr;
		Report(TestName.c_str(), NoError);
	}
	catch (std::exception& e)
	{
		Report(TestName.c_str(), e.what());
	}
	catch (...)
	{
		Report(TestName.c_str(), "Unknown exception");
	}
}

void Test_Decoder_DecodeTestFile(const char* Filename, const char* DecoderName, size_t DataRepeat);
void Test_Decoder_CreateAndDestroy();
void Test_Decoder_DecodeRainbow();
void Test_Decoder_DestroyMidDecodeRainbow();


__export void PopH264_UnitTestThrows()
{
	H264::UnitTest();
}



//	todo: drop this
//		these are not really unit tests, they're integration tests and should be covered by the test app's integration tests
__export void PopH264_UnitTest(PopH264_UnitTestCallback* OnTestResult)
{
	try
	{
		std::function<PopH264_UnitTestCallback> Report = [](const char* TestName, const char* Result)
		{
			if (!TestName)
				TestName = "Unknown Test";
			if (!Result)
				Result = "OK";
			std::cout << "Unit Test " << TestName << " result: " << Result << std::endl;
		};
		if (OnTestResult)
		{
			Report = OnTestResult;
		}

		Test_Decoder_DestroyMidDecodeRainbow();
		RunTest("Decoder_CreateAndDestroy", Test_Decoder_CreateAndDestroy, Report);
		RunTest("Decoder_DecodeRainbow", Test_Decoder_DecodeRainbow, Report);
		RunTest("Decoder_DestroyMidDecode", Test_Decoder_DestroyMidDecodeRainbow, Report);
	}
	catch (std::exception& e)
	{
		//	if the whole unit test throws an exception, report that as a test result
		if (OnTestResult)
		{
			OnTestResult(__PRETTY_FUNCTION__, e.what());
		}
	}
}


void Test_Decoder_CreateAndDestroy()
{
	auto Handle = PopH264_CreateDecoder(nullptr, nullptr, 0);
	PopH264_DestroyDecoder(Handle);
}

void Test_Decoder_DecodeRainbow()
{
	Test_Decoder_DecodeTestFile("RainbowGradient.h264",nullptr,1);
}

json11::Json ParseJsonObject(const std::string& JsonString)
{
	std::string Error;
	auto Json = json11::Json::parse(JsonString, Error);
	if (!Error.empty())
	{
		std::stringstream DebugError;
		DebugError << "Error parsing json; " << Error << "; Json=" << JsonString;
		throw std::runtime_error(DebugError.str());
	}
	if (!Json.is_object())
		throw std::runtime_error(std::string("Expecting json to parse to object; Json=") + JsonString);
	return Json;
}

void Test_Decoder_DecodeTestFile(const char* TestDataName, const char* DecoderName, size_t DataRepeat)
{
	//	gr: 1mb too big for windows on stack
	static uint8_t TestDataBuffer[1 * 1024 * 1024];

	Array<uint8_t> TestData;

	//	gr: using int (auto) here, causes some resolve problem with GetRemoteArray below
	size_t TestDataSize = PopH264_GetTestData(TestDataName, TestDataBuffer, std::size(TestDataBuffer));
	if (TestDataSize < 0)
		throw std::runtime_error("Missing test data");
	if (TestDataSize == 0)
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
	

	std::stringstream OptionsStr;
	OptionsStr << "{";
	if (DecoderName)
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":true";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options, ErrorBuffer, std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName ? TestDataName : "<null>") << ") Size: " << TestData.GetDataSize() << std::endl;

	bool HadEof = false;
	int FirstFrameNumber = 9999 - 100;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.GetArray(), (int)TestData.GetDataSize(), FirstFrameNumber);
		if (Result < 0)
			throw std::runtime_error("DecoderTest: PushData error");

		//	flush
		if (LastIteration)
			PopH264_PushEndOfStream(Handle);

		//	wait for it to decode
		for (auto i = 0; i < 1000; i++)
		{
			char MetaJson[1000];
			PopH264_PeekFrame(Handle, MetaJson, std::size(MetaJson));

			auto Meta = ParseJsonObject(MetaJson);
			bool ThisIsEof = false;
			if (Meta.object_items().count("EndOfStream"))
			{
				auto EndOfStream = Meta["EndOfStream"];
				if (!EndOfStream.is_bool())
					throw std::runtime_error("Frame meta had .EndOfStream but isn't a bool");

				if (EndOfStream.bool_value() == true)
				{
					ThisIsEof = true;
					HadEof = true;
				}
			}

			static uint8_t Plane0[1024 * 1024];
			static uint8_t Plane1[1024 * 1024];
			static uint8_t Plane2[1024 * 1024];
			auto FrameNumber = PopH264_PopFrame(Handle, Plane0, std::size(Plane0), Plane1, std::size(Plane1), Plane2, std::size(Plane2));
			std::stringstream Error;
			Error << "Decoded testdata; " << MetaJson << " FrameNumber=" << FrameNumber << " Should be " << FirstFrameNumber;
			//std::Debug << Error.str() << std::endl;
			bool IsValid = FrameNumber >= 0;
			if (!IsValid)
			{
				//std::this_thread::sleep_for(std::chrono::milliseconds(500));
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				continue;
			}

			//	if this is an EOF frame, the frame number might be 0
			if ( !ThisIsEof )
				if (FrameNumber != FirstFrameNumber)
					throw std::runtime_error("Wrong frame number from decoder");

			/*
			if (Compare)
				Compare(MetaJson, Plane0, Plane1, Plane2);
				*/
			if ( HadEof )
				break;
		}
	}

	if (!HadEof)
		throw std::runtime_error("Never had EOF");

	PopH264_DestroyInstance(Handle);
}



void Test_Decoder_DestroyMidDecode(const char* TestDataName, const char* DecoderName,size_t DataRepeat)
{
	//	gr: 1mb too big for windows on stack
	static uint8_t TestDataBuffer[1 * 1024 * 1024];

	Array<uint8_t> TestData;

	//	gr: using int (auto) here, causes some resolve problem with GetRemoteArray below
	size_t TestDataSize = PopH264_GetTestData(TestDataName, TestDataBuffer, std::size(TestDataBuffer));
	if (TestDataSize < 0)
		throw std::runtime_error("Missing test data");
	if (TestDataSize == 0)
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


	std::stringstream OptionsStr;
	OptionsStr << "{";
	if (DecoderName)
		OptionsStr << "\"Decoder\":\"" << DecoderName << "\",";
	OptionsStr << "\"VerboseDebug\":true";
	OptionsStr << "}";
	auto OptionsString = OptionsStr.str();
	auto* Options = OptionsString.c_str();
	char ErrorBuffer[1024] = { 0 };
	std::Debug << "PopH264_CreateDecoder()" << std::endl;
	auto Handle = PopH264_CreateDecoder(Options, ErrorBuffer, std::size(ErrorBuffer));

	std::Debug << "TestData (" << (TestDataName ? TestDataName : "<null>") << ") Size: " << TestData.GetDataSize() << std::endl;

	bool HadEof = false;
	std::string FatalError;
	int FirstFrameNumber = 9999 - 100;
	int FramesDecoded = 0;
	bool SentTermination = false;
	for (auto Iteration = 0; Iteration < DataRepeat; Iteration++)
	{
		auto LastIteration = Iteration == (DataRepeat - 1);
		FirstFrameNumber += 100;
		auto Result = PopH264_PushData(Handle, TestData.GetArray(), static_cast<int>(TestData.GetDataSize()), FirstFrameNumber);
		if (Result < 0)
		{
			if ( SentTermination )
				break;
			throw std::runtime_error("DecoderTest: PushData error");
		}

		//	flush
		if (LastIteration)
			PopH264_PushEndOfStream(Handle);

		//	at a "random" moment, once we've decoded at least one frame, free
		if ( FramesDecoded > 99999 )
		{
			if ( (rand() % 20) == 0 )
			{
				PopH264_DestroyInstance(Handle);
				SentTermination = true;
			}
		}

		int SleepOnNoFrameMs = 1;

		//	check for some decoded frames
		for (auto i = 0; i < 1; i++)
		{
			char MetaJson[1000] = {0};
			PopH264_PeekFrame(Handle, MetaJson, std::size(MetaJson));

			auto Meta = ParseJsonObject(MetaJson);
			if (Meta.object_items().count("EndOfStream"))
			{
				auto EndOfStream = Meta["EndOfStream"];
				if (!EndOfStream.is_bool())
					throw std::runtime_error("Frame meta had .EndOfStream but isn't a bool");

				if (EndOfStream.bool_value() == true)
					HadEof = true;
			}

			if (Meta.object_items().count("Error"))
			{
				FatalError = Meta["Error"].string_value();
				std::Debug << "Decoder error: " << FatalError << std::endl;
				throw std::runtime_error(std::string("Decoder had fatal error;")+FatalError);
				break;
			}
			
			static uint8_t Plane0[1024 * 1024];
			static uint8_t Plane1[1024 * 1024];
			static uint8_t Plane2[1024 * 1024];
			auto FrameNumber = PopH264_PopFrame(Handle, Plane0, std::size(Plane0), Plane1, std::size(Plane1), Plane2, std::size(Plane2));
			std::stringstream Error;
			Error << "Decoded testdata; " << MetaJson << " FrameNumber=" << FrameNumber << " Should be " << FirstFrameNumber;
			//std::Debug << Error.str() << std::endl;
			bool IsValid = FrameNumber >= 0;
			if (!IsValid)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(SleepOnNoFrameMs));
				continue;
			}

			FramesDecoded++;

			if ( HadEof )
				break;
		}
	}


	if (!HadEof)
	{
		//throw std::runtime_error("Never had EOF");
	}

	PopH264_DestroyInstance(Handle);
}

void Test_Decoder_DestroyMidDecodeRainbow()
{
	//	stress test creating, decoding and destroying mid-decode
	//for ( int i=0;	i<99000;	i++ )
	{
		Test_Decoder_DestroyMidDecode("RainbowGradient.h264",nullptr,100);
	}
}
