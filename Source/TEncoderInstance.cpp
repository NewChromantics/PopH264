#include "TEncoderInstance.h"
#include "SoyH264.h"
#include "magic_enum.hpp"
#include "json11.hpp"
#include "PopH264.h"

#if defined(TARGET_IOS) || defined(TARGET_OSX)
#define ENABLE_AVF
#endif

#if defined(TARGET_WINDOWS)
#define ENABLE_MEDIAFOUNDATION
#endif

#if defined(ENABLE_AVF)
#include "AvfEncoder.h"
#endif

#if defined(ENABLE_MEDIAFOUNDATION)
#include "MediaFoundationEncoder.h"
#endif

//	ENABLE_NVIDIA should be defined by makefile/project now
#if defined(ENABLE_NVIDIA)
#pragma warning("Nvidia encoder enabled")
#include "NvidiaEncoder.h"
#endif

PopH264::TEncoderInstance::TEncoderInstance(const std::string& OptionsJsonString)
{
	//	get options
	std::string ParseError;
	auto Options = json11::Json::parse(OptionsJsonString,ParseError);
	if ( ParseError.length() )
	{
		ParseError = "Error parsing encoder options json: " + ParseError;
		throw Soy::AssertException(ParseError);
	}
	
	auto OnOutputPacket = [this](TPacket& Packet)
	{
		this->OnNewPacket(Packet);
	};
	
	auto EncoderName = Options[POPH264_ENCODER_KEY_ENCODERNAME].string_value();

#if defined(ENABLE_AVF)
	if ( EncoderName.empty() || EncoderName == Avf::TEncoder::Name )
	{
		Avf::TEncoderParams Params(Options);
		mEncoder.reset( new Avf::TEncoder(Params,OnOutputPacket) );
		return;
	}
#endif
	
#if defined(ENABLE_MEDIAFOUNDATION)
	if (EncoderName.empty() || EncoderName == MediaFoundation::TEncoder::Name)
	{
		try
		{
			MediaFoundation::TEncoderParams Params(Options);
			mEncoder.reset(new MediaFoundation::TEncoder(Params, OnOutputPacket));
			return;
		}
		catch (std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}
#endif
	
	
#if defined(ENABLE_NVIDIA)
	if ( EncoderName.empty() || EncoderName == Nvidia::TEncoder::Name )
	{
		Nvidia::TEncoderParams Params(Options);
		mEncoder.reset( new Nvidia::TEncoder(Params,OnOutputPacket) );
		return;
	}
#endif
	
	//	success!
	if ( mEncoder )
		return;
	
	std::stringstream Error;
	Error << "No encoder supported (requested \"" << EncoderName << "\") empty=" << EncoderName.empty();
	throw Soy::AssertException(Error);
}

void PopH264::TEncoderInstance::EndOfStream()
{
	mEncoder->FinishEncoding();
}

static SoyPixels DummyPixels;
SoyPixelsImpl& GetDummyPixels(SoyPixelsMeta Meta)
{
	DummyPixels.Init(Meta);
	return DummyPixels;
}

void PopH264::TEncoderInstance::PushFrame(const std::string& Meta,const uint8_t* LumaDataPtr,const uint8_t* ChromaUDataPtr,const uint8_t* ChromaVDataPtr)
{
	std::string ParseError;
	auto Json = json11::Json::parse( Meta, ParseError );
	//	these return 0 if missing
	auto Width = Json[POPH264_ENCODEFRAME_KEY_WIDTH].int_value();
	auto Height = Json[POPH264_ENCODEFRAME_KEY_HEIGHT].int_value();
	auto LumaSize = Json[POPH264_ENCODEFRAME_KEY_LUMASIZE].int_value();
	auto ChromaUSize = Json[POPH264_ENCODEFRAME_KEY_CHROMAUSIZE].int_value();
	auto ChromaVSize = Json[POPH264_ENCODEFRAME_KEY_CHROMAVSIZE].int_value();
	auto Keyframe = Json[POPH264_ENCODEFRAME_KEY_KEYFRAME].bool_value();
	auto FormatName = Json[POPH264_ENCODEFRAME_KEY_FORMAT].string_value();

	//	check for data/size mismatch
	if ( LumaDataPtr && LumaSize==0 )
		throw Soy::AssertException("Luma pointer but zero LumaSize");
	if ( !LumaDataPtr && LumaSize!=0 )
		throw Soy::AssertException("Luma null but LumaSize nonzero");
	
	if ( ChromaUDataPtr && ChromaUSize==0 )
		throw Soy::AssertException("ChromaU pointer but zero ChromaUSize");
	if ( !ChromaUDataPtr && ChromaUSize!=0 )
		throw Soy::AssertException("ChromaU null but ChromaUSize nonzero");
	
	if ( ChromaVDataPtr && ChromaVSize==0 )
		throw Soy::AssertException("ChromaV pointer but zero ChromaVSize");
	if ( !ChromaVDataPtr && ChromaVSize!=0 )
		throw Soy::AssertException("ChromaV null but ChromaVSize nonzero");
	
	std::span LumaData( const_cast<uint8_t*>(LumaDataPtr), LumaSize );
	std::span ChromaUData( const_cast<uint8_t*>(ChromaUDataPtr), ChromaUSize );
	std::span ChromaVData( const_cast<uint8_t*>(ChromaVDataPtr), ChromaVSize );

	
	//	look out for striped data and make a single pixel buffer
	if ( !LumaData.empty() && !ChromaUData.empty() && ChromaVData.empty() )
	{
		SoyPixelsMeta YuvMeta(Width, Height, SoyPixelsFormat::Yuv_8_88);
		BufferArray<SoyPixelsMeta, 3> YuvMetas;
		YuvMeta.GetPlanes(GetArrayBridge(YuvMetas));
		
		auto ChromaUVData = ChromaUData;
		auto* StripedLumaData = LumaData.data();
		auto* StripedChromaUVData = LumaData.data() + YuvMetas[0].GetDataSize();
		if ( StripedLumaData==LumaData.data() && StripedChromaUVData==ChromaUVData.data() )
		{
			auto TotalSize = LumaSize + ChromaUVData.size();
			SoyPixelsRemote PixelsStriped( StripedLumaData, TotalSize, YuvMeta );
			mEncoder->Encode( PixelsStriped, Meta, Keyframe );
			return;
		}
		
		//	gr: we don't support 2 planes below, so take a slow path and stripe the data
		std::Debug << "Warning, slow path: 2 plane YUV turning into striped YUV data" << std::endl;
		std::vector<uint8_t> StripedData;
		std::copy( LumaData.begin(), LumaData.end(), std::back_inserter(StripedData) );
		std::copy( ChromaUVData.begin(), ChromaUVData.end(), std::back_inserter(StripedData) );
		SoyPixelsRemote PixelsStriped( StripedData.data(), StripedData.size(), YuvMeta );
		mEncoder->Encode( PixelsStriped, Meta, Keyframe );
		return;
	}

	
	
	
	//	check for special case of 1 plane
	if ( !LumaData.empty() && ChromaUData.empty() && ChromaVData.empty() )
	{
		auto PixelFormat = SoyPixelsFormat::Greyscale;
		//	only one plane
		if (!FormatName.empty())
		{
			//	todo: check for fourccs
			auto PixelFormatMaybe = magic_enum::enum_cast<SoyPixelsFormat::Type>(FormatName);
			if (PixelFormatMaybe.has_value())
			{
				PixelFormat = *PixelFormatMaybe;
			}
			else
			{
				//	gr: magic_enum currenty isnt working
				PixelFormat = SoyPixelsFormat::ToType(FormatName);
				if (PixelFormat == SoyPixelsFormat::Invalid)
					throw Soy::AssertException(std::string("Unrecognised pixel format ") + FormatName);
			}
		}

		SoyPixelsRemote LumaPixels( LumaData.data(), Width, Height, LumaData.size(), PixelFormat );
		
		#if defined(TARGET_IOS)
		{
			//	gr: ios seems to encode this pure grey, so pad with fake buffers;
			//		osx seems to encode greyscale without fuss?
			//	gr: that Encode() isn't implemented, turn into yuv
			if ( PixelFormat == SoyPixelsFormat::Greyscale )
			{
				SoyPixels YuvPixels(LumaPixels);
				YuvPixels.SetFormat( SoyPixelsFormat::Yuv_8_88 );
				mEncoder->Encode( YuvPixels, Meta, Keyframe);
				return;
				/*
				//	gr: ios seems to encode this pure grey, so pad with fake buffers;
				#if defined(TARGET_IOS)
				{
					SoyPixelsMeta ChromaPlaneMeta( Width/2, Height/2, SoyPixelsFormat::Greyscale );
					auto& ChromaXPlane = GetDummyPixels(ChromaPlaneMeta);
					mEncoder->Encode( LumaPixels, ChromaXPlane, ChromaXPlane, Meta, Keyframe );
				}
				#else
				{
					mEncoder->Encode( LumaPixels, Meta, Keyframe);
				}
				#endif
				*/
			}
		}
		#endif

		mEncoder->Encode( LumaPixels, Meta, Keyframe);
		return;
	}

	if ( !LumaData.empty() && !ChromaUData.empty() && !ChromaVData.empty() )
	{
		SoyPixelsMeta YuvMeta(Width, Height, SoyPixelsFormat::Yuv_8_8_8);
		
		BufferArray<SoyPixelsMeta, 3> YuvMetas;
		YuvMeta.GetPlanes(GetArrayBridge(YuvMetas));
		auto WidthChroma = YuvMetas[1].GetWidth();
		auto HeightChroma = YuvMetas[1].GetHeight();
		
		//	look out for striped data and make a single pixel buffer
		{
			auto* StripedLumaData = LumaData.data();
			auto* StripedChromaUData = LumaData.data() + YuvMetas[0].GetDataSize();
			auto* StripedChromaVData = StripedChromaUData + YuvMetas[1].GetDataSize();
			if ( StripedLumaData==LumaData.data() && StripedChromaUData==ChromaUData.data() && StripedChromaVData==ChromaVData.data() )
			{
				auto TotalSize = LumaSize + ChromaUSize + ChromaVSize;
				SoyPixelsRemote PixelsStriped( LumaData.data(), TotalSize, YuvMeta );
				mEncoder->Encode( PixelsStriped, Meta, Keyframe );
				return;
			}
		}
		
		//	yuv_8_8_8
		SoyPixelsRemote PixelsY( LumaData.data(), Width, Height, LumaSize, SoyPixelsFormat::Luma );
		SoyPixelsRemote PixelsU( ChromaUData.data(), WidthChroma, HeightChroma, ChromaUSize, SoyPixelsFormat::ChromaU_8 );
		SoyPixelsRemote PixelsV( ChromaVData.data(), WidthChroma, HeightChroma, ChromaVSize, SoyPixelsFormat::ChromaV_8 );
		
		mEncoder->Encode( PixelsY, PixelsU, PixelsV, Meta, Keyframe );
		return;
	}
	
	std::stringstream Error;
	Error << "Unable to get a YUV combination from YUV sizes [" << LumaSize << "," << ChromaUSize << "," << ChromaVSize << "]";
	throw Soy::AssertException(Error);
}


void PopH264::TEncoderInstance::PeekPacket(json11::Json::object& Meta)
{
	if (mPackets.IsEmpty())
		return;

	TPacket Packet;
	{
		std::scoped_lock Lock(mPacketsLock);
		auto& Packet0 = mPackets[0];
		Packet = Packet0;
	}
	
	//	write meta
	auto DataSize = Packet.mData ? Packet.mData->size() : 0;
	Meta[POPH264_ENCODEDFRAME_DATASIZE] = static_cast<int>(DataSize);

	if ( !Packet.mError.empty() )
		Meta[POPH264_ENCODEDFRAME_ERROR] = Packet.mError;

	if ( Packet.mEndOfStream )
		Meta[POPH264_ENCODEDFRAME_ENDOFSTREAM] = Packet.mEndOfStream;

	auto EncodeDuration = Packet.mEncodeMeta.GetEncodeDurationMs();
	if ( EncodeDuration.count() != 0 )
		Meta[POPH264_ENCODEDFRAME_ENCODEDURATIONMS] = static_cast<int>(EncodeDuration.count());

	auto InputMeta = Packet.GetInputMeta();
	if ( !InputMeta.empty() )
	{
		using namespace json11;
		//	we're expecting json, so make it an object
		std::string ParseError;
		auto MetaObject = Json::parse( std::string(InputMeta), ParseError );

		//	this shouldn't come up, as we've already parsed it on input, but just in case
		if (!ParseError.empty())
		{
			Meta[POPH264_ENCODEDFRAME_INPUTMETA] = std::string("PopH264 Unexpected parse error ") + ParseError;
		}
		else
		{
			Meta[POPH264_ENCODEDFRAME_INPUTMETA] = MetaObject;
		}
	}
}

size_t PopH264::TEncoderInstance::PeekNextFrameSize()
{
	{
		std::scoped_lock Lock(mPacketsLock);
		if ( !mPackets.IsEmpty() )
		{
			auto& Packet0 = mPackets[0];
			if ( !Packet0.mData )
				return 0;
			return Packet0.mData->size();
		}
	}
	
	return 0;
}

PopH264::TPacket PopH264::TEncoderInstance::PopPacket()
{
	std::scoped_lock Lock(mPacketsLock);
	if ( !this->mPackets.GetSize() )
		throw std::runtime_error("PopH264::TEncoderInstance::PopPacket no packets queued");
		
	auto Packet = mPackets.PopAt(0);
	return Packet;
}

void PopH264::TEncoderInstance::OnNewPacket(TPacket& Packet)
{
	static bool Debug = false;
	if ( Debug )
	{
		try
		{
			auto H264PacketType = H264::GetPacketType( Packet.GetData() );
			std::Debug << __PRETTY_FUNCTION__ << "(" << magic_enum::enum_name(H264PacketType) << ")" << std::endl;
		}
		catch (std::exception& e)
		{
			std::Debug << __PRETTY_FUNCTION__ << " Error getting Nalu packet type; " << e.what() << std::endl;
		}
	}
	
	{
		std::scoped_lock Lock(mPacketsLock);
		
		//	gr: if a packet encode duration wasn't written, write one here with a warning, so the low level encoder is reminded to add it
		Packet.mEncodeMeta.OnEncoded();
		mPackets.PushBack(Packet);
	}
	
	if ( mOnNewPacket )
		mOnNewPacket();
}

void PopH264::TEncoderInstance::AddOnNewFrameCallback(std::function<void()> Callback)
{
	//	does this need to be threadsafe?
	mOnNewPacket = Callback;
}

std::string PopH264::TEncoderInstance::GetEncoderName()
{
	auto Encoder = mEncoder;
	if ( !Encoder )
		return {};

	return Encoder->GetEncoderName();
}
