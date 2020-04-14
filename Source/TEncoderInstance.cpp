#include "TEncoderInstance.h"

#define ENABLE_X264

#if defined(ENABLE_X264)
#include "X264Encoder.h"
#endif


PopH264::TEncoderInstance::TEncoderInstance(const std::string& Encoder_)
{
	auto OnOutputPacket = [this](TPacket& Packet)
	{
		this->OnNewPacket(Packet);
	};
	
	std::string Encoder = Encoder_;

#if defined(ENABLE_X264)
	if ( Encoder.empty() )
		Encoder = std::string(X264::TEncoder::NamePrefix);
	
	if ( Soy::StringTrimLeft( Encoder, X264::TEncoder::NamePrefix, false ) )
	{
		//	extract preset
		size_t Preset = X264::TEncoder::DefaultPreset;
		auto PresetString = Encoder;
		if ( !PresetString.empty() )
			Soy::StringToType(Preset,PresetString);
		
		mEncoder.reset( new X264::TEncoder(Preset,OnOutputPacket) );
	}
#endif
	
	//	success!
	if ( mEncoder )
		return;
	
	std::stringstream Error;
	Error << "No encoder supported (requested " << Encoder << ")";
	throw Soy::AssertException(Error);
}


void PopH264::TEncoderInstance::PushFrame(const std::string& Meta,const uint8_t* LumaData,const uint8_t* ChromaUData,const uint8_t* ChromaVData)
{
	std::string ParseError;
	auto Json = json11::Json::parse( Meta, ParseError );
	//	these return 0 if missing
	auto Width = Json["Width"].int_value();
	auto Height = Json["Height"].int_value();
	auto Fail = Json["sgfgfsdgf"].int_value();
	auto LumaSize = Json["LumaSize"].int_value();
	auto ChromaUSize = Json["ChromaUSize"].int_value();
	auto ChromaVSize = Json["ChromaVSize"].int_value();
	
	//	check for data/size mismatch
	if ( LumaData && LumaSize==0 )
		throw Soy::AssertException("Luma pointer but zero LumaSize");
	if ( !LumaData && LumaSize!=0 )
		throw Soy::AssertException("Luma null but LumaSize nonzero");
	
	if ( ChromaUData && ChromaUSize==0 )
		throw Soy::AssertException("ChromaU pointer but zero ChromaUSize");
	if ( !ChromaUData && ChromaUSize!=0 )
		throw Soy::AssertException("ChromaU null but ChromaUSize nonzero");
	
	if ( ChromaVData && ChromaVSize==0 )
		throw Soy::AssertException("ChromaV pointer but zero ChromaVSize");
	if ( !ChromaVData && ChromaVSize!=0 )
		throw Soy::AssertException("ChromaV null but ChromaVSize nonzero");
	
	SoyPixelsMeta YuvMeta(Width,Height,SoyPixelsFormat::Yuv_8_8_8_Ntsc);
	BufferArray<SoyPixelsMeta,3> YuvMetas;
	YuvMeta.GetPlanes(GetArrayBridge(YuvMetas));
	auto WidthChroma = YuvMetas[1].GetWidth();
	auto HeightChroma = YuvMetas[1].GetHeight();

	if ( LumaData && ChromaUData && ChromaVData )
	{
		//	yuv_8_8_8
		SoyPixelsRemote PixelsY( const_cast<uint8_t*>(LumaData), Width, Height, LumaSize, SoyPixelsFormat::Luma_Ntsc );
		SoyPixelsRemote PixelsU( const_cast<uint8_t*>(ChromaUData), WidthChroma, HeightChroma, ChromaUSize, SoyPixelsFormat::ChromaU_8 );
		SoyPixelsRemote PixelsV( const_cast<uint8_t*>(ChromaVData), WidthChroma, HeightChroma, ChromaVSize, SoyPixelsFormat::ChromaV_8 );
		
		mEncoder->Encode( PixelsY, PixelsU, PixelsV, Meta );
		return;
	}
	else if ( LumaData && !ChromaUData && !ChromaVData )
	{
		//	greyscale/luma
		SoyPixelsRemote PixelsY( const_cast<uint8_t*>(LumaData), Width, Height, LumaSize, SoyPixelsFormat::Luma_Ntsc );
		//	need some dummy chroma
		SoyPixels PixelsU( SoyPixelsMeta( WidthChroma, HeightChroma, SoyPixelsFormat::ChromaU_8 ) );
		SoyPixels PixelsV( SoyPixelsMeta( WidthChroma, HeightChroma, SoyPixelsFormat::ChromaV_8 ) );
		mEncoder->Encode( PixelsY, PixelsU, PixelsV, Meta );
		return;
	}
	
	std::stringstream Error;
	Error << "Unable to get a YUV combination from YUV sizes [" << LumaSize << "," << ChromaUSize << "," << ChromaVSize << "]";
	throw Soy::AssertException(Error);
}


void PopH264::TEncoderInstance::PeekPacket(json11::Json::object& Meta)
{
	Soy_AssertTodo();
}

size_t PopH264::TEncoderInstance::PeekNextFrameSize()
{
	{
		std::lock_guard<std::mutex> Lock(mPacketsLock);
		if ( !mPackets.IsEmpty() )
		{
			auto& Packet0 = mPackets[0];
			return Packet0.mData->GetSize();
		}
	}
	
	return 0;
}

void PopH264::TEncoderInstance::PopPacket(ArrayBridge<uint8_t>&& Data)
{
	Soy_AssertTodo();
}

void PopH264::TEncoderInstance::OnNewPacket(TPacket& Packet)
{
	{
		std::lock_guard<std::mutex> Lock(mPacketsLock);
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

