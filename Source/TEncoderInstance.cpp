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
	
	std::stringstream Error;
	Error << "No encoder supported (requested " << Encoder << ")";
	throw Soy::AssertException(Error);
}


void PopH264::TEncoderInstance::PushFrame(const SoyPixelsImpl& Frame,const std::string& Meta)
{
	mEncoder->Encode( Frame, Meta );
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

