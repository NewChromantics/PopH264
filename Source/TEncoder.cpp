#include "TEncoder.h"
#include "Json11/json11.hpp"
#include "SoyPixels.h"
#include "SoyH264.h"

PopH264::TEncoder::TEncoder(std::function<void(TPacket&)> OnOutputPacket) :
	mOnOutputPacket	( OnOutputPacket )
{
	if ( !mOnOutputPacket )
		throw Soy::AssertException("PopH264::TEncoder missing OnOutputPacket");
}


void PopH264::TEncoder::OnOutputPacket(TPacket& Packet)
{
	auto OutputPacket = [&](std::span<uint8_t> Data)
	{
		TPacket NextPacket;
		NextPacket.mEncodeMeta = Packet.mEncodeMeta;
		NextPacket.mData.reset(new std::vector<uint8_t>());
		std::copy( Data.begin(), Data.end(), std::back_inserter( *NextPacket.mData ) );
		mOnOutputPacket(NextPacket);
	};

	H264::SplitNalu( Packet.GetData(), OutputPacket );
}

void PopH264::TEncoder::OnError(std::string_view Error)
{
	TPacket Packet;
	Packet.mError = Error;
	mOnOutputPacket( Packet );
	mHasOutputError = true;
}

void PopH264::TEncoder::OnFinished()
{
	TPacket Packet;
	Packet.mEndOfStream = true;
	mOnOutputPacket( Packet );
	mHasOutputEndOfStream = true;
}





PopH264::FrameNumber_t PopH264::TEncoder::PushFrameMeta(const std::string& Meta)
{
	auto NewFrameNumber = mFrameCount;
	mFrameCount++;
	
	TEncoderFrameMeta FrameMeta;
	FrameMeta.mInputMeta = Meta;
	FrameMeta.mPushTime = EventTime_t::clock::now();
	mFrameMetas.insert({ NewFrameNumber, FrameMeta });

	return NewFrameNumber;
}

PopH264::TEncoderFrameMeta PopH264::TEncoder::GetFrameMeta(FrameNumber_t FrameNumber)
{
	auto Match = mFrameMetas.find(FrameNumber);
	if ( Match == mFrameMetas.end() )
	{
		std::stringstream Error;
		Error << "No frame meta matching frame number " << FrameNumber;
		throw std::runtime_error(Error.str());
	}
	
	auto& Meta = Match->second;
	return Meta;
}
