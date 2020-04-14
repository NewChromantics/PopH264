#include "TEncoder.h"
#include "Json11/json11.hpp"
#include "SoyPixels.h"

PopH264::TEncoder::TEncoder(std::function<void(TPacket&)> OnOutputPacket) :
	mOnOutputPacket	( OnOutputPacket )
{
	if ( !mOnOutputPacket )
		throw Soy::AssertException("PopH264::TEncoder missing OnOutputPacket");
}


void PopH264::TEncoder::OnOutputPacket(TPacket& Packet)
{
	mOnOutputPacket(Packet);
}


