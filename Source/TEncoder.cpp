#include "TEncoder.h"


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

/*
PopH264::TPacket PopH264::TEncoder::PopPacket()
{
	std::lock_guard<std::mutex> Lock(mPacketsLock);
	if (mPackets.IsEmpty())
		throw TNoFrameException();
	
	auto Packet = mPackets.PopAt(0);
	return Packet;
}
*/
