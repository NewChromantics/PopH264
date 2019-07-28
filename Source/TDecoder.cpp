#include "TDecoder.h"



void PopH264::TDecoder::Decode(ArrayBridge<uint8_t>&& PacketData,std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)
{
	{
		std::lock_guard<std::mutex> Lock(mPendingDataLock);
		mPendingData.PushBackArray(PacketData);
	}
	
	while ( true )
	{
		//	keep decoding until no more data to process
		if ( !DecodeNextPacket( OnFrameDecoded ) )
			break;
	}
}

