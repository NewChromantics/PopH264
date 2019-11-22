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


void PopH264::TDecoder::PopPendingData(ArrayBridge<unsigned char>&& Buffer)
{
	std::lock_guard<std::mutex> Lock( mPendingDataLock );
	auto GetNextNalOffset = [this]
	{
		for ( int i=3;	i<mPendingData.GetDataSize();	i++ )
		{
			if ( mPendingData[i+0] != 0 )	continue;
			if ( mPendingData[i+1] != 0 )	continue;
			if ( mPendingData[i+2] != 0 )	continue;
			if ( mPendingData[i+3] != 1 )	continue;
			return i;
		}
		//	assume is complete...
		return (int)mPendingData.GetDataSize();
		//return 0;
	};
	
	auto DataSize = GetNextNalOffset();
	auto* Data = mPendingData.GetArray();
	auto PendingData = GetRemoteArray( Data, DataSize );
	
	Buffer.Copy( PendingData );
	
	mPendingData.RemoveBlock(0, DataSize);
}

void PopH264::TDecoder::RemovePendingData(size_t Size)
{
	std::lock_guard<std::mutex> Lock(mPendingDataLock);
	mPendingData.RemoveBlock(0, Size);
}


