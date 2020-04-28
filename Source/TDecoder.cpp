#include "TDecoder.h"
#include "SoyH264.h"



void PopH264::TDecoder::OnEndOfStream(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)
{
	//	send an explicit end of stream nalu
	//	todo: overload this for implementation specific flushes
	auto Eos = H264::EncodeNaluByte(H264NaluContent::EndOfStream,H264NaluPriority::Important);
	uint8_t EndOfStreamNalu[]{ 0,0,0,1,Eos };
	auto DataArray = GetRemoteArray( EndOfStreamNalu );
	
	//	mark pending data as finished
	mPendingDataFinished = true;
	Decode( GetArrayBridge(DataArray), OnFrameDecoded );
}

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


void PopH264::TDecoder::UnpopNalu(ArrayBridge<uint8_t>&& Buffer)
{
	//	put-back data at the start of the queue
	//	todo: if pending data offset size is > buffersize, can I always just reduce this?
	InsertPendingData(Buffer);
}


bool PopH264::TDecoder::PopNalu(ArrayBridge<uint8_t>&& Buffer)
{
	std::lock_guard<std::mutex> Lock( mPendingDataLock );
	auto* PendingData = &mPendingData[mPendingOffset];
	auto PendingDataSize = mPendingData.GetDataSize()-mPendingOffset;
	
	auto GetNextNalOffset = [&]
	{
		//	todo: handle 001 as well as 0001
		for ( int i=3;	i<PendingDataSize;	i++ )
		{
			if ( PendingData[i+0] != 0 )	continue;
			if ( PendingData[i+1] != 0 )	continue;
			if ( PendingData[i+2] != 0 )	continue;
			if ( PendingData[i+3] != 1 )	continue;
			return i;
		}
		
		//	gr: to deal with fragmented data (eg. udp) we now wait
		//		for the next complete NAL
		//	we should look for EOF packets to handle this though
		//	assume is complete...
		//return (int)mPendingData.GetDataSize();
		return 0;
	};
	
	auto DataSize = GetNextNalOffset();
	//	no next nal yet
	if ( DataSize == 0 )
	{
		if ( !mPendingDataFinished )
			return false;

		//	we're out of data, so pop the remaining data
		DataSize = PendingDataSize;
	}
	
	auto* Data = PendingData;
	auto PendingDataArray = GetRemoteArray( Data, DataSize );
	
	Buffer.Copy( PendingDataArray );
	RemovePendingData( DataSize );
	return true;
}

void PopH264::TDecoder::RemovePendingData(size_t Size)
{
	//	this function is expensive because of giant memmoves when we cut a small amount of data
	//	we should use a RingArray, but for now, have a start offset, and remove when the offset
	//	gets over a certain size

	//	only called from this class, so should be locked
	///std::lock_guard<std::mutex> Lock(mPendingDataLock);
	mPendingOffset += Size;
	static int KbThreshold = 1024 * 5;
	if ( mPendingOffset > KbThreshold * 1024 )
	{
		mPendingData.RemoveBlock(0, mPendingOffset);
		mPendingOffset = 0;
	}
}

void PopH264::TDecoder::InsertPendingData(ArrayBridge<uint8_t>& Data)
{
	std::lock_guard<std::mutex> Lock(mPendingDataLock);

	//	overwrite the used-data part
	//	note: can we just reset this? can we ensure this call is always from unpop?
	uint8_t* Dest = nullptr;
	auto DataSize = Data.GetDataSize();
	if (mPendingOffset >= DataSize)
	{
		mPendingOffset -= DataSize;
		Dest = &mPendingData[mPendingOffset];
	}
	else
	{
		Dest = mPendingData.InsertBlock(mPendingOffset, DataSize);
	}
	memcpy(Dest, Data.GetArray(), DataSize);
}



