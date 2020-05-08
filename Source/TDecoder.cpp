#include "TDecoder.h"
#include "SoyH264.h"


PopH264::TDecoder::TDecoder(std::function<void(const SoyPixelsImpl&,size_t)> OnDecodedFrame) :
	mOnDecodedFrame	( OnDecodedFrame )
{
}

void PopH264::TDecoder::OnDecodedFrame(const SoyPixelsImpl& Pixels)
{
	auto FrameNumber = mPendingFrameNumbers.PopAt(0);
	OnDecodedFrame( Pixels, FrameNumber );
}


void PopH264::TDecoder::OnDecodedFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber)
{
	//	check against pending frame numbers?
	mPendingFrameNumbers.Remove(FrameNumber);
	mOnDecodedFrame( Pixels, FrameNumber );
}

void PopH264::TDecoder::OnDecodedEndOfStream()
{
	SoyPixels Null;
	mOnDecodedFrame(Null,0);
}

void PopH264::TDecoder::PushEndOfStream()
{
	//	send an explicit end of stream nalu
	//	todo: overload this for implementation specific flushes
	auto Eos = H264::EncodeNaluByte(H264NaluContent::EndOfStream,H264NaluPriority::Important);
	uint8_t EndOfStreamNalu[]{ 0,0,0,1,Eos };
	auto DataArray = GetRemoteArray( EndOfStreamNalu );
	
	//	mark pending data as finished
	mPendingDataFinished = true;
	Decode( GetArrayBridge(DataArray), 0 );
}

void PopH264::TDecoder::Decode(ArrayBridge<uint8_t>&& PacketData,size_t FrameNumber)
{
	{
		std::lock_guard<std::mutex> Lock(mPendingDataLock);
		mPendingData.PushBackArray(PacketData);
		//	todo: proper data<->number relationships, but we also need to cope with
		//		when we don't have this
		mPendingFrameNumbers.PushBack(FrameNumber);
	}
	
	while ( true )
	{
		//	keep decoding until no more data to process
		if ( !DecodeNextPacket() )
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
	auto* _PendingDataPtr = &mPendingData[mPendingOffset];
	auto PendingDataSize = mPendingData.GetDataSize() - mPendingOffset;
	auto PendingDataArray = GetRemoteArray(_PendingDataPtr, PendingDataSize);

	
	auto DataSize = H264::GetNextNaluOffset( GetArrayBridge(PendingDataArray) );
	//	no next nal yet
	if ( DataSize == 0 )
	{
		if ( !mPendingDataFinished )
			return false;

		//	we're out of data, so pop the remaining data
		DataSize = PendingDataSize;
		
		//	no more data, finished!
		if ( DataSize == 0 )
			return false;
	}
	
	auto* Data = PendingDataArray.GetArray();
	auto DataArray = GetRemoteArray( Data, DataSize );
	
	Buffer.Copy(DataArray);
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



