#pragma once

#include "Array.hpp"
#include "HeapArray.hpp"
#include "SoyTime.h"
#include <functional>

class SoyPixelsImpl;

namespace PopH264
{
	class TDecoder;
}

class PopH264::TDecoder
{
public:
	TDecoder(std::function<void(const SoyPixelsImpl&,size_t)> OnDecodedFrame);
	
	void			Decode(ArrayBridge<uint8_t>&& PacketData,size_t FrameNumber);

	//	gr: this has a callback because of flushing old packets. Need to overhaul the framenumber<->packet relationship
	void			OnEndOfStream();
	
protected:
	void			OnDecodedFrame(const SoyPixelsImpl& Pixels);
	void			OnDecodedFrame(const SoyPixelsImpl& Pixels,size_t FrameNumber);
	virtual bool	DecodeNextPacket()=0;	//	returns true if more data to proccess
	
	bool			HasPendingData()	{	return !mPendingData.IsEmpty();	}
	bool			PopNalu(ArrayBridge<uint8_t>&& Buffer);
	void			UnpopNalu(ArrayBridge<uint8_t>&& Buffer);

private:
	void			RemovePendingData(size_t Size);
	void			InsertPendingData(ArrayBridge<uint8_t>& Data);	//	insert data (back) to the start
	
private:
	std::mutex		mPendingDataLock;
	size_t			mPendingOffset = 0;		//	to reduce reallocations, we keep an offset where we've read
	bool			mPendingDataFinished = false;	//	when we know we're at EOS
	Array<uint8_t>	mPendingData;
	Array<size_t>	mPendingFrameNumbers;
	std::function<void(const SoyPixelsImpl&,size_t)>	mOnDecodedFrame;
};
