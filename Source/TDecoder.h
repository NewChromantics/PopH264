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
	void			Decode(ArrayBridge<uint8_t>&& PacketData,std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded);
	
protected:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)=0;	//	returns true if more data to proccess
	
	bool			HasPendingData()	{	return !mPendingData.IsEmpty();	}
	bool			PopNalu(ArrayBridge<uint8_t>&& Buffer);
	void			UnpopNalu(ArrayBridge<uint8_t>&& Buffer);

private:
	void			RemovePendingData(size_t Size);
	void			InsertPendingData(ArrayBridge<uint8_t>& Data);	//	insert data (back) to the start
	
private:
	std::mutex		mPendingDataLock;
	size_t			mPendingOffset = 0;		//	to reduce reallocations, we keep an offset where we've read
	Array<uint8_t>	mPendingData;
};
