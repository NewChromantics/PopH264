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
	void			PopPendingData(ArrayBridge<unsigned char>&& Buffer);
	void			RemovePendingData(size_t Size);
	
protected:
	std::mutex		mPendingDataLock;
	Array<uint8_t>	mPendingData;
};
