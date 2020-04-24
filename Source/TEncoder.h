#pragma once

#include "Array.hpp"
#include "HeapArray.hpp"
#include "SoyTime.h"
#include <functional>

class SoyPixelsImpl;


namespace PopH264
{
	class TEncoder;
	class TPacket;
}

class PopH264::TPacket
{
public:
	std::shared_ptr<Array<uint8_t>>	mData;
	std::string						mInputMeta;	//	original input meta json
};


class PopH264::TEncoder
{
public:
	TEncoder(std::function<void(TPacket&)> OnOutputPacket);
	
	virtual void	Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta,bool Keyframe)=0;
	virtual void	FinishEncoding()=0;
	
protected:
	void			OnOutputPacket(TPacket& Packet);

private:
	std::function<void(TPacket&)>	mOnOutputPacket;
};
