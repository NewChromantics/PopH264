#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "TDecoder.h"
#include "mfxvideo.h"

namespace IntelMedia
{
	class TDecoder;
}


class IntelMedia::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder();
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
private:
	mfxSession	mSession = nullptr;
};
