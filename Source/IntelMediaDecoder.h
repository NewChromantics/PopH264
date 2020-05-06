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
	TDecoder(std::function<void(const SoyPixelsImpl&, size_t)> OnDecodedFrame);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket() override;
	
private:
	mfxSession	mSession = nullptr;
};
