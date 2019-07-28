#pragma once

#if !defined(TARGET_LUMIN)
#error Trying to compile MagicLeap Decoder on non-Lumin target
#endif

#include "TDecoder.h"
#include <ml_media_codec.h>

namespace MagicLeap
{
	class TDecoder;
}


class MagicLeap::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder();
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
private:
	MLHandle		mHandle = ML_INVALID_HANDLE;
};
