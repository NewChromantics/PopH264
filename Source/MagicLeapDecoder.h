#pragma once


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
	
	void			OnInputBufferAvailible(int64_t BufferIndex);
	
private:
	MLHandle		mHandle = ML_INVALID_HANDLE;
	
	uint64_t		mPacketCounter = 0;	//	we don't pass around frame/presentation time, so we just use a counter
	std::mutex		mInputBufferLock;
	Array<int64_t>	mInputBuffers;	//	availible input buffers
};
