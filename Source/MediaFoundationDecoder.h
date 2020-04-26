#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "TDecoder.h"

namespace MediaFoundation
{
	class TDecoder;
}

class IMFTransform;

class MediaFoundation::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder();
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
	void			SetOutputFormat();
	void			ProcessNextOutputPacket();

private:
	IMFTransform*	mDecoder = nullptr;
	DWORD			mInputStreamId = 0;
	DWORD			mOutputStreamId = 0;
};