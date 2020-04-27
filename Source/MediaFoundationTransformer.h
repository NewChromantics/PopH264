#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "SoyFourcc.h"

namespace MediaFoundation
{
	class TTransformer;
}

class IMFTransform;

class MediaFoundation::TTransformer
{
public:
	TTransformer(const ArrayBridge<Soy::TFourcc>&& InputFormats, const ArrayBridge<Soy::TFourcc>&& OutputFormats);
	~TTransformer();

public:
	void			PushFrame(const ArrayBridge<uint8_t>&& Data);
	void			PopFrame(const ArrayBridge<uint8_t>&& Data);

private:
	void			SetOutputFormat();
	void			ProcessNextOutputPacket();

private:
	IMFTransform*	mTransformer = nullptr;
	DWORD			mInputStreamId = 0;
	DWORD			mOutputStreamId = 0;
};
