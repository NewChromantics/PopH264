#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "SoyFourcc.h"

namespace MediaFoundation
{
	class TTransformer;

	namespace TransformerCategory
	{
		enum Type
		{
			VideoDecoder,
			VideoEncoder,
		};
	}
}

class IMFTransform;

class MediaFoundation::TTransformer
{
public:
	TTransformer(TransformerCategory::Type Category,const ArrayBridge<Soy::TFourcc>&& InputFormats, const ArrayBridge<Soy::TFourcc>&& OutputFormats);
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
