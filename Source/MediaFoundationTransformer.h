#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "SoyFourcc.h"
#include <SoyAutoReleasePtr.h>
#include <SoyPixels.h>

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
class IMFMediaType;

class MediaFoundation::TTransformer
{
public:
	TTransformer(TransformerCategory::Type Category,const ArrayBridge<Soy::TFourcc>&& InputFormats, const ArrayBridge<Soy::TFourcc>&& OutputFormats);
	~TTransformer();

public:
	//	this returns false if the data was not pushed (where we need to unpop the data, as to not lose it)
	bool			PushFrame(const ArrayBridge<uint8_t>&& Data);
	void			PopFrame(ArrayBridge<uint8_t>&& Data,SoyTime& Format);

	IMFMediaType&	GetOutputMediaType();	//	get access to media type to read output meta
	SoyPixelsMeta	GetOutputPixelMeta();

	void			SetOutputFormat(IMFMediaType& MediaType);
	void			SetInputFormat(IMFMediaType& MediaType);
	bool			IsInputFormatReady();

private:
	void			SetOutputFormat();
	void			ProcessNextOutputPacket();
	void			LockTransformer(std::function<void()> Run);

public:
	IMFTransform*	mTransformer = nullptr;
private:
	DWORD			mInputStreamId = 0;
	DWORD			mOutputStreamId = 0;
	bool			mInputFormatSet = false;
	Soy::AutoReleasePtr<IMFMediaType> mOutputMediaType;

public:
	Array<Soy::TFourcc>	mSupportedInputFormats;
};
