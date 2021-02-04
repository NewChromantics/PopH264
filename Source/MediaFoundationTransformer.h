#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "SoyFourcc.h"
#include <SoyAutoReleasePtr.h>
#include <SoyPixels.h>

#include <combaseapi.h>
#include <mftransform.h>	//	can't forward declare _MFT_MESSAGE_TYPE

class IMFAttributes;

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

	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result, const std::string& Context);

	GUID					GetGuid(TransformerCategory::Type Category);
	GUID					GetGuid(Soy::TFourcc Fourcc);
	SoyPixelsFormat::Type	GetPixelFormat(const GUID& Guid);
	SoyPixelsFormat::Type	GetPixelFormat(Soy::TFourcc Fourcc);
	std::string				GetName(const GUID& Guid);	//	get friendly known name of guid
	Soy::TFourcc			GetFourcc(SoyPixelsFormat::Type Format);
	constexpr uint32_t		GetFourcc(const char Str[]);
	Soy::TFourcc			GetFourCC(const GUID& Guid);

	void					EnumAttributes(IMFAttributes& Attributes);
	std::string				GetValue(const PROPVARIANT& Variant, const GUID& Key);
}
std::ostream&	operator<<(std::ostream &out, const PROPVARIANT& in);

class IMFTransform;
class IMFMediaType;

class MediaFoundation::TTransformer
{
public:
	TTransformer(TransformerCategory::Type Category,const ArrayBridge<Soy::TFourcc>&& InputFormats, const ArrayBridge<Soy::TFourcc>&& OutputFormats,bool VerboseDebug);
	~TTransformer();

public:
	//	this returns false if the data was not pushed (where we need to unpop the data, as to not lose it)
	bool			PushFrame(const ArrayBridge<uint8_t>&& Data, int64_t FrameNumber);

	//	returns true if we should call again (ie, there are more frames to get)
	bool			PopFrame(ArrayBridge<uint8_t>&& Data,int64_t& FrameNumber);
	
	void			ProcessCommand(MFT_MESSAGE_TYPE Command);

	IMFMediaType&	GetOutputMediaType();	//	get access to media type to read output meta
	SoyPixelsMeta	GetOutputPixelMeta();

	void			SetOutputFormat(IMFMediaType& MediaType);

	void			SetInputFormat(IMFMediaType& MediaType);
	void			SetInputFormat(Soy::TFourcc Fourcc, std::function<void(IMFMediaType&)> ConfigMedia);
	bool			IsInputFormatReady();

	bool			IsInputFormatSet() { return mInputFormatSet; }
	bool			IsOutputFormatSet() { return mOutputFormatSet; }
	
	void			SetLowLatencyMode(bool Enable);
	void			SetLowPowerMode(bool Enable);
	void			SetDropBadFrameMode(bool Enable);

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
	bool			mOutputFormatSet = false;
	Soy::AutoReleasePtr<IMFMediaType> mOutputMediaType;
	bool			mVerboseDebug = false;

public:
	Array<Soy::TFourcc>	mSupportedInputFormats;
};
