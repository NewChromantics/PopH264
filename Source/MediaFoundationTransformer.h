#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "SoyFourcc.h"
#include <SoyAutoReleasePtr.h>
#include <SoyPixels.h>

#include <combaseapi.h>
//#include <mfobjects.h>	//	can't forward declare _MFT_MESSAGE_TYPE
#include <mftransform.h>	//	can't forward declare _MFT_MESSAGE_TYPE

#include "json11.hpp"
#include <span>



namespace MediaFoundation
{
	class TTransformer;
	class TActivateMeta;

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


class MediaFoundation::TActivateMeta
{
public:
	TActivateMeta() {}
	TActivateMeta(IMFActivate& Activate);

	std::span<Soy::TFourcc>			GetInputs()		{	return std::span( mInputs.GetArray(), mInputs.GetSize() );	}
	std::span<Soy::TFourcc>			GetOutputs()	{	return std::span( mOutputs.GetArray(), mOutputs.GetSize() );	}

public:
	std::string						mName;
	bool							mHardwareAccelerated = false;
	BufferArray<Soy::TFourcc, 20>	mInputs;
	BufferArray<Soy::TFourcc, 20>	mOutputs;
	Soy::AutoReleasePtr<IMFActivate>	mActivate;
};


class MediaFoundation::TTransformer
{
public:
	TTransformer(TransformerCategory::Type Category,std::span<Soy::TFourcc> InputFormats,std::span<Soy::TFourcc> OutputFormats,bool VerboseDebug);
	~TTransformer();

public:
	//	this returns false if the data was not pushed (where we need to unpop the data, as to not lose it)
	bool			PushFrame(std::span<uint8_t> Data, int64_t FrameNumber);
	void			PushEndOfStream();

	//	returns true if we should call again (ie, there are more frames to get)
	bool			PopFrame(std::vector<uint8_t>& Data,int64_t& FrameNumber,json11::Json::object& Meta,bool& EndOfStream);
	
	void			ProcessCommand(MFT_MESSAGE_TYPE Command);

	IMFMediaType&	GetOutputMediaType();	//	get access to media type to read output meta
	SoyPixelsMeta	GetOutputPixelMeta();

	void			SetOutputFormat(IMFMediaType& MediaType);

	void			SetInputFormat(IMFMediaType& MediaType);
	void			SetInputFormat(Soy::TFourcc Fourcc, std::function<void(IMFMediaType&)> ConfigMedia);
	bool			IsInputFormatReady();
	void			WaitForInputReady(std::chrono::milliseconds Timeout);	//	throw if it never becomes ready

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
	Soy::AutoReleasePtr<IMFTransform>	mTransformer;
private:
public:
	DWORD			mInputStreamId = 0;
	DWORD			mOutputStreamId = 0;
	bool			mInputFormatSet = false;
	bool			mOutputFormatSet = false;
	bool			mInputSentEof = false;
	Soy::AutoReleasePtr<IMFMediaType>	mOutputMediaType;
	json11::Json::object	mOutputMediaMetaCache;		//	reset with mOutputMediaType

	bool			mVerboseDebug = false;

public:
	TActivateMeta		mActivate;		//	gr: this is to store the IMFActivate so we can shutdown, but might be handy for the meta is has already extracted at bootup
};
