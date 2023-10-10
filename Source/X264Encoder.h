#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"
//#include "json11.hpp"

#if defined(TARGET_WINDOWS)
#include "X264/include/x264.h"
//#pragma comment(lib,"libx264.lib")
#elif defined(TARGET_OSX)
#include "X264/osx/x264.h"
#elif defined(TARGET_IOS)
#include "X264/Ios/include/x264.h"
#elif defined(TARGET_LINUX)
#include <x264.h>
#endif


namespace json11
{
	class Json;
}

namespace X264
{
	class TEncoder;
	class TEncoderParams;
}


class X264::TEncoderParams
{
public:
	TEncoderParams(){}
	TEncoderParams(json11::Json& Options);
	
	size_t	mPreset = 2;
	size_t	mProfileLevel = 30;

	size_t	mEncoderThreads = 2;
	size_t	mLookaheadThreads = 2;
	bool	mBSlicedThreads = true;
	bool	mEnableLog = false;
	bool	mDeterministic = false;	//	non-deterministic optimisations
	bool	mCpuOptimisations = true;
};

class X264::TEncoder : public PopH264::TEncoder
{
public:
	static inline std::string_view	Name = "x264";
	
public:
	TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutputPacket);
	~TEncoder();

	virtual void		Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe) override;
	virtual void		Encode(const SoyPixelsImpl& Luma,const std::string& Meta,bool Keyframe) override;
	virtual void		FinishEncoding() override;

	virtual std::string	GetEncoderName() override	{	return std::string(Name);	}

	static std::string	GetVersion();

private:
	/*
	void				PushFrame(const SoyPixelsImpl& Pixels, int32_t FrameTime);
	TPacket				PopPacket();
	bool				HasPackets() {	return !mPackets.IsEmpty();	}
	*/
	
private:
	void			AllocEncoder(const SoyPixelsMeta& Meta);
	void			Encode(x264_picture_t* InputPicture);
	
protected:
	SoyPixelsMeta	mPixelMeta;	//	invalid until we've pushed first frame
	x264_t*			mHandle = nullptr;
	x264_param_t	mParam = {0};
	x264_picture_t	mPicture;
	
	TEncoderParams		mParams;
};
