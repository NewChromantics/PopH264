#pragma once

#include <memory>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/SoyPixels.h"
#include "TDecoder.h"
#include <span>

typedef std::chrono::time_point<std::chrono::steady_clock> EventTime_t;



class SoyPixelsImpl;
#include "json11.hpp"

namespace PopH264
{
	class TDecoder;
	class TFrame;
	class TDecoderFrameMeta;
	class TDecoderInstance;

	void		EnumDecoderNames(std::function<void(const std::string&)> EnumDecoderName);
}

namespace json11
{
	class Json;
}

class PopH264::TFrame
{
public:
	std::shared_ptr<SoyPixelsImpl>	mPixels;
	int32_t							mFrameNumber = -1;		//	this may be time, specified by user, so is really just Meta
	bool							mEndOfStream = false;
	json11::Json::object			mMeta;					//	additional (maybe decoder dependent) meta
	SoyTime							mPushedTime;
	SoyTime							mDecodedTime;
	SoyTime							mPoppedTime;
};

//	meta we send back to caller
class PopH264::TDecoderFrameMeta
{
public:
	SoyPixelsMeta			mPixelsMeta;			//	next frame format
	int32_t					mFrameNumber = -1;		//	this may be time, specified by user, so is really just Meta
	bool					mEndOfStream = false;	//	this is the last frame
	size_t					mFramesQueued = 0;		//	frames buffered up
	json11::Json::object	mMeta;					//	additional (maybe decoder dependent) meta
	SoyTime					mDecodedTime;
};

#if defined(_MSC_VER)
#define __exportfunc __declspec(dllexport)
#else
#define __exportfunc
#endif

class PopH264::TDecoderInstance
{
public:
	TDecoderInstance(json11::Json& Options);
	~TDecoderInstance();
	
	void									AddOnNewFrameCallback(std::function<void()> Callback);

	//	input
	void									PushData(std::span<uint8_t> Data,FrameNumber_t FrameNumber);
	void									PushEndOfStream();
	
	//	output
	void									PopFrame(int32_t& FrameNumber,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2);
	__exportfunc bool						PopFrame(TFrame& Frame);
	void									PushFrame(const SoyPixelsImpl& Frame,PopH264::FrameNumber_t FrameNumber,const json11::Json::object& Meta);
	void									OnFatalError(const std::string& Error);
	void									PushErrorFrame(const std::string& Error,FrameNumber_t FrameNumber);
	TDecoderFrameMeta						GetMeta();
	
public:
	std::function<void()>					mOnNewFrame;	//	called when a new frame is pushed
	
private:
	std::shared_ptr<PopH264::TDecoder>		mDecoder;
	std::mutex								mFramesLock;
	Array<TFrame>							mFrames;
	SoyPixelsMeta							mMeta;
};
