#pragma once

#include <memory>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/SoyPixels.h"
#include "TDecoder.h"

//	gr: can't seem to forward declare this... would like to keep passing a json object rather than abstract it away,
//		but we could do a EnumMeta() parameter instead which would be good to remove the json dependency
#include "Json11/json11.hpp"


class SoyPixelsImpl;

namespace PopH264
{
	class TEncoder;
}

namespace PopH264
{
	class TInputFrame;		//	merge this with the decoder's frame
	class TPacket;
	class TEncoderInstance;
}


class PopH264::TInputFrame
{
public:
	std::shared_ptr<SoyPixelsImpl>	mPixels;
	SoyTime							mQueueTime;
	std::string						mMetaJson;
};


class PopH264::TPacket
{
public:
	std::shared_ptr<ArrayInterface<uint8_t>>	mData;
	std::string						mInputMeta;			//	original input meta json
	std::chrono::milliseconds		mDecodeDuration;	//	time the last packet that resulted in this picture took to decode
};

class PopH264::TEncoderInstance
{
public:
	TEncoderInstance(const std::string& Encoder);
	
	//	meta
	size_t			GetFrameQueueCount()	{	return mFrames.GetSize();	}
	size_t			GetPacketQueueCount()	{	return mPackets.GetSize();	}

	//	input
	void			PushFrame(const SoyPixelsImpl& Frame,const std::string& Meta);
	
	//	output
	void			PeekPacket(json11::Json::object& Meta);
	size_t			PeekNextFrameSize();
	void			PopPacket(ArrayBridge<uint8_t>&& Data);
	//bool			PopFrame(TFrame& Frame);
	//void			PushFrame(const SoyPixelsImpl& Frame,int32_t FrameNumber,std::chrono::milliseconds DecodeDurationMs);
	
public:
	std::function<void()>	mOnNewPacket;	//	called when a new packet is decoded and ready to be popped
	
private:
	std::shared_ptr<PopH264::TEncoder>		mEncoder;
	
	//	input
	std::mutex				mFramesLock;
	Array<TInputFrame>		mFrames;

	//	output
	std::mutex				mPacketsLock;
	Array<TPacket>			mPackets;
};
