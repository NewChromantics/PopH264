#pragma once

#include <memory>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/SoyPixels.h"

class SoyPixelsImpl;
namespace Broadway
{
	class TDecoder;
}


class TFrame
{
public:
	std::shared_ptr<SoyPixelsImpl>	mPixels;
	int32_t							mFrameNumber;	//	this may be time, specified by user, so is really just Meta
};

class TDecoderInstance
{
public:
	TDecoderInstance();
	
	//	input
	void									PushData(const uint8_t* Data,size_t DataSize,int32_t FrameNumber);
	
	//	output
	void									PopFrame(int32_t& FrameNumber,ArrayBridge<uint8_t>&& Plane0,ArrayBridge<uint8_t>&& Plane1,ArrayBridge<uint8_t>&& Plane2);
	bool									PopFrame(TFrame& Frame);
	void									PushFrame(const SoyPixelsImpl& Frame,int32_t FrameNumber);
	const SoyPixelsMeta&					GetMeta() const	{	return mMeta;	}
	
private:
	std::shared_ptr<Broadway::TDecoder>		mDecoder;
	std::mutex								mFramesLock;
	Array<TFrame>							mFrames;
	SoyPixelsMeta							mMeta;
};
