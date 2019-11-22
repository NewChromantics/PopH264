#pragma once


#include "TDecoder.h"
#include <ml_media_codec.h>
#include "SoyThread.h"
#include "SoyPixels.h"

namespace MagicLeap
{
	class TDecoder;
	class TOutputThread;
}


class MagicLeap::TOutputThread : public SoyWorkerThread
{
public:
	TOutputThread();

	virtual bool	Iteration() override;
	virtual bool	CanSleep() override;
	
	void			PushOnOutputFrameFunc(std::function<void(const SoyPixelsImpl&,SoyTime)>& PushFrameFunc);
	void			OnInputSubmitted(int32_t PresentationTime);
	void			OnOutputBufferAvailible(MLHandle CodecHandle,SoyPixelsMeta PixelFormat,int64_t BufferIndex);
	std::string		GetDebugState();
	
private:
	void			PopOutputBuffer(int64_t OutputBuffer);
	void			PushFrame(const SoyPixelsImpl& Pixels);

private:
	std::mutex		mPushFunctionsLock;
	Array<std::function<void(const SoyPixelsImpl&,SoyTime)>>	mPushFunctions;

	//	list of buffers with some pending output data
	std::mutex		mOutputBuffersLock;
	Array<int64_t>	mOutputBuffers;
	
	MLHandle		mCodecHandle = ML_INVALID_HANDLE;
	SoyPixelsMeta	mPixelFormat;		//	this may need syncing with buffers
};



class MagicLeap::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder(int32_t Mode);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
	void			OnInputBufferAvailible(int64_t BufferIndex);
	void			OnOutputBufferAvailible(int64_t BufferIndex);
	void			OnOutputFormatChanged(MLHandle NewFormat);
	
	std::string		GetDebugState();

private:
	MLHandle		mHandle = ML_INVALID_HANDLE;
	
	uint64_t		mPacketCounter = 0;	//	we don't pass around frame/presentation time, so we just use a counter

	//	list of buffers we can write to
	std::mutex		mInputBufferLock;
	Array<int64_t>	mInputBuffers;
	
	TOutputThread	mOutputThread;
	SoyPixelsMeta	mOutputPixelMeta;
};
