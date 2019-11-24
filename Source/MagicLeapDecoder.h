#pragma once


#include "TDecoder.h"
#include <ml_media_codec.h>
#include "SoyThread.h"
#include "SoyPixels.h"

namespace MagicLeap
{
	class TDecoder;
	class TInputThread;
	class TOutputThread;
	class TOutputBufferMeta;
}


class MagicLeap::TOutputBufferMeta
{
public:
	MLMediaCodecBufferInfo	mMeta;
	int64_t					mBufferIndex = -1;
	SoyPixelsMeta			mPixelMeta;			//	meta at time of availibility
};

class MagicLeap::TOutputThread : public SoyWorkerThread
{
public:
	TOutputThread();

	virtual bool	Iteration() override;
	virtual bool	CanSleep() override;
	
	void			PushOnOutputFrameFunc(std::function<void(const SoyPixelsImpl&,SoyTime)>& PushFrameFunc);
	void			OnInputSubmitted(int32_t PresentationTime);
	void			OnOutputBufferAvailible(MLHandle CodecHandle,const TOutputBufferMeta& BufferMeta);
	std::string		GetDebugState();
	
private:
	void			PopOutputBuffer(const TOutputBufferMeta& BufferMeta);
	void			PushFrame(const SoyPixelsImpl& Pixels);

private:
	std::mutex		mPushFunctionsLock;
	Array<std::function<void(const SoyPixelsImpl&,SoyTime)>>	mPushFunctions;

	//	list of buffers with some pending output data
	std::mutex					mOutputBuffersLock;
	Array<TOutputBufferMeta>	mOutputBuffers;
	
	MLHandle		mCodecHandle = ML_INVALID_HANDLE;
};




class MagicLeap::TInputThread : public SoyWorkerThread
{
public:
	TInputThread(std::function<void(ArrayBridge<uint8_t>&&)> PopPendingData,std::function<bool()> HasPendingData);
	
	virtual bool	Iteration() override;
	virtual bool	CanSleep() override;
	
	bool			HasPendingData()	{	return mHasPendingData();	}
	void			OnInputBufferAvailible(MLHandle CodecHandle,int64_t BufferIndex);
	std::string		GetDebugState();
	void			OnInputSubmitted(int32_t PresentationTime)	{}
	
private:
	void			PushInputBuffer(int64_t BufferIndex);
	
private:
	MLHandle		mHandle = ML_INVALID_HANDLE;

	std::function<void(ArrayBridge<uint8_t>&&)>	mPopPendingData;
	std::function<bool()>						mHasPendingData;
	uint64_t		mPacketCounter = 0;	//	we don't pass around frame/presentation time, so we just use a counter
	
	//	list of buffers we can write to
	std::mutex		mInputBuffersLock;
	Array<int64_t>	mInputBuffers;
};



class MagicLeap::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder(int32_t Mode);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
	void			OnInputBufferAvailible(int64_t BufferIndex);
	void			OnOutputBufferAvailible(int64_t BufferIndex,const MLMediaCodecBufferInfo& BufferMeta);
	void			OnOutputFormatChanged(MLHandle NewFormat);
	
	std::string		GetDebugState();
	
private:
	MLHandle		mHandle = ML_INVALID_HANDLE;
	
	TInputThread	mInputThread;
	TOutputThread	mOutputThread;
	SoyPixelsMeta	mOutputPixelMeta;
};
