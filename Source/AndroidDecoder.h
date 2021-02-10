#pragma once

#include "TDecoder.h"
#include "SoyThread.h"
#include "SoyPixels.h"


#include "media/NdkMediaCodec.h"
//#include <NdkMediaError.h>
#include "json11.hpp"


//	NDK media formats
typedef AMediaCodecBufferInfo MediaBufferInfo_t;
typedef AMediaCodec* MediaCodec_t;
typedef AMediaFormat* MediaFormat_t;
typedef media_status_t MediaResult_t;


namespace Android
{
	class TDecoder;
	class TInputThread;
	class TOutputThread;
	class TOutputBufferMeta;
	class TOutputTexture;
}


class Android::TOutputBufferMeta
{
public:
	MediaBufferInfo_t		mMeta;
	int64_t					mBufferIndex = -1;
	SoyPixelsMeta			mPixelMeta;			//	meta at time of availibility
};
/*
class Android::TOutputTexture
{
public:
	bool			IsReadyToBePushed()
	{
		if ( mPushed )
			return false;
		if ( mPresentationTime < 0 )
			return false;
		return true;
	}
	
	inline bool		operator==(const MLHandle& Handle) const
	{
		return mTextureHandle == Handle;
	}
	
	//	docs say this is a texture handle that can be used with OES_external
	MLHandle		mTextureHandle = ML_INVALID_HANDLE;
	int64_t			mPresentationTime = -1;	//	if -1, the texture hasn't been written to yet
	bool			mPushed = false;			//	sent to caller
	bool			mReleased = false;			//	released by caller
};
*/
class Android::TOutputThread : public SoyWorkerThread
{
public:
	TOutputThread(PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError);

	virtual bool	Iteration() override;
	virtual bool	CanSleep() override;
	
	void			OnInputSubmitted(int32_t PresentationTime);
	void			OnOutputBufferAvailible(MediaCodec_t Codec,bool AsyncBuffers,const TOutputBufferMeta& BufferMeta);
	std::string		GetDebugState();
	/*
	void			OnOutputTextureWritten(int64_t PresentationTime);
	void			OnOutputTextureAvailible();
*/
private:
	void			PopOutputBuffer(const TOutputBufferMeta& BufferMeta);
	/*
	void			RequestOutputTexture();
	void			PushOutputTextures();
	void			PushOutputTexture(TOutputTexture& OutputTexture);
	void			ReleaseOutputTexture(MLHandle TextureHandle);
	bool			IsAnyOutputTextureReady();
	*/
	void			PushFrame(const SoyPixelsImpl& Pixels,PopH264::FrameNumber_t FrameNumber,const json11::Json& Meta);
	void			OnFrameError(const std::string& Error,PopH264::FrameNumber_t FrameNumber);

private:
	PopH264::OnDecodedFrame_t	mOnDecodedFrame;
	PopH264::OnFrameError_t		mOnFrameError;

	//	list of buffers with some pending output data
	std::mutex					mOutputBuffersLock;
	Array<TOutputBufferMeta>	mOutputBuffers;
/*	
	//	texture's we've acquired
	size_t						mOutputTexturesAvailible = 0;	//	shouldbe atomic
	std::recursive_mutex		mOutputTexturesLock;
	Array<TOutputTexture>		mOutputTextures;
	
	size_t						mOutputTextureCounter = 0;	//	for debug colour output
	*/
	MediaCodec_t				mCodec = nullptr;
	bool						mAsyncBuffers = false;

public:
	json11::Json::object		mOutputMeta;
};




class Android::TInputThread : public SoyWorkerThread
{
public:
	TInputThread(std::function<void(ArrayBridge<uint8_t>&&,PopH264::FrameNumber_t&)> PopPendingData,std::function<bool()> HasPendingData);
	
	virtual bool	Iteration() override	{	return true;	}
	virtual bool	Iteration(std::function<void(std::chrono::milliseconds)> Sleep) override;
	virtual bool	CanSleep() override;
	
	bool			HasPendingData()	{	return mHasPendingData();	}
	void			OnInputBufferAvailible(MediaCodec_t CodecHandle,bool AsyncBuffers,int64_t BufferIndex);
	std::string		GetDebugState();
	void			OnInputSubmitted(int32_t PresentationTime)	{}
	
private:
	void			PushInputBuffer(int64_t BufferIndex);
	
private:
	MediaCodec_t	mCodec = nullptr;
	bool			mAsyncBuffers = false;

	std::function<void(ArrayBridge<uint8_t>&&,PopH264::FrameNumber_t&)>	mPopPendingData;
	std::function<bool()>						mHasPendingData;
	
	//	list of buffers we can write to
	std::mutex		mInputBuffersLock;
	Array<int64_t>	mInputBuffers;
};



class Android::TDecoder : public PopH264::TDecoder
{
public:
	static inline const char*	Name = "Android";
public:
	TDecoder(PopH264::TDecoderParams Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket() override;	//	returns true if more data to proccess
	
	void			OnInputBufferAvailible(int64_t BufferIndex);
	void			OnOutputBufferAvailible(int64_t BufferIndex,const MediaBufferInfo_t& BufferMeta);
	void			OnOutputFormatChanged(MediaFormat_t NewFormat);
	//void			OnOutputTextureWritten(int64_t PresentationTime);
	//void			OnOutputTextureAvailible();

	std::string		GetDebugState();

	//std::shared_ptr<Platform::TMediaFormat>		AllocFormat();
	//void			Alloc(SoyPixelsMeta SurfaceMeta,std::shared_ptr<Platform::TMediaFormat> Format,std::shared_ptr<Opengl::TContext> OpenglContext,bool SingleBufferMode);

	void			CreateCodec();		//	returns false if we're not ready to push packets (ie, waiting for headers still)
	void 			DequeueOutputBuffers();
	void			DequeueInputBuffers();
	//	input thread pulling data
	void			GetNextInputData(ArrayBridge<uint8_t>&& PacketBuffer,PopH264::FrameNumber_t& FrameNumber);

private:
	//	need SPS & PPS to create format, before we can create codec
	Array<uint8_t>	mPendingSps;
	Array<uint8_t>	mPendingPps;	
	//std::shared_ptr<JniMediaFormat>		mFormat;	//	format for codec!
	MediaCodec_t	mCodec = nullptr;
	bool			mAsyncBuffers = false;
	PopH264::TDecoderParams	mParams;
	//std::shared_ptr<TSurfaceTexture>	mSurfaceTexture;
	
	std::function<void()>	mOnStartThread;
	TInputThread			mInputThread;
	TOutputThread			mOutputThread;
	SoyPixelsMeta			mOutputPixelMeta;
	json11::Json::object	mOutputMeta;
};
