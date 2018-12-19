#pragma once

#include <SoyMediaFoundation.h>
#if defined(ENABLE_DIRECTX)
#include <SoyDirectx.h>
#endif

//	compression example:
//	https://graphics.stanford.edu/~mdfisher/Code/Engine/VideoCompressor.cpp.html

//	https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/419de14a-f08b-46cc-be47-051a2078f5cc/extract-mpeg4-video-clip?forum=mediafoundationdevelopment
//	https://msdn.microsoft.com/en-us/library/ms697548(VS.85).aspx


namespace MediaFoundation
{
	void			GetMediaFileExtensions(ArrayBridge<std::string>&& Extensions);

	TStreamMeta		GetStreamMeta(IMFMediaType& MediaType,bool VerboseDebug);
}


class MfByteStream
{
public:
	MfByteStream(const std::string& Filename);

public:
	std::string						mFilename;
	Soy::AutoReleasePtr<IMFByteStream>	mByteStream;
};



class MFExtractorCallback : public IMFSourceReaderCallback
{
public:
    MFExtractorCallback(MfExtractor& Parent);

    STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
    STDMETHODIMP_(ULONG) AddRef();
    STDMETHODIMP_(ULONG) Release();
    STDMETHODIMP OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex,DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample *pSample);
    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent *);
    STDMETHODIMP OnFlush(DWORD);

private:
	// Destructor is private. Caller should call Release.
	virtual ~MFExtractorCallback()	{}

private:
	MfExtractor&		mParent;
	std::atomic<long>	mRefCount;
};



class MfExtractor : public TMediaExtractor
{
public:
	MfExtractor(const TMediaExtractorParams& Params);
	~MfExtractor();
	
	virtual std::shared_ptr<TMediaPacket>	ReadNextPacket() override;
	virtual void		GetStreams(ArrayBridge<TStreamMeta>&& Streams) override;
	virtual std::shared_ptr<Platform::TMediaFormat>	GetStreamFormat(size_t StreamIndex) override	{	return nullptr;	}

	void				PushPacket(Soy::AutoReleasePtr<IMFSample> Sample,SoyTime Timecode,bool Eof,size_t StreamIndex);
	void				PushPacket(std::shared_ptr<TMediaPacket>& Sample);
	void				TriggerAsyncRead();		//	do a ReadSample to get MF to read
	
protected:
	virtual void		AllocSourceReader(const std::string& Filename)=0;
	virtual void		FilterStreams(ArrayBridge<TStreamMeta>& Streams)	{}
	virtual void		CorrectIncomingTimecode(TMediaPacket& Timecode)		{}

	void				Init();		//	call this from constructor!
	virtual bool		OnSeek() override;
	virtual bool		CanSeekBackwards() override	{	return CanSeek();	}
	virtual bool		CanSeek();

	virtual bool		CanSleep() override;
	
private:
	void				CreateSourceReader(const std::string& Filename);
	void				ConfigureStream(TStreamMeta EnumeratedStreamMeta);
	void				ConfigureVideoStream(TStreamMeta& EnumeratedStreamMeta);
	void				ConfigureAudioStream(TStreamMeta& EnumeratedStreamMeta);
	void				ConfigureOtherStream(TStreamMeta& EnumeratedStreamMeta);

	const TStreamMeta&	GetStreamMeta(size_t StreamIndex);

	void				ProcessPendingSeek();

private:
	std::string			mFilename;

public:
	std::shared_ptr<MediaFoundation::TContext>	mMediaFoundationContext;

	Soy::AutoReleasePtr<IMFSourceReader>		mSourceReader;
	Soy::AutoReleasePtr<MFExtractorCallback>	mSourceReaderCallback;	

	std::map<size_t,TStreamMeta>		mStreams;			//	streams we have configured for output

	std::mutex								mPacketQueueLock;
	Array<std::shared_ptr<TMediaPacket>>	mPacketQueue;

	//	to avoid over-seeking, we store the time of the last sample read
	std::atomic<std::chrono::milliseconds>	mLastReadTime;
	SoyTime								mPendingSeek;

	std::atomic<int>					mAsyncReadSampleRequests;
};


class MfFileExtractor : public MfExtractor
{
public:
	MfFileExtractor(const TMediaExtractorParams& Params) :
		MfExtractor	( Params )
	{
		Init();
	}

	virtual void		AllocSourceReader(const std::string& Filename) override;

public:
	std::shared_ptr<MfByteStream>		mByteStream;
};


class MfPixelBuffer : public TPixelBuffer
{
public:
	MfPixelBuffer(Soy::AutoReleasePtr<IMFSample>& Sample,const TStreamMeta& Meta,bool ApplyHeightPadding,bool ApplyWidthPadding,bool Win7Emulation);
	~MfPixelBuffer();

	virtual void		Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform) override	{}
	virtual void		Lock(ArrayBridge<Directx::TTexture>&& Textures,Directx::TContext& Context,float3x3& Transform) override;
	virtual void		Lock(ArrayBridge<Metal::TTexture>&& Textures,Metal::TContext& Context,float3x3& Transform) override		{}
	virtual void		Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform) override;
	virtual void		Unlock();
	
	//virtual bool		IsDumb() const override			{	return false;	}

private:
	void				GetMediaBuffer(Soy::AutoReleasePtr<IMFMediaBuffer>& Buffer);

	void				LockPixelsMediaBuffer2D(ArrayBridge<SoyPixelsImpl*>& Textures,Soy::AutoReleasePtr<IMFMediaBuffer>& MediaBuffer,float3x3& Transform);	//	try and lock 2D buffer type throws if unsuccessfull
	void				LockPixelsMediaBuffer(ArrayBridge<SoyPixelsImpl*>& Textures,Soy::AutoReleasePtr<IMFMediaBuffer>& MediaBuffer,float3x3& Transform);

	void				ApplyPadding(SoyPixelsMeta& Meta,float3x3& Transform,size_t Pitch,size_t DataSize);

public:
	bool								mApplyHeightPadding;
	bool								mApplyWidthPadding;
	bool								mWin7Emulation;

	Soy::AutoReleasePtr<IMFSample>			mSample;
	TStreamMeta							mMeta;

	Soy::AutoReleasePtr<IMFMediaBuffer>			mLockedMediaBuffer;
	Soy::AutoReleasePtr<IMF2DBuffer>				mLockedBuffer2D;
	Array<std::shared_ptr<SoyPixelsImpl>>	mLockedPixels;
};

