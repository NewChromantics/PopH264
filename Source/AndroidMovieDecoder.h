#pragma once

#include "PopMovieDecoder.h"
#include "SoyJava.h"



namespace Android
{
	void	GetMediaFileExtensions(ArrayBridge<std::string>&& Extensions);
}


class TSurfaceTexture
{
public:
	TSurfaceTexture(Opengl::TContext& Context,SoyPixelsMeta DesiredBufferMeta,Soy::TSemaphore* Sempahore,bool SingleBufferMode);	//	if semaphore passed, we block for the texture creation
	~TSurfaceTexture();

	//	must be called in opengl thread
	bool				Update(SoyTime& Timestamp,bool& Changed);	//	returns success/fail
	
	bool				IsValid() const;
	Opengl::TTexture&	GetTexture() 		{	return mTexture;	}//	eos texture

private:

public:
	std::shared_ptr<JSurfaceTexture>	mSurfaceTexture;
	std::shared_ptr<JSurface>			mSurface;
	Opengl::TTexture	mTexture;	//	actual texture used on surface texture
	SoyTime				mCurrentContentsTimestamp;
};



class TSurfacePixelBuffer : public TPixelBuffer
{
public:
	TSurfacePixelBuffer(std::shared_ptr<TSurfaceTexture>& SurfaceTexture) :
		mSurfaceTexture	( SurfaceTexture )
	{
	}
	
	virtual void		Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform) override;
	virtual void		Lock(ArrayBridge<Directx::TTexture>&& Textures,Directx::TContext& Context,float3x3& Transform) override	{}
	virtual void		Lock(ArrayBridge<Metal::TTexture>&& Textures,Metal::TContext& Context,float3x3& Transform) override	{}
	virtual void		Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform) override;
	virtual void		Unlock() override;
	
public:
	std::shared_ptr<TSurfaceTexture>	mSurfaceTexture;
};





class Platform::TMediaFormat
{
public:
	TMediaFormat(const JniMediaFormat& Format) :
		mFormat		( Format )
	{
	}
	
	JniMediaFormat	mFormat;
};




class AndroidMediaExtractor : public TMediaExtractor
{
public:
	AndroidMediaExtractor(const TMediaExtractorParams& Params);
	~AndroidMediaExtractor();
	
	virtual void		GetStreams(ArrayBridge<TStreamMeta>&& Streams) override;
	virtual std::shared_ptr<Platform::TMediaFormat>	GetStreamFormat(size_t StreamIndex) override;

protected:
	virtual std::shared_ptr<TMediaPacket>	ReadNextPacket() override;
	
private:
	Array<TStreamMeta>					mStreams;
	std::shared_ptr<JniMediaExtractor>	mExtractor;
	std::shared_ptr<TJniObject>			mJavaBuffer;
	std::shared_ptr<JSurface>			mSurface;
	bool								mDoneInitialAdvance;
};


class AndroidEncoderBuffer : public TPixelBuffer
{
public:
	AndroidEncoderBuffer(int OutputBufferIndex,const std::shared_ptr<TJniObject>& Codec,const std::shared_ptr<TSurfaceTexture>& Surface);
	AndroidEncoderBuffer(int OutputBufferIndex,const std::shared_ptr<TJniObject>& Codec,const std::shared_ptr<SoyPixelsImpl>& ByteBufferPixels);
	~AndroidEncoderBuffer();

	virtual void		Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform) override;
	virtual void		Lock(ArrayBridge<Directx::TTexture>&& Textures,Directx::TContext& Context,float3x3& Transform) override		{}
	virtual void		Lock(ArrayBridge<Metal::TTexture>&& Textures,Metal::TContext& Context,float3x3& Transform) override			{}
	virtual void		Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform) override;
	virtual void		Unlock() override;
	
protected:
	void				ReleaseBuffer(bool Render);
	
private:
	int									mOutputBufferIndex;
	std::shared_ptr<TJniObject>			mCodec;
	std::shared_ptr<SoyPixelsImpl>		mByteBufferPixels;
	std::shared_ptr<TSurfaceTexture>	mSurfaceTexture;	//	to access the name of the texture that the surface is bound to. todo; get from codec
};


class AndroidMediaDecoder : public TMediaDecoder
{
public:
	AndroidMediaDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer>& InputBuffer,std::shared_ptr<TPixelBufferManager>& OutputBuffer,std::shared_ptr<Platform::TMediaFormat>& StreamFormat,const TVideoDecoderParams& Params,std::shared_ptr<Opengl::TContext> OpenglContext);
	AndroidMediaDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer>& InputBuffer,std::shared_ptr<TAudioBufferManager>& OutputBuffer,std::shared_ptr<Platform::TMediaFormat>& StreamFormat,const TVideoDecoderParams& Params);
	~AndroidMediaDecoder();
	
	virtual bool		ProcessPacket(const TMediaPacket& Packet) override;

	//	gr: this should be on a seperate thread really, but need to test android threadsafety first
	virtual void		ProcessOutputPacket(TPixelBufferManager& FrameBuffer) override;
	virtual void		ProcessOutputPacket(TAudioBufferManager& FrameBuffer) override;

	virtual bool		IsDecodingFramesInOrder() const override			{	return false;	}
	
private:

	void				Alloc(SoyPixelsMeta SurfaceMeta,std::shared_ptr<Platform::TMediaFormat> Format,std::shared_ptr<Opengl::TContext> OpenglContext,bool SingleBufferMode);
	void				Alloc(SoyPixelsMeta SurfaceMeta,Platform::TMediaFormat& Format,std::shared_ptr<Opengl::TContext> OpenglContext,bool SingleBufferMode);

	void				AllocAudio(Platform::TMediaFormat& Format);

	void				ProcessOutputPacket(std::function<bool(std::shared_ptr<TJniObject>,size_t,int,const TStreamMeta&,SoyTime)> HandleBufferFunc,TMediaBufferManager& Output);
	
public:
	std::atomic<bool>							mCodecStarted;
	std::shared_ptr<TJniObject>					mCodec;
	std::shared_ptr<Platform::TMediaFormat>		mFormatDesc;

	std::shared_ptr<TSurfaceTexture>			mSurfaceTexture;
};


class AndroidAudioDecoder : public TMediaDecoder
{
public:
	AndroidAudioDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer> InputBuffer,std::shared_ptr<TAudioBufferManager> OutputBuffer);
	
	virtual bool		ProcessPacket(const TMediaPacket& Packet) override;
	
	void				ConvertPcmLinear16ToPcmFloat(const ArrayBridge<sint16>&& Input,ArrayBridge<float>&& Output);
};



