#pragma once

#include "PopMovieDecoder.h"
#include <SoyCuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>


//	cuvid generic stuff
namespace Cuda
{
	std::string				GetEnumString(cudaVideoChromaFormat Error);
	std::string				GetEnumString(cudaVideoCodec Error);
	SoyPixelsFormat::Type	GetPixelFormat(cudaVideoChromaFormat Format);
	SoyTime					GetTime(CUvideotimestamp Timestamp);
}
std::ostream& operator<<(std::ostream &out,cudaVideoCodec& in);
std::ostream& operator<<(std::ostream &out,cudaVideoChromaFormat& in);
std::ostream& operator<<(std::ostream &out,CUVIDEOFORMAT& in);
std::ostream& operator<<(std::ostream &out,CUVIDPICPARAMS& in);
std::ostream& operator<<(std::ostream &out,CUVIDPARSERDISPINFO& in);


namespace Cuda
{
	class TVideoLock;
	class TDisplayFrame;
};
class CudaVideoDecoder;




class Cuda::TVideoLock
{
public:
	TVideoLock(Cuda::TContext& Context);
	~TVideoLock();
	
	bool	Lock();
	void	Unlock();
	
	CUvideoctxlock	GetLock()		{	return mLock;	}
	
private:
	CUvideoctxlock	mLock;
};




class Cuda::TDisplayFrame : public TPixelBuffer
{
public:
	TDisplayFrame(CUVIDPARSERDISPINFO& DisplayInfo,CudaVideoDecoder& Decoder);

	virtual void		Lock(ArrayBridge<Opengl::TTexture>&& Textures, Opengl::TContext& Context) override;
	virtual void		Lock(ArrayBridge<SoyPixelsImpl*>&& Textures) override;
	virtual void		Unlock() override;

protected:
	void				UpdateFramePixels(ArrayBridge<SoyPixelsImpl*>& Textures,Cuda::TContext& Context);

private:
	CUVIDPARSERDISPINFO	mDisplayInfo;
	CudaVideoDecoder&	mParent;
	bool				mOpenglSupport;
	Array<std::shared_ptr<Cuda::TBuffer>>	mLockedPixelBuffers;
	Array<std::shared_ptr<SoyPixelsRemote>>	mLockedPixels;
};


class CudaVideoDecoder : public TVideoDecoder
{
public:
	CudaVideoDecoder(const TVideoDecoderParams& Params,std::shared_ptr<Cuda::TContext>& Context);
	~CudaVideoDecoder();

	virtual bool	Iteration() override;
	virtual SoyTime	GetDuration() override;

	virtual void	StartMovie(Opengl::TContext& Context) override;	//	also unpause
	virtual bool	PauseMovie(Opengl::TContext& Context) override;	//	return false if unsupported
	virtual void	Shutdown(Opengl::TContext& Context) override;
	
	SoyPixelsMeta	GetTargetMeta()			{	return mTargetMeta;	}
	SoyPixelsMeta	GetDecoderMeta();
	CUvideodecoder		GetDecoder()			{	return mDecoder;	}
	bool				IsStarted();
	std::shared_ptr<Cuda::TStream>	GetStream();
	CUVIDEOFORMAT		GetSourceFormat();
	CUVIDPICPARAMS		GetFrameParams(int FrameIndex);

	std::shared_ptr<Cuda::TBuffer>	GetDisplayFrameBuffer(size_t Index,size_t DataSize);
	std::shared_ptr<Cuda::TBuffer>	GetInteropFrameBuffer(size_t Index,size_t DataSize);
	std::shared_ptr<Cuda::TContext>	GetContext()			{	return mContext;	}

protected:
	void			CreateSource(const TVideoDecoderParams& Params);
	void			CreateDecoder(const TVideoDecoderParams& Params,Cuda::TContext& Context);
	void			CreateParser(const TVideoDecoderParams& Params);
	void			OnVideoPacket(CUVIDSOURCEDATAPACKET& Packet);
	void			OnAudioPacket(CUVIDSOURCEDATAPACKET& Packet);
	void			OnDisplayFrame(CUVIDPARSERDISPINFO& Frame);
	void			OnFrameParams(CUVIDPICPARAMS& Frame);

	cudaVideoCodec		GetCodec();
	size_t				GetMaxDecodeSurfaces();
	size_t				GetMaxOutputSurfaces();

private:
	std::map<int,CUVIDPICPARAMS>	mPicParams;		//	cache of frame data which comes at a different time to decoded frame data
	Array<std::shared_ptr<Cuda::TBuffer>>	mDisplayFrameBuffers;
	Array<std::shared_ptr<Cuda::TBuffer>>	mInteropFrameBuffers;
	std::shared_ptr<Cuda::TStream>	mStream;

	CUvideosource	mSource;
	CUvideoparser	mParser;
	CUvideodecoder	mDecoder;
	std::shared_ptr<Cuda::TVideoLock>	mLock;
	std::shared_ptr<Cuda::TContext>		mContext;
	SoyPixelsMeta	mTargetMeta;

};

