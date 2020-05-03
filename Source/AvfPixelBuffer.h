#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <map>
#include "SoyMedia.h"	//	TPixelBuffer

#if defined(__OBJC__)
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#endif

class AvfMovieDecoder;
class AvfDecoderRenderer;
class AvfTextureCache;



namespace Platform
{
	std::string				GetCVReturnString(CVReturn Error);
	std::string				GetCodec(CMFormatDescriptionRef FormatDescription);
	std::string				GetExtensions(CMFormatDescriptionRef FormatDescription);
}


namespace Directx
{
	class TContext;
	class TTexture;
}

namespace Metal
{
	class TContext;
}

#if defined(__OBJC__)
class AvfTextureCache
{
public:
	AvfTextureCache(Metal::TContext* MetalContext);
	~AvfTextureCache();
	
	void				Flush();
	void				AllocOpengl();
	void				AllocMetal(Metal::TContext& MetalContext);
	
#if defined(ENABLE_METAL)
	CFPtr<CVMetalTextureCacheRef>		mMetalTextureCache;
#endif
#if defined(TARGET_IOS)
	CFPtr<CVOpenGLESTextureCacheRef>	mOpenglTextureCache;
#elif defined(TARGET_OSX)
	CFPtr<CVOpenGLTextureCacheRef>		mOpenglTextureCache;
#endif
};
#endif



class AvfDecoderRenderer
{
public:
	AvfDecoderRenderer()
	{
	}
	~AvfDecoderRenderer()
	{
		mTextureCaches.Clear();
	}
	
	std::shared_ptr<AvfTextureCache>	GetTextureCache(size_t Index,Metal::TContext* MetalContext);
	
public:
	Array<std::shared_ptr<AvfTextureCache>>	mTextureCaches;
};









#if defined(__OBJC__)
class AvfPixelBuffer : public TPixelBuffer
{
public:
	AvfPixelBuffer(bool DoRetain,std::shared_ptr<AvfDecoderRenderer>& Decoder,const float3x3& Transform) :
		mDecoder		( Decoder ),
		mLockedPixels	( 2 ),
		mReadOnlyLock	( true ),
		mTransform		( Transform )
	{
	}
	~AvfPixelBuffer();
	
	virtual void			Lock(ArrayBridge<Directx::TTexture>&& Textures,Directx::TContext& Context,float3x3& Transform) override		{}
	virtual void			Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform) override;
	virtual void			Lock(ArrayBridge<Metal::TTexture>&& Textures,Metal::TContext& Context,float3x3& Transform) override;
	virtual void			Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform) override;
	virtual void			Unlock() override;
	
	//virtual bool			IsDumb() const override			{	return false;	}

	void					WaitForUnlock();
	
private:
	virtual CVImageBufferRef	LockImageBuffer()=0;
	virtual void				UnlockImageBuffer()=0;
	void						LockPixels(ArrayBridge<SoyPixelsImpl*>& Planes,void* _Data,size_t BytesPerRow,SoyPixelsMeta Meta,float3x3& Transform,ssize_t DataSize=-1);
	
protected:
	bool						mReadOnlyLock;
	//SoyPixelsFormat::Type		mFormat;
	std::shared_ptr<AvfDecoderRenderer>		mDecoder;
	float3x3					mTransform;
	
	std::mutex					mLockLock;
	
	Array<std::shared_ptr<AvfTextureCache>>	mTextureCaches;
#if defined(TARGET_IOS)
	//	ios has 2 texture caches for multiple planes. Just 0 is used for non-planar
	CFPtr<CVOpenGLESTextureRef>			mLockedTexture0;
	CFPtr<CVOpenGLESTextureRef>			mLockedTexture1;
	CFPtr<CVMetalTextureRef>			mMetal_LockedTexture0;
	CFPtr<CVMetalTextureRef>			mMetal_LockedTexture1;
#elif defined(TARGET_OSX)
	CFPtr<CVOpenGLTextureRef>			mLockedTexture;
#endif
	BufferArray<SoyPixelsRemote,2>		mLockedPixels;
};
#endif

#if defined(__OBJC__)
class CFPixelBuffer : public AvfPixelBuffer
{
public:
	CFPixelBuffer(CMSampleBufferRef Buffer,bool DoRetain,std::shared_ptr<AvfDecoderRenderer>& Decoder,const float3x3& Transform) :
	AvfPixelBuffer	( DoRetain, Decoder, Transform),
	mSample			( Buffer, DoRetain )
	{
		if ( !Soy::Assert( mSample, "Sample expected") )
			return;
		//std::Debug << "CFPixelBuffer() retain count=" << mSample.GetRetainCount() << std::endl;
	}
	~CFPixelBuffer();
	
private:
	virtual CVImageBufferRef	LockImageBuffer() override;
	virtual void				UnlockImageBuffer() override;
	
private:
	CFPtr<CVImageBufferRef>		mLockedImageBuffer;	//	this has been [retained] for safety
	CFPtr<CMSampleBufferRef>	mSample;
};
#endif



#if defined(__OBJC__)
class CVPixelBuffer : public AvfPixelBuffer
{
public:
	CVPixelBuffer(CVPixelBufferRef Buffer,bool DoRetain,std::shared_ptr<AvfDecoderRenderer>& Decoder,const float3x3& Transform) :
		AvfPixelBuffer	( DoRetain, Decoder, Transform ),
		mSample			( Buffer, DoRetain )
	{
		if ( !mSample )
			throw Soy::AssertException("Sample expected");
		auto RetainCount = mSample.GetRetainCount();
	}
	~CVPixelBuffer();
	
private:
	virtual CVImageBufferRef	LockImageBuffer() override;
	virtual void				UnlockImageBuffer() override;
	
private:
	CFPtr<CVPixelBufferRef>	mSample;
};
#endif



