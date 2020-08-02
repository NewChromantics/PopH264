#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"
#include "SoyThread.h"

// Only target Jetsons for now

namespace json11
{
	class Json;
}

namespace Nvidia
{
	class TEncoder;
	class TEncoderParams;

	class TFrameMeta;
	class TNative;
}
class NvVideoEncoder;
class NvV4l2ElementPlane;
class NvBuffer;
struct v4l2_buffer;

class Nvidia::TEncoderParams
{
public:
	TEncoderParams(){}
	TEncoderParams(json11::Json& Options);
};


class Nvidia::TEncoder : public PopH264::TEncoder
{
public:
	static inline const char*	Name = "Nvidia";

public:
	TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutPacket);
	~TEncoder();
	
	virtual void		Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe) override;
	virtual void		Encode(const SoyPixelsImpl& Pixels,const std::string& Meta,bool Keyframe) override;
	virtual void		FinishEncoding() override;

private:
	bool			IsRunning()	{	return true;	}

	void			InitEncoder(SoyPixelsMeta PixelMeta);		//	once we have some meta, set everything up

	void			InitYuvMemoryMode();
	void			InitH264MemoryMode();
	void			InitH264Format(SoyPixelsMeta PixelMeta);
	void			InitYuvFormat(SoyPixelsMeta InputMeta);
	void			InitDmaBuffers(size_t BufferCount);
	void			InitH264Callback();
	void			InitYuvCallback();
	void			InitEncodingParams();
	void			QueueNextYuvBuffer(std::function<void(NvBuffer&)> FillBuffer);
	bool			OnEncodedBuffer(v4l2_buffer&, NvBuffer* buffer,NvBuffer* shared_buffer);
	void			OnFrameEncoded(ArrayBridge<uint8_t>&& FrameData,uint32_t Flags,std::chrono::milliseconds Timestamp);
	void			Sync();
	void			Start();
	void			ReadNextFrame();
	void			WaitForEnd();
	void			Shutdown();

	
	//	nvidia has awkward names for these so we have a helper func
	NvV4l2ElementPlane&			GetYuvPlane();
	NvV4l2ElementPlane&			GetH264Plane();

	uint32_t					PopYuvUnusedBufferIndex();
	void						PushYuvUnusedBufferIndex(uint32_t Index);

private:
	std::mutex					mYuvBufferLock;
	BufferArray<uint32_t,20>	mYuvBufferIndexesUnused;
	Soy::TSemaphore				mYuvBufferSemaphore;

	NvVideoEncoder*				mEncoder = nullptr;
	std::shared_ptr<TNative>	mNative;
	bool						mInitialised = false;
};
