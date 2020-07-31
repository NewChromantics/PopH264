#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"

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
	void			InitEncoder(SoyPixelsMeta PixelMeta);		//	once we have some meta, set everything up

	void			InitYuvMemoryMode();
	void			InitH264MemoryMode();
	void			InitH264Format(SoyPixelsMeta PixelMeta);
	void			InitYuvFormat(SoyPixelsMeta InputMeta);
	void			InitDmaBuffers(size_t BufferCount);
	void			InitH264Callback();
	void			InitYuvCallback();
	void			InitEncodingParams();
	void			Sync();
	void			Start();
	void			ReadNextFrame();
	void			WaitForEnd();
	void			Shutdown();
	
	//	nvidia has awkward names for these so we have a helper func
	NvV4l2ElementPlane&	GetYuvPlane();
	NvV4l2ElementPlane&	GetH264Plane();

	
	NvVideoEncoder*				mEncoder = nullptr;
	std::shared_ptr<TNative>	mNative;
	bool						mInitialised = false;
};
