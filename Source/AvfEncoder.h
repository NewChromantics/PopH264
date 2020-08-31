#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"

//#include "json11.hpp"
namespace json11
{
	class Json;
}

namespace Avf
{
	class TEncoder;
	class TEncoderParams;
	
	//	platform type (obj-c)
	class TCompressor;
}

class Avf::TEncoderParams
{
public:
	TEncoderParams(){}
	TEncoderParams(json11::Json& Options);
	
	bool	mRealtime = true;
	bool	mMaximisePowerEfficiency = true;

	//	zero means don't apply
	size_t	mAverageKbps = 0;
	size_t	mMaxKbps = 0;
	size_t	mMaxFrameBuffers = 0;
	size_t	mMaxSliceBytes = 0;
	size_t	mProfileLevel = 0;
	size_t	mKeyFrameFrequency = 0;
};

class Avf::TEncoder : public PopH264::TEncoder
{
public:
	static inline const char*	Name = "Avf";
	
public:
	TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutputPacket);
	~TEncoder();

	virtual void		Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta,bool Keyframe) override;
	virtual void		Encode(const SoyPixelsImpl& Pixels,const std::string& Meta,bool Keyframe) override;
	virtual void		FinishEncoding() override;
	
private:
	void			AllocEncoder(const SoyPixelsMeta& Meta);
	void			OnPacketCompressed(const ArrayBridge<uint8_t>& Data,size_t FrameNumber);


protected:
	TEncoderParams		mParams;
	std::shared_ptr<TCompressor>	mCompressor;
	SoyPixelsMeta		mPixelMeta;	//	format the compressor is currently setup for
};
