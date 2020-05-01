#pragma once

#include "MediaFoundationTransformer.h"
#include "TEncoder.h"

//#include "json11.hpp"
namespace json11
{
	class Json;
}

namespace MediaFoundation
{
	class TEncoder;
	class TEncoderParams;
}

class IMFTransform;


class MediaFoundation::TEncoderParams
{
public:
	TEncoderParams() {}
	TEncoderParams(json11::Json& Options);

	size_t	mQuality = 100;

	//	zero means don't apply
	size_t	mAverageKbps = 0;
	size_t	mProfileLevel = 0;
};




class MediaFoundation::TEncoder : public PopH264::TEncoder
{
public:
	static inline const char*	Name = "MediaFoundation"; 
public:
	TEncoder(TEncoderParams Params,std::function<void(PopH264::TPacket&)> OnOutputPacket);
	~TEncoder();

private:
	virtual void	Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe) override;
	virtual void	FinishEncoding() override;
	
	void			SetInputFormat(SoyPixelsFormat::Type PixelFormat);

private:
	TEncoderParams					mParams;
	std::shared_ptr<TTransformer>	mTransformer;
};
