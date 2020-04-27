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
	TEncoderParams(json11::Json& Options)
	{

	}
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

private:
	std::shared_ptr<TTransformer>	mTransformer;
};
