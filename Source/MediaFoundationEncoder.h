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

	//	these are required by MediaFoundation
	size_t	mAverageKbps = 2000;	//	REALLY high rate gives D3D error for nvidia encoder
	size_t	mProfileLevel = 30;
	bool	mVerboseDebug = false;
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
	virtual void	Encode(const SoyPixelsImpl& Pixels, const std::string& Meta, bool Keyframe) override;
	virtual void	FinishEncoding() override;
	
	void			SetFormat(SoyPixelsMeta ImageMeta);
	void			SetInputFormat(TTransformer& Transformer,SoyPixelsMeta PixelsMeta,Soy::TFourcc InputFormat);
	void			SetOutputFormat(TTransformer& Transformer,SoyPixelsMeta ImageMeta);
	SoyPixelsFormat::Type	GetInputFormat(SoyPixelsFormat::Type Format);

	//	returns true if there are more to try
	bool			FlushOutputFrame();

private:
	TEncoderParams					mParams;
	std::shared_ptr<TTransformer>	mTransformer;
};
