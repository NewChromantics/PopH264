#pragma once

#include "MediaFoundationTransformer.h"
#include "TDecoder.h"

namespace MediaFoundation
{
	class TDecoder;
}

class IMFTransform;

class MediaFoundation::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder();
	~TDecoder();

private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	
private:
	std::shared_ptr<TTransformer>	mTransformer;
};
