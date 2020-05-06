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
	TDecoder(std::function<void(const SoyPixelsImpl&, size_t)> OnDecodedFrame);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket() override;
	
	void			SetInputFormat();

private:
	std::shared_ptr<TTransformer>	mTransformer;
};
