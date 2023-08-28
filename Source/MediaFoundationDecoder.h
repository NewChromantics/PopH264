#pragma once

#include "MediaFoundationTransformer.h"
#include "TDecoder.h"

namespace MediaFoundation
{
	class TDecoder;
}


class MediaFoundation::TDecoder : public PopH264::TDecoder
{
public:
	static inline const char*	Name = "MediaFoundation";

public:
	TDecoder(PopH264::TDecoderParams& Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError);
	~TDecoder();

private:
	virtual bool	DecodeNextPacket() override;
	size_t			PopFrames();

	void			SetInputFormat(ContentType::Type ContentType);

private:
	std::mutex						mTransformerLock;
	std::shared_ptr<TTransformer>	mTransformer;
	
	//	gr: only push SPS once per stream
	//	todo: handle changing SPS/PPS mid-stream, but currently I think this breaks things a bit
	//		maybe need to reset output format?
	bool			mSpsSet = false;
	bool			mPpsSet = false;
	bool			mSeiSet = false;
};
