#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "TDecoder.h"
#include "PopH264.h"

class SoyPixelsImpl;

extern "C"
{
#include "H264SwDecApi.h"
}
//	todo: implement a TMediaExtractor interface,
//	but for now I just want to expose the C api to match the WASM version (which currently doesnt run in v8, but does on the web)

//	from broadway
//extern "C" void broadwayOnHeadersDecoded();
//extern "C" void broadwayOnPictureDecoded(u8 *buffer, u32 width, u32 height, u32 framenumber);

namespace Broadway
{
	class TDecoder;
}


class Broadway::TDecoder : public PopH264::TDecoder
{
public:
	static inline const char*	Name = "Broadway";

public:
	TDecoder(PopH264::TDecoderParams Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError);
	virtual ~TDecoder();

private:
	void			OnMeta(const H264SwDecInfo& Meta);
	void			OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,SoyTime DecodeDuration);
	virtual bool	DecodeNextPacket() override;	//	returns true if more data to proccess
	
private:
	H264SwDecInst	mDecoderInstance = nullptr;

	//	broadway goes wrong if we try and decode frames without SPS/PPS and wont recover
	//	so, reject other packets until we get them
	//	gr: NEED to process sps before pps
	bool			mProcessedSps = false;
	bool			mProcessedPps = false;
	
	//	gr: it seems the decoder gets stuck (no error) if we try and push i-frames before keyframes
	bool			mProcessedKeyframe = false;	
};
