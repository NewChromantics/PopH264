#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"
#include "TDecoder.h"

class SoyPixelsImpl;

extern "C"
{
#include "H264SwDecApi.h"
}
//	todo: implement a TMediaExtractor interface,
//	but for now I just want to expose the C api to match the WASM version (which currently doesnt run in v8, but does on the web)

//	from broadway
//extern "C" void broadwayOnHeadersDecoded();
//extern "C" void broadwayOnPictureDecoded(u8 *buffer, u32 width, u32 height);

namespace Broadway
{
	class TDecoder;
}


class Broadway::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder(std::function<void(const SoyPixelsImpl&,size_t)> OnDecodedFrame);
	~TDecoder();

private:
	void			OnMeta(const H264SwDecInfo& Meta);
	void			OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,SoyTime DecodeDuration);
	virtual bool	DecodeNextPacket() override;	//	returns true if more data to proccess
	
private:
	H264SwDecInst	mDecoderInstance = nullptr;

	//	broadway goes wrong if we try and decode frames without SPS/PPS and wont recover
	//	so, reject other packets until we get them
	bool			mProcessedHeaderPackets = false;
};
