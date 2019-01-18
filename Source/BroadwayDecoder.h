#pragma once

#include <functional>
#include "SoyLib/src/Array.hpp"
#include "SoyLib/src/HeapArray.hpp"

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


class Broadway::TDecoder
{
public:
	TDecoder();
	~TDecoder();
	
	void			Decode(ArrayBridge<uint8_t>&& PacketData,std::function<void(const SoyPixelsImpl&)> OnFrameDecoded);

private:
	void			OnMeta(const H264SwDecInfo& Meta);
	void			OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,std::function<void(const SoyPixelsImpl&)> OnFrameDecoded);
	bool			DecodeNextPacket(std::function<void(const SoyPixelsImpl&)> OnFrameDecoded);	//	returns true if more data to proccess
	
public:
	H264SwDecInst	mDecoderInstance = nullptr;
	std::mutex		mPendingDataLock;
	Array<uint8_t>	mPendingData;
};
