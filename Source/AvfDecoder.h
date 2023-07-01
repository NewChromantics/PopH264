#pragma once

#include "TDecoder.h"
#include "SoyPixels.h"

//#include "json11.hpp"
namespace json11
{
	class Json;
}

namespace Avf
{
	class TDecoder;
	
	//	platform type (obj-c)
	class TDecompressor;	//	base class
	class TDecompressorH264;
	class TDecompressorJpeg;

	//	same as X264
	class TFrameMeta;
}

class TPixelBuffer;


class Avf::TDecoder : public PopH264::TDecoder
{
public:
	static inline const char*	Name = "Avf";
public:
	TDecoder(const PopH264::TDecoderParams& Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError);
	~TDecoder();
	
private:
	virtual bool	DecodeNextPacket() override;	//	returns true if more data to proccess
	void			AllocDecoderH264();
	void			AllocDecoderJpeg();

	using			PopH264::TDecoder::OnDecodedFrame;	//	reveal inherited versions of OnDecodedFrame when resolving
	void			OnDecodedFrame(TPixelBuffer& PixelBuffer,PopH264::FrameNumber_t FrameNumber,const json11::Json& Meta);

private:
	std::shared_ptr<TDecompressor>	mDecompressor;
};


