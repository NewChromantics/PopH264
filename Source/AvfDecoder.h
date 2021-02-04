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
	class TDecompressor;

	//	same as X264
	class TFrameMeta;
}

class TPixelBuffer;


class Avf::TDecoder : public PopH264::TDecoder
{
public:
	static inline const char*	Name = "Avf";
public:
	TDecoder(std::function<void(const SoyPixelsImpl&,size_t,const json11::Json&)> OnDecodedFrame);
	~TDecoder();
	
private:
	virtual bool	DecodeNextPacket() override;	//	returns true if more data to proccess
	void			AllocDecoder();

	using			PopH264::TDecoder::OnDecodedFrame;	//	reveal inherited versions of OnDecodedFrame when resolving
	void			OnDecodedFrame(TPixelBuffer& PixelBuffer,SoyTime PresentationTime,const json11::Json& Meta);

private:
	std::shared_ptr<TDecompressor>	mDecompressor;
	Array<uint8_t>					mNaluSps;
	Array<uint8_t>					mNaluPps;
};


