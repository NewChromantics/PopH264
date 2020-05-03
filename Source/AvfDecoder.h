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


class Avf::TDecoder : public PopH264::TDecoder
{
public:
	TDecoder();
	~TDecoder();
	
private:
	virtual bool	DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded) override;	//	returns true if more data to proccess
	void			AllocDecoder();
	
private:
	size_t							mFrameNumber = 0;
	std::shared_ptr<TDecompressor>	mDecompressor;
	Array<uint8_t>					mNaluSps;
	Array<uint8_t>					mNaluPps;
};


