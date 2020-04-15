#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"


namespace Avf
{
	class TEncoder;
	
	//	platform type (obj-c)
	class TCompressor;

	//	same as X264
	class TFrameMeta;
}


//	same as X264
class Avf::TFrameMeta
{
public:
	size_t		mFrameNumber = 0;
	std::string	mMeta;
};

class Avf::TEncoder : public PopH264::TEncoder
{
public:
	static inline const char*	NamePrefix = "Avf";
	
public:
	TEncoder(std::function<void(PopH264::TPacket&)> OnOutputPacket);
	~TEncoder();

	virtual void		Encode(const SoyPixelsImpl& Luma,const SoyPixelsImpl& ChromaU,const SoyPixelsImpl& ChromaV,const std::string& Meta) override;
	virtual void		FinishEncoding() override;
	
private:
	void			AllocEncoder(const SoyPixelsMeta& Meta);
	void			OnPacketCompressed(const ArrayBridge<uint8_t>& Data,SoyTime PresentationTime);

	//	returns frame number used as PTS and stores meta
	size_t			PushFrameMeta(const std::string& Meta);
	//	gr: SOME frames will yield multiple packets (eg SPS & PPS) so some we need to keep around...
	//		gotta work out a way to figure out what we can discard
	std::string		GetFrameMeta(size_t FrameNumber);
	
protected:
	std::shared_ptr<TCompressor>	mCompressor;
	SoyPixelsMeta		mPixelsMeta;
	
	size_t				mFrameCount = 0;
	Array<TFrameMeta>	mFrameMetas;
};
