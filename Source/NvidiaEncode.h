#pragma once

#include "TEncoder.h"
#include "SoyPixels.h"

// Only target Jetsons for now

namespace json11
{
	class Json;
}

namespace Nvidia
{
	class TEncoder;
	class TEnchoderParams;

	class TFrameMeta;
}

class Nvidia::TEncoder : public PopH264::TEncoder
{
public:
	static inline const char*	Name = "Nvidia";

public:
	TEncoder(TEnchoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutPacket);

	// Three plane system :=[ ]
	virtual void		Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, context_t& ctx) override;
	virtual void		FinishEncoding() override;

private:
	void			AllocEncoder(const SoyPixelsMeta& Meta);
	void			Encode();
	int 			setCapturePlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height, uint32_t sizeimage)
};