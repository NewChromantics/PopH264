#pragma once

#include <mutex>
#include "SoyLib\src\SoyPixels.h"

class TPixelBuffer;


class TCameraDevice
{
public:
	bool			PopLastFrame(ArrayBridge<uint8_t>& Plane0, ArrayBridge<uint8_t>& Plane1, ArrayBridge<uint8_t>& Plane2);
	SoyPixelsMeta	GetMeta() const { return mLastPixelsMeta; }

protected:
	virtual void	PushFrame(std::shared_ptr<TPixelBuffer> FramePixelBuffer,const SoyPixelsMeta& Meta);

private:
	//	store the last one
	std::mutex						mLastPixelBufferLock;
	std::shared_ptr<TPixelBuffer>	mLastPixelBuffer;
	SoyPixelsMeta					mLastPixelsMeta;
};

