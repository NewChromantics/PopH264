#include "TCameraDevice.hpp"


void TCameraDevice::PushFrame(std::shared_ptr<TPixelBuffer> FramePixelBuffer)
{
	std::lock_guard<std::mutex> Lock(mLastPixelBufferLock);
	mLastPixelBuffer = FramePixelBuffer;
}
