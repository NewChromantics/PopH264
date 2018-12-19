#pragma once

#include <mutex>

class TPixelBuffer;
class SoyPixelsImpl;


class TCameraDevice
{
public:
	void			PopLastFrame(SoyPixelsImpl& Target);

protected:
	virtual void	PushFrame(std::shared_ptr<TPixelBuffer> FramePixelBuffer);

private:
	//	store the last one
	std::mutex						mLastPixelBufferLock;
	std::shared_ptr<TPixelBuffer>	mLastPixelBuffer;
};

