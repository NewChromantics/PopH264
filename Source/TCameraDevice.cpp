#include "TCameraDevice.hpp"
#include "SoyLib\src\SoyMedia.h"


void TCameraDevice::PushFrame(std::shared_ptr<TPixelBuffer> FramePixelBuffer,const SoyPixelsMeta& Meta)
{
	std::lock_guard<std::mutex> Lock(mLastPixelBufferLock);
	mLastPixelBuffer = FramePixelBuffer;
	mLastPixelsMeta = Meta;
}


bool TCameraDevice::PopLastFrame(ArrayBridge<uint8_t>& Plane0, ArrayBridge<uint8_t>& Plane1, ArrayBridge<uint8_t>& Plane2)
{
	std::shared_ptr<TPixelBuffer> PixelBuffer;
	{
		std::lock_guard<std::mutex> Lock(mLastPixelBufferLock);
		PixelBuffer = mLastPixelBuffer;
		mLastPixelBuffer.reset();
	}
	if ( !PixelBuffer )
		return false;

	float3x3 Transform;
	BufferArray<SoyPixelsImpl*, 3> Textures;
	PixelBuffer->Lock(GetArrayBridge(Textures), Transform);
	try
	{
		auto& Texture0 = *Textures[0];
		auto& PixelArray0 = Texture0.GetPixelsArray();
		auto MaxSize = std::min(Plane0.GetDataSize(), PixelArray0.GetDataSize());
		//	copy as much as possible
		auto PixelArray0Min = GetRemoteArray(PixelArray0.GetArray(), MaxSize);
		Plane0.Copy(PixelArray0Min);
	}
	catch(...)
	{
		PixelBuffer->Unlock();
		throw;
	}

	return true;
}


