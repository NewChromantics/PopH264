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
	BufferArray<SoyPixelsImpl*, 10> Textures;
	PixelBuffer->Lock(GetArrayBridge(Textures), Transform);
	try
	{
		BufferArray<std::shared_ptr<SoyPixelsImpl>, 10> Planes;

		//	get all the planes
		for ( auto t = 0; t < Textures.GetSize(); t++ )
		{
			auto& Texture = *Textures[t];
			Texture.SplitPlanes(GetArrayBridge(Planes));
		}

		ArrayBridge<uint8_t>* PlanePixels[] = { &Plane0, &Plane1, &Plane2 };
		for ( auto p = 0; p < Planes.GetSize() && p<3; p++ )
		{
			auto& Plane = *Planes[p];
			auto& PlaneDstPixels = *PlanePixels[p];
			auto& PlaneSrcPixels = Plane.GetPixelsArray();

			auto MaxSize = std::min(PlaneDstPixels.GetDataSize(), PlaneSrcPixels.GetDataSize());
			//	copy as much as possible
			auto PlaneSrcPixelsMin = GetRemoteArray(PlaneSrcPixels.GetArray(), MaxSize);
			PlaneDstPixels.Copy(PlaneSrcPixelsMin);
		}
	}
	catch(...)
	{
		PixelBuffer->Unlock();
		throw;
	}

	return true;
}


