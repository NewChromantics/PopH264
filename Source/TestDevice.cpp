#include "TestDevice.hpp"
#include "SoyLib\src\SoyMedia.h"


auto const TestDeviceName = "Test";

void TestDevice::EnumDeviceNames(std::function<void(const std::string&)> Enum)
{
	Enum(TestDeviceName);
}

std::shared_ptr<TCameraDevice> TestDevice::CreateDevice(const std::string& Name)
{
	if ( Name == TestDeviceName )
	{
		std::shared_ptr<TCameraDevice> Device(new TestDevice);
		return Device;
	}

	return nullptr;
}


void TestDevice::GenerateFrame()
{
	std::shared_ptr<TPixelBuffer> pPixelBuffer(new TDumbPixelBuffer());
	auto& PixelBuffer = dynamic_cast<TDumbPixelBuffer&>(*pPixelBuffer);
	auto& Pixels = PixelBuffer.mPixels;

	//	set the type, alloc pixels, then fill the test planes
	Pixels.mMeta = SoyPixelsMeta(200, 100, SoyPixelsFormat::Yuv_8_88_Full);
	Pixels.mArray.SetSize(Pixels.mMeta.GetDataSize());

	BufferArray<std::shared_ptr<SoyPixelsImpl>,3> Planes;
	Pixels.SplitPlanes(GetArrayBridge(Planes));

	BufferArray<uint8_t, 4> Components;
	Components.PushBack(128);
	Components.PushBack(0);
	Components.PushBack(255);
	Components.PushBack(255);
	for (auto p=0;	p<Planes.GetSize();	p++ )
	{
		auto& Plane = *Planes[p];
		Plane.SetPixels(GetArrayBridge(Components));
	}

	this->PushFrame(pPixelBuffer);
}
	

