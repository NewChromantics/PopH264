#pragma once

#include "TCameraDevice.hpp"


class TestDevice : public TCameraDevice
{
public:
	TestDevice()
	{
		GenerateFrame();
	}

	static void								EnumDeviceNames(std::function<void(const std::string&)> Enum);
	static std::shared_ptr<TCameraDevice>	CreateDevice(const std::string& Name);

	void		GenerateFrame();
};
