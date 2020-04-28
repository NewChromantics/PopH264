#pragma once

#include "Array.hpp"

namespace PopH264
{
	void		GetTestData(const std::string& Name,ArrayBridge<uint8_t>&& Data,size_t& FullSize);
}
