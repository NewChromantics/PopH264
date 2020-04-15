#pragma once

#include "SoyLib/src/SoyH264.h"


//	gr; these should be in SoyH264, but that currently ends up with too many dependencies
namespace H264
{
	size_t					GetNaluLength(const ArrayBridge<uint8_t>& Data);
	inline size_t			GetNaluLength(const ArrayBridge<uint8_t>&& Data) { return GetNaluLength(Data); }
	H264NaluContent::Type	GetPacketType(const ArrayBridge<uint8_t>&& Data);
}

