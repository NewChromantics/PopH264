#pragma once

#include "SoyLib/src/SoyH264.h"


//	gr; these should be in SoyH264, but that currently ends up with too many dependencies
namespace H264
{
	namespace NaluPrefix
	{
		enum Type
		{
			AnnexB		= 0,	//	001 or 0001
			Eight		= 1,
			Sixteen		= 2,
			ThirtyTwo	= 4
		};
	}
	
	size_t					GetNaluLength(const ArrayBridge<uint8_t>& Data);
	inline size_t			GetNaluLength(const ArrayBridge<uint8_t>&& Data) { return GetNaluLength(Data); }
	size_t					GetNaluAnnexBLength(const ArrayBridge<uint8_t>& Data);
	inline size_t			GetNaluAnnexBLength(const ArrayBridge<uint8_t>&& Data) { return GetNaluAnnexBLength(Data); }
	H264NaluContent::Type	GetPacketType(const ArrayBridge<uint8_t>&& Data);
	void					ConvertNaluPrefix(ArrayBridge<uint8_t>& Nalu,H264::NaluPrefix::Type NaluSize);
}

