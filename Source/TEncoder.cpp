#include "TEncoder.h"
#include "Json11/json11.hpp"
#include "SoyPixels.h"
#include "SoyH264.h"

PopH264::TEncoder::TEncoder(std::function<void(TPacket&)> OnOutputPacket) :
	mOnOutputPacket	( OnOutputPacket )
{
	if ( !mOnOutputPacket )
		throw Soy::AssertException("PopH264::TEncoder missing OnOutputPacket");
}


void PopH264::TEncoder::OnOutputPacket(TPacket& Packet)
{
	auto OutputPacket = [&](const ArrayBridge<uint8_t>&& Data)
	{
		TPacket NextPacket;
		NextPacket.mInputMeta = Packet.mInputMeta;
		NextPacket.mData.reset(new Array<uint8_t>());
		NextPacket.mData->Copy(Data);
		mOnOutputPacket(NextPacket);
	};

	//	split up packet if there are multiple nalus
	size_t PrevNalu = 0;
	while (true)
	{
		auto PacketData = GetArrayBridge(*Packet.mData).GetSubArray(PrevNalu);
		auto NextNalu = H264::GetNextNaluOffset(GetArrayBridge(PacketData));
		if (NextNalu == 0)
		{
			//	everything left
			OutputPacket(GetArrayBridge(PacketData));
			break;
		}
		else
		{
			auto SubArray = GetArrayBridge(PacketData).GetSubArray(0, NextNalu);
			OutputPacket(GetArrayBridge(PacketData));
			PrevNalu += NextNalu;
		}
	}
	
}



size_t PopH264::TEncoder::PushFrameMeta(const std::string& Meta)
{
	TEncoderFrameMeta FrameMeta;
	FrameMeta.mFrameNumber = mFrameCount;
	FrameMeta.mMeta = Meta;
	mFrameMetas.PushBack(FrameMeta);
	mFrameCount++;
	return FrameMeta.mFrameNumber;
}

std::string PopH264::TEncoder::GetFrameMeta(size_t FrameNumber)
{
	for (auto i = 0; i < mFrameMetas.GetSize(); i++)
	{
		auto& FrameMeta = mFrameMetas[i];
		if (FrameMeta.mFrameNumber != FrameNumber)
			continue;

		//	gr: for now, sometimes we get multiple packets for one frame, so we can't discard them all
		//auto Meta = mFrameMetas.PopAt(i);
		auto Meta = mFrameMetas[i];
		return Meta.mMeta;
	}

	std::stringstream Error;
	Error << "No frame meta matching frame number " << FrameNumber;
	throw Soy::AssertException(Error);
}
