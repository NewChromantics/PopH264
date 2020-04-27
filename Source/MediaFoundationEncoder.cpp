#include "MediaFoundationEncoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"
#include "SoyFourcc.h"

#include <mfapi.h>
#include <mftransform.h>
#include <codecapi.h>
#include <Mferror.h>

#include <SoyAutoReleasePtr.h>

//	https://github.com/sipsorcery/mediafoundationsamples/blob/master/MFH264RoundTrip/MFH264RoundTrip.cpp

namespace MediaFoundation
{
	void	IsOkay(HRESULT Result, const char* Context);
	void	IsOkay(HRESULT Result,const std::string& Context);
}


MediaFoundation::TEncoder::TEncoder(TEncoderParams Params,std::function<void(PopH264::TPacket&)> OnOutputPacket) :
	PopH264::TEncoder	( OnOutputPacket )
{
	Soy::TFourcc InputFourccs[] = { "NV12" };
	Soy::TFourcc OutputFourccs[] = { "H264" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer(GetArrayBridge(Inputs), GetArrayBridge(Outputs)));
}

MediaFoundation::TEncoder::~TEncoder()
{
	mTransformer.reset();
}

void MediaFoundation::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe)
{
	//	convert to the format required
	//	todo: allow override of base Encode so we dont split & rejoin


	//	pop H264 frames
	try
	{
		PopH264::TPacket Packet;
		Packet.mData.reset(new Array<uint8_t>());
		mTransformer->PopFrame(GetArrayBridge(*Packet.mData));
		if (!Packet.mData->IsEmpty())
		{
			OnOutputPacket(Packet);
		}
	}
	catch (std::exception& e)
	{
		std::Debug << e.what() << std::endl;
	}
}

void MediaFoundation::TEncoder::FinishEncoding()
{

}
