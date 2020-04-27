#include "MediaFoundationDecoder.h"
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


MediaFoundation::TDecoder::TDecoder()
{
	Soy::TFourcc InputFourccs[] = { "H264" };
	Soy::TFourcc OutputFourccs[] = { "NV12" };
	auto Inputs = FixedRemoteArray(InputFourccs);
	auto Outputs = FixedRemoteArray(OutputFourccs);

	mTransformer.reset(new MediaFoundation::TTransformer(GetArrayBridge(Inputs), GetArrayBridge(Outputs)));
}

MediaFoundation::TDecoder::~TDecoder()
{
	mTransformer.reset();
}

bool MediaFoundation::TDecoder::DecodeNextPacket(std::function<void(const SoyPixelsImpl&, SoyTime)> OnFrameDecoded)
{
	Array<uint8_t> Nalu;
	if (!PopNalu(GetArrayBridge(Nalu)))
		return false;

	mTransformer->PushFrame(GetArrayBridge(Nalu));

	//	try and pop frames
	//	todo: other thread
	{
		Array<uint8_t> OutFrame;
		mTransformer->PopFrame(GetArrayBridge(OutFrame));
	}

	return true;
}
