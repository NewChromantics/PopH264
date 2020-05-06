#include "IntelMediaDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyLib/src/SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"

#pragma comment(lib,"libmfx.lib")	
//	libmfx references swscanf_s, which is now inlined
#pragma comment(lib,"legacy_stdio_definitions.lib")

namespace IntelMedia
{
	void	IsOkay(mfxStatus Result,const char* Context);
}

template<typename T>
void MemZero(T& Object)
{
	memset(&Object, 0, sizeof(T));
}


void IntelMedia::IsOkay(mfxStatus Result, const char* Context)
{
	if (Result == MFX_ERR_NONE)
		return;

	std::stringstream Error;
	Error << "IntelMedia error: " << magic_enum::enum_name(Result) << " in " << Context;
	throw Soy::AssertException(Error);
}


IntelMedia::TDecoder::TDecoder(std::function<void(const SoyPixelsImpl&, size_t)> OnDecodedFrame) :
	PopH264::TDecoder	( OnDecodedFrame )
{
	mfxInitParam Params;
	MemZero(Params);
	/*
	//mfxExtThreadsParam threadsPar;
	//mfxExtBuffer* extBufs[1];
	//mfxVersion version;	// real API version with which library is initialized
	MSDK_ZERO_MEMORY(threadsPar);

	// we set version to 1.0 and later we will query actual version of the library which will got leaded
	//initPar.Version.Major = 1;
	initPar.Version.Minor = 0;

	initPar.GPUCopy = pParams->gpuCopy;

	init_ext_buffer(threadsPar);

	bool needInitExtPar = false;

	if (pParams->eDeinterlace)
	{
		m_diMode = pParams->eDeinterlace;
	}

	if (pParams->bUseFullColorRange)
	{
		m_bVppFullColorRange = pParams->bUseFullColorRange;
	}

	if (pParams->nThreadsNum) {
		threadsPar.NumThread = pParams->nThreadsNum;
		needInitExtPar = true;
	}
	if (pParams->SchedulingType) {
		threadsPar.SchedulingType = pParams->SchedulingType;
		needInitExtPar = true;
	}
	if (pParams->Priority) {
		threadsPar.Priority = pParams->Priority;
		needInitExtPar = true;
	}
	if (needInitExtPar) {
		extBufs[0] = (mfxExtBuffer*)&threadsPar;
		initPar.ExtParam = extBufs;
		initPar.NumExtParam = 1;
	}

	// Init session
	if (pParams->bUseHWLib)
	{
		// try searching on all display adapters
		initPar.Implementation = MFX_IMPL_HARDWARE_ANY;

		// if d3d11 surfaces are used ask the library to run acceleration through D3D11
		// feature may be unsupported due to OS or MSDK API version

		if (D3D11_MEMORY == pParams->memType)
			initPar.Implementation |= MFX_IMPL_VIA_D3D11;
			*/
	//Params.Implementation = MFX_IMPL_HARDWARE_ANY;
	Params.Implementation = MFX_IMPL_AUTO_ANY;
	auto Result = MFXInitEx(Params, &mSession);
	IsOkay(Result, "MFXInitEx");
}

IntelMedia::TDecoder::~TDecoder()
{
	if (mSession)
	{
		MFXClose(mSession);
		mSession = nullptr;
	}
}

bool IntelMedia::TDecoder::DecodeNextPacket()
{
	Soy_AssertTodo();
}
