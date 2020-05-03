#pragma once

#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include "Array.hpp"
#include "SoyPixels.h"
#include "SoyH264.h"

class SoyPixelsImpl;

namespace Avf
{
#if defined(__OBJC__)
	CVPixelBufferRef				PixelsToPixelBuffer(const SoyPixelsImpl& Pixels);
	void							GetFormatDescriptionData(ArrayBridge<uint8>&& Data,CMFormatDescriptionRef FormatDesc,size_t ParamIndex);

	//	OSStatus == CVReturn
	void							IsOkay(OSStatus Error,const std::string& Context);
	void							IsOkay(OSStatus Error,const char* Context);
	std::string						GetString(OSStatus Status);
	SoyPixelsFormat::Type			GetPixelFormat(OSType Format);
	SoyPixelsFormat::Type			GetPixelFormat(NSNumber* Format);
	OSType							GetPlatformPixelFormat(SoyPixelsFormat::Type Format);

	CFPtr<CMFormatDescriptionRef>	GetFormatDescriptionH264(const ArrayBridge<uint8_t>& Sps,const ArrayBridge<uint8_t>& Pps,H264::NaluPrefix::Type NaluPrefixType);

#endif
}

