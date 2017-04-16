#pragma once

#include <cstdint>
#include "Unity/IUnityInterface.h"

#define __export			extern "C"


__export int32_t			EncodeJpeg(uint8_t* JpegData,int32_t JpegDataSize,int32_t JpegQuality,uint8_t* ImageData,int32_t ImageDataSize,int32_t ImageWidth,int32_t ImageHeight,int32_t ImageComponents);

__export const char*		PopDebugString();
__export void				ReleaseDebugString(const char* String);
