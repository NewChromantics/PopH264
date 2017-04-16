#pragma once

#include <cstdint>
#include "Unity/IUnityInterface.h"

UNITY_INTERFACE_EXPORT int32_t			EncodeJpeg(uint8_t* JpegData,int32_t JpegDataSize,int32_t JpegQuality,uint8_t* ImageData,int32_t ImageDataSize);

UNITY_INTERFACE_EXPORT const char*		PopDebugString();
UNITY_INTERFACE_EXPORT void				ReleaseDebugString(const char* String);
