#pragma once

#include "SoyLib\src\SoyTypes.h"

#if defined(TARGET_WINDOWS)
#include <SDKDDKVer.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


#include <cstdint>
#include "Unity/IUnityInterface.h"

#if defined(TARGET_WINDOWS)
#define __export			extern "C" __declspec(dllexport)
#else
#define __export			extern "C"
#endif

__export void				EnumCameraDevices(char* StringBuffer,int32_t StringBufferLength);

__export const char*		PopDebugString();
__export void				ReleaseDebugString(const char* String);
