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
__export int32_t			CreateCameraDevice(const char* Name);
__export void				FreeCameraDevice(int32_t Instance);
__export void				GetMeta(int32_t Instance,int32_t* MetaValues,int32_t MetaValuesCount);
__export int32_t			PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);


__export const char*		PopDebugString();
__export void				ReleaseDebugString(const char* String);
