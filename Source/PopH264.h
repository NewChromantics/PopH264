#pragma once

//#include "SoyLib\src\SoyTypes.h"
#include <stdint.h>

#if defined(TARGET_WINDOWS)
#include <SDKDDKVer.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif



#if defined(TARGET_WINDOWS)
#define __export			extern "C" __declspec(dllexport)
#else
#define __export			extern "C"
#endif


__export int32_t			CreateInstance();
__export void				DestroyInstance(int32_t Instance);
