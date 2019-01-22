#pragma once

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

//	todo: document values with a function that outputs labels!
__export void				GetMeta(int32_t Instance,int32_t* MetaValues,int32_t MetaValuesCount);

//	expecting one frame per packet, split up by highlevel code (mp4 demuxer etc)
__export int32_t			PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber);

//	returns frame number
__export int32_t			PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);

