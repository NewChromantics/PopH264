#pragma once

//	if you're using this header to link to the DLL, you'll probbaly need the lib :)
//#pragma comment(lib, "PopH264.lib")

#include <stdint.h>


#if !defined(__export)

#if defined(_MSC_VER) && !defined(TARGET_PS4)
	#define __export			extern "C" __declspec(dllexport)
#else
	#define __export			extern "C"
#endif

#endif



__export int32_t			CreateInstance();
__export void				DestroyInstance(int32_t Instance);

//	todo: document values with a function that outputs labels!
__export void				GetMeta(int32_t Instance,int32_t* MetaValues,int32_t MetaValuesCount);

//	expecting one frame per packet, split up by highlevel code (mp4 demuxer etc)
__export int32_t			PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber);

//	returns frame number
__export int32_t			PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);

