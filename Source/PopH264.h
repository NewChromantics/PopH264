#pragma once

#include <stdint.h>


#if !defined(__export)

#if defined(_MSC_VER) && !defined(TARGET_PS4)
#define __export			extern "C" __declspec(dllexport)
#else
#define __export			extern "C"
#endif

#endif


//	ditch these for strings, magic leap already has 4
#define POPH264_DECODERMODE_SOFTWARE	0
#define POPH264_DECODERMODE_HARDWARE	1


__export int32_t			PopH264_GetVersion();

//	todo: rename these to CreateDecoder and DestroyDecoder
__export int32_t			PopH264_CreateInstance(int32_t Mode);
__export void				PopH264_DestroyInstance(int32_t Instance);

//	deprecate meta values for json
__export void				PopH264_GetMeta(int32_t Instance, int32_t* MetaValues, int32_t MetaValuesCount);
__export void				PopH264_PeekFrame(int32_t Instance,char* JsonBuffer,int32_t JsonBufferSize);

//	push NALU packets (even fragmented)
//	todo; fix framenumber to mix with fragmented data; for now, if framenumber is important, defragment nalu packets at high level
__export int32_t			PopH264_PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber);

//	returns frame number. -1 on error
__export int32_t			PopH264_PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);


