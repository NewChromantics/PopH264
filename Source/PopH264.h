#pragma once

#include <stdint.h>


#if !defined(__export)

#if defined(_MSC_VER) && !defined(TARGET_PS4)
#define __export			extern "C" __declspec(dllexport)
#else
#define __export			extern "C"
#endif

#endif


//	forward declare this c++ class. May need to export the class...
#if defined(__cplusplus)
namespace PopH264
{
	class TDecoderInstance;
	
	enum Mode
	{
		Mode_Software = 0,
		Mode_Hardware = 1,
	};
}
#define EXPORTCLASS	PopH264::TDecoderInstance
#else
#define EXPORTCLASS	void
#endif

__export int32_t			PopH264_GetVersion();

__export int32_t			PopH264_CreateInstance(int32_t Mode);
__export void				PopH264_DestroyInstance(int32_t Instance);

//	for C++ interfaces, to give access to known types and callbacks
//	todo: proper shared_ptr sharing, dllexport class etc. this is essentially unsafe, but caller can manage this between CreateInstance and DestroyInstance
__export EXPORTCLASS*		PopH264_GetInstancePtr(int32_t Instance);

//	todo: document values with a function that outputs labels!
__export void				PopH264_GetMeta(int32_t Instance,int32_t* MetaValues,int32_t MetaValuesCount);

//	expecting one frame per packet, split up by highlevel code (mp4 demuxer etc)
__export int32_t			PopH264_PushData(int32_t Instance,uint8_t* Data,int32_t DataSize,int32_t FrameNumber);

//	returns frame number
__export int32_t			PopH264_PopFrame(int32_t Instance,uint8_t* Plane0,int32_t Plane0Size,uint8_t* Plane1,int32_t Plane1Size,uint8_t* Plane2,int32_t Plane2Size);

