#include "BroadwayDecoder.h"
#include <sstream>
#include "SoyLib/src/SoyDebug.h"
#include "SoyLib/src/SoyPixels.h"
#include "SoyH264.h"
#include "MagicEnum/include/magic_enum.hpp"



namespace Broadway
{
	void		IsOkay(H264SwDecRet Result,const char* Context);
	prmem::Heap	Heap(false,false,"Broadway");
}



//	gr: commented these out from H264SwDecApi.c so I can debug
//		(and monitor mem usage)

void H264SwDecTrace(char *string)
{
	std::Debug << "Broadway: " << string << std::endl;
}

void* H264SwDecMalloc(u32 size)
{
	return malloc(size);
	//return Broadway::Heap.Alloc(size);
}

void H264SwDecFree(void *ptr)
{
	free(ptr);
	//	need a free that doesn't know size
	//return Broadway::Heap.Free(ptr);
}

void H264SwDecMemcpy(void *dest, void *src, u32 count)
{
	memcpy(dest, src, count);
}

void H264SwDecMemset(void *ptr, i32 value, u32 count)
{
	memset(ptr, value, count);
}


Broadway::TDecoder::TDecoder()
{
	auto disableOutputReordering = false;
	auto Result = H264SwDecInit( &mDecoderInstance, disableOutputReordering );
	if ( Result != H264SWDEC_OK)
	{
		std::stringstream Error;
		Error << "H264SwDecInit failed: " << Result;
		mDecoderInstance = nullptr;
		throw Soy::AssertException(Error.str());
	}

	/*
	mDecoderInstance.pStream = decoder[i]->byteStrmStart;
	mDecoderInstance.dataLen = strmLen;
	mDecoderInstance.intraConcealmentMethod = 0;
	 */
}

Broadway::TDecoder::~TDecoder()
{
	H264SwDecRelease( mDecoderInstance );
	mDecoderInstance = nullptr;
/*
	if (decoder[i]->foutput)
	fclose(decoder[i]->foutput);
	
	free(decoder[i]->byteStrmStart);
	
	free(decoder[i]);
 */
}

std::string GetDecodeResultString(H264SwDecRet Result)
{
	switch ( Result )
	{
		case H264SWDEC_OK:	return "H264SWDEC_OK";
		case H264SWDEC_STRM_PROCESSED:	return "H264SWDEC_STRM_PROCESSED";
		case H264SWDEC_PIC_RDY:	return "H264SWDEC_PIC_RDY";
		case H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY:	return "H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY";
		case H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY:	return "H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY";
		case H264SWDEC_PARAM_ERR:	return "H264SWDEC_PARAM_ERR";
		case H264SWDEC_STRM_ERR:	return "H264SWDEC_STRM_ERR";
		case H264SWDEC_NOT_INITIALIZED:	return "H264SWDEC_NOT_INITIALIZED";
		case H264SWDEC_MEMFAIL:	return "H264SWDEC_MEMFAIL";
		case H264SWDEC_INITFAIL:	return "H264SWDEC_INITFAIL";
		case H264SWDEC_HDRS_NOT_RDY:	return "H264SWDEC_HDRS_NOT_RDY";
		case H264SWDEC_EVALUATION_LIMIT_EXCEEDED:	return "H264SWDEC_EVALUATION_LIMIT_EXCEEDED";
		default:
		{
			std::stringstream Error;
			Error << "Unhandled H264SwDecRet: " << Result;
			return Error.str();
		}
	}
}


void Broadway::IsOkay(H264SwDecRet Result,const char* Context)
{
	switch ( Result )
	{
		case H264SWDEC_OK:
		case H264SWDEC_STRM_PROCESSED:
		case H264SWDEC_PIC_RDY:
		case H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY:
		case H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY:
			return;
		
		default:
			break;
	}
	
	std::stringstream Error;
	Error << "Broadway error: " << GetDecodeResultString(Result) << " in " << Context;
	throw Soy::AssertException(Error.str());
}

bool IsNal(const uint8_t* pData)
{
	if ( pData[0] != 0 )	return false;
	if ( pData[1] != 0 )	return false;
	if ( pData[2] != 0 )	return false;
	if ( pData[3] != 1 )	return false;
	return true;
}

//	returns true if more data to proccess
bool Broadway::TDecoder::DecodeNextPacket(std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded)
{
	if ( mPendingData.GetDataSize() < 4 )
		return false;
	
	auto StartsWithNal = IsNal(mPendingData.GetArray());
	
	const unsigned IntraGrayConcealment = 0;
	const unsigned IntraReferenceConcealment = 1;
	
	//	if we're getting small packets in, we may not have enough for a frame, but we could wait...
	//	todo: this should be false if our current packet is end of stream
	static bool WaitForNextNal = true;
	
	auto GetNextNalOffset = [this]
	{
		for ( int i=3;	i<mPendingData.GetDataSize();	i++ )
		{
			if ( !IsNal( &mPendingData[i] ) )
				continue;
			return i;
		}
		
		if ( WaitForNextNal )
			return 0;

		//	assume is complete...
		return (int)mPendingData.GetDataSize();
	};
	
	H264SwDecInput Input;
	{
		std::lock_guard<std::mutex> Lock(mPendingDataLock);
		Input.pStream = mPendingData.GetArray();
		Input.dataLen = GetNextNalOffset();
		//Input.dataLen = mPendingData.GetDataSize();
	}
	
	if ( Input.dataLen == 0 )
		return false;

	Input.picId = 0;
	Input.intraConcealmentMethod = IntraGrayConcealment;
	
	H264SwDecOutput Output;
	Output.pStrmCurrPos = nullptr;
	
	static bool Debug = false;
	if ( Debug )
	{
		try
		{
			auto H264PacketType = H264::GetPacketType(GetArrayBridge(mPendingData));
			std::Debug << "H264SwDecDecode(" << magic_enum::enum_name(H264PacketType) << ")" << std::endl;
		}
		catch (std::exception& e)
		{
			std::Debug << "Error getting Nalu packet type; " << e.what() << std::endl;
		}
	}
	
	Soy::TScopeTimerPrint Timer("H264 Decode",15);
	auto Result = H264SwDecDecode( mDecoderInstance, &Input, &Output );
	auto DecodeDuration = Timer.Stop();
	IsOkay( Result, "H264SwDecDecode" );
	
	//	calc what data wasn't used
	auto BytesProcessed = static_cast<ssize_t>(Output.pStrmCurrPos - Input.pStream);
	//	todo: keep this meta for external debugging
	//std::Debug << "H264SwDecDecode result: " << GetDecodeResultString(Result) << ". Bytes processed: "  << BytesProcessed << "/" << Input.dataLen << std::endl;
	
	//	gr: can we delete data here? or do calls below use this data...
	//	gr: still not super clear from API, need to dive into code
	RemovePendingData( BytesProcessed );
	
	auto GetMeta = [&]()
	{
		H264SwDecInfo Meta;
		auto GetInfoResult = H264SwDecGetInfo( mDecoderInstance, &Meta );
		IsOkay( GetInfoResult, "H264SwDecGetInfo" );
		return Meta;
	};
	
	switch( Result )
	{
		case H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY:
		{
			auto Meta = GetMeta();
			OnMeta( Meta );
			return true;
		}
		
		case H264SWDEC_PIC_RDY_BUFF_NOT_EMPTY:
			//	ref code eats data, then falls through...
		//	android just does both https://android.googlesource.com/platform/frameworks/av/+/2b6f22dc64d456471a1dc6df09d515771d1427c8/media/libstagefright/codecs/on2/h264dec/source/EvaluationTestBench.c#158
		case H264SWDEC_PIC_RDY:
		{
			//	gr: this should be fast, it just extracts references from the buffer
			Soy::TScopeTimerPrint PictureTimer("H264 Picture Decode",1);
			//	if no callback, just skip image extraction
			if ( !OnFrameDecoded )
			{
				//return true;
			}
			auto Meta = GetMeta();
			H264SwDecPicture Picture;
			u32 EndOfStream = false;
			while ( true )
			{
				//	decode pictures until we get a non "picture ready" (not sure what we'll get)
				//	gr: this func is free, image already decoded
				auto DecodeResult = H264SwDecNextPicture( mDecoderInstance, &Picture, EndOfStream );
				PictureTimer.Stop();
				IsOkay( Result, "H264SwDecNextPicture" );
				if ( DecodeResult != H264SWDEC_PIC_RDY )
				{
					//	OK just means it's finished
					if ( DecodeResult != H264SWDEC_OK )
						std::Debug << "H264SwDecNextPicture result: " << GetDecodeResultString(DecodeResult) << std::endl;
					break;
				}
				/*
				 picNumber++;
				printf("PIC %d, type %s, concealed %d\n", picNumber,
					   decPicture.isIdrPicture ? "IDR" : "NON-IDR",
					   decPicture.nbrOfErrMBs);
				*/
				//	YuvToRgb( decPicture.pOutputPicture, pRgbPicture );
				OnPicture( Picture, Meta, OnFrameDecoded, DecodeDuration );
			}
			return true;
		}
			
			//	data eaten, no output
		case H264SWDEC_STRM_PROCESSED:
			return true;
		
		default:
		{
			std::Debug << "Unhandled H264SwDecDecode result: " << GetDecodeResultString(Result) << ". Bytes processed: "  << BytesProcessed << "/" << Input.dataLen << std::endl;
			return true;
		}
	}
		
}


void Broadway::TDecoder::OnMeta(const H264SwDecInfo& Meta)
{
	
}

void Broadway::TDecoder::OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,std::function<void(const SoyPixelsImpl&,SoyTime)> OnFrameDecoded,SoyTime DecodeDuration)
{
	//		headers just say
	//	u32 *pOutputPicture;    /* Pointer to the picture, YUV format       */
	auto Format = SoyPixelsFormat::Yuv_8_8_8_Full;
	SoyPixelsMeta PixelMeta( Meta.picWidth, Meta.picHeight, Format );
	//std::Debug << "Decoded picture " << PixelMeta << std::endl;
	
	//	gr: wish we knew exactly how many bytes Picture.pOutputPicture pointed at!
	//		but demos all use this measurement
	//auto DataSize = PixelMeta.GetDataSize();
	auto DataSize = (3 * Meta.picWidth * Meta.picHeight)/2;

	auto* Pixels8 = reinterpret_cast<uint8_t*>(Picture.pOutputPicture);
	SoyPixelsRemote Pixels( Pixels8, DataSize, PixelMeta );
	if ( OnFrameDecoded )
		OnFrameDecoded( Pixels, DecodeDuration );
}


