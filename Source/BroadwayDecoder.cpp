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
/*
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
*/

Broadway::TDecoder::TDecoder(PopH264::TDecoderParams Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnError_t OnError) :
	PopH264::TDecoder	( OnDecodedFrame, OnError ),
	mParams				( Params )
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

//	returns true if more data to proccess
bool Broadway::TDecoder::DecodeNextPacket()
{
	Array<uint8_t> Nalu;
	PopH264::FrameNumber_t FrameNumber = 0;
	if ( !PopNalu( GetArrayBridge(Nalu), FrameNumber ) )
		return false;
	
	const unsigned IntraGrayConcealment = 0;
	const unsigned IntraReferenceConcealment = 1;
	
	H264SwDecInput Input;
	Input.pStream = Nalu.GetArray();
	Input.dataLen = Nalu.GetDataSize();
	
	if ( Input.dataLen == 0 )
		return false;

	Input.picId = FrameNumber;
	Input.intraConcealmentMethod = IntraGrayConcealment;
	
	H264SwDecOutput Output;
	Output.pStrmCurrPos = nullptr;
	
	//	this may throw, probably shouldn't let it, but not sure what to do yet, so let it throw
	auto H264PacketType = H264::GetPacketType(GetArrayBridge(Nalu));

	//	if we havent had headers yet, broadway will fail (and not recover)
	//	if we try and process frames, so drop them
	bool DecodePacket = true;
	
	switch (H264PacketType)
	{
		//	can always process sps
		case H264NaluContent::SequenceParameterSet:
			break;
			
		case H264NaluContent::PictureParameterSet:
			//	broadway needs SPS before PPS
			if ( !mProcessedSps )
				DecodePacket = false;
			break;
			
		case H264NaluContent::Slice_CodedIDRPicture:
			//	don't decode until sps && pps done
			if ( !mProcessedSps || !mProcessedPps )
				DecodePacket = false;
			break;
				
		case H264NaluContent::Slice_NonIDRPicture:
		case H264NaluContent::Slice_CodedPartitionA:
		case H264NaluContent::Slice_CodedPartitionB:
		case H264NaluContent::Slice_CodedPartitionC:
		case H264NaluContent::Slice_AuxCodedUnpartitioned:
		default:
			//	need to process keyframe before intra frames or decoder gets stuck
			if ( !mProcessedKeyframe )
				DecodePacket = false;
			break;
	}

	if ( mParams.mVerboseDebug )
	{
		//std::Debug << "Packet H264SwDecDecode(" << magic_enum::enum_name(H264PacketType) << ") x" << Nalu.GetDataSize() << " DecodePacket=" << DecodePacket << std::endl;
	}
	
	H264SwDecRet Result = H264SWDEC_STRM_PROCESSED;
	SoyTime DecodeDuration;
	ssize_t BytesProcessed = 0; 
	
	if ( DecodePacket )
	{
		Soy::TScopeTimerPrint Timer("H264 Decode",15);
		Result = H264SwDecDecode( mDecoderInstance, &Input, &Output );
	
		//	the first time we decode the first keyframe, it recognises new headers, but doesn't
		//	actually decode the frame, so it never comes out (we havent got it flushing correctly yet)
		//	processing this keyframe again has the active SPS setup, so we get a frame decoded immediately!
		if ( Result == H264SWDEC_HDRS_RDY_BUFF_NOT_EMPTY && H264PacketType == H264NaluContent::Slice_CodedIDRPicture )
		{
			Result = H264SwDecDecode( mDecoderInstance, &Input, &Output );
			//	gr; OnMeta() may not be called now
		}
		DecodeDuration = Timer.Stop();
		IsOkay( Result, "H264SwDecDecode" );
	
		//	calc what data wasn't used
		BytesProcessed = static_cast<ssize_t>(Output.pStrmCurrPos - Input.pStream);
		
		if ( BytesProcessed != Input.dataLen )
		//if ( mParams.mVerboseDebug )
			std::Debug << "H264SwDecDecode result: " << GetDecodeResultString(Result) << ". Bytes processed: "  << BytesProcessed << "/" << Input.dataLen << std::endl;
	}
	
	//	using AVF encoder, we weren't getting meta even after SPS/PPS (and SEI)
	//	instead, if the encoder ate a SPS, consider headers delivered (maybe after SPS AND PPS?)
	if ( DecodePacket )
	{
		if ( mParams.mVerboseDebug )
			std::Debug << "Decoded " << H264PacketType << " result=" << GetDecodeResultString(Result) << std::endl;
		
		switch (H264PacketType)
		{
			case H264NaluContent::SequenceParameterSet:
				mProcessedSps = true;
				break;
				
			case H264NaluContent::PictureParameterSet:
				mProcessedPps = true;
				break;
	
			case H264NaluContent::Slice_CodedIDRPicture:
				mProcessedKeyframe = true;
				break;
	
			default:
				break;
		}
	}
	else
	{
		if ( mParams.mVerboseDebug )
			std::Debug << "Dropped " << H264PacketType << " x" << Nalu.GetDataSize() << std::endl;
	}

	//	handle result
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
				OnPicture( Picture, Meta, DecodeDuration );
			}
			return true;
		}
			
			//	data eaten, no ouput
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
	if ( mParams.mVerboseDebug )
	{
		std::Debug << __PRETTY_FUNCTION__ << 
			" profile=" << Meta.profile << 
			" picWidth=" << Meta.picWidth <<
			" picHeight=" << Meta.picHeight <<
			" videoRange=" << Meta.videoRange <<
			" matrixCoefficients=" << Meta.matrixCoefficients <<
			" parWidth=" << Meta.parWidth <<
			" parHeight=" << Meta.parHeight <<
			" croppingFlag=" << Meta.croppingFlag <<
			" cropParams.cropLeftOffset=" << Meta.cropParams.cropLeftOffset <<
			" cropParams.cropOutWidth=" << Meta.cropParams.cropOutWidth <<
			" cropParams.cropTopOffset=" << Meta.cropParams.cropTopOffset <<
			" cropParams.cropOutHeight=" << Meta.cropParams.cropOutHeight <<
			std::endl;
	}
}

void Broadway::TDecoder::OnPicture(const H264SwDecPicture& Picture,const H264SwDecInfo& Meta,SoyTime DecodeDuration)
{
	auto FrameNumber = Picture.picId;
	
	//		headers just say
	//	u32 *pOutputPicture;    /* Pointer to the picture, YUV format       */
	auto Format = SoyPixelsFormat::Yuv_8_8_8;
	SoyPixelsMeta PixelMeta( Meta.picWidth, Meta.picHeight, Format );
	
	if ( mParams.mVerboseDebug )
		std::Debug << "Decoded picture " << PixelMeta << std::endl;
	
	//	gr: wish we knew exactly how many bytes Picture.pOutputPicture pointed at!
	//		but demos all use this measurement
	//auto DataSize = PixelMeta.GetDataSize();
	auto DataSize = (3 * Meta.picWidth * Meta.picHeight)/2;

	auto* Pixels8 = reinterpret_cast<uint8_t*>(Picture.pOutputPicture);
	SoyPixelsRemote Pixels( Pixels8, DataSize, PixelMeta );
	
	OnDecodedFrame( Pixels, FrameNumber );
}


