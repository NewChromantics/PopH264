#include "TDecoder.h"
#include "SoyH264.h"
#include "json11.hpp"


PopH264::TDecoder::TDecoder(PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	mOnDecodedFrame	( OnDecodedFrame ),
	mOnFrameError	( OnFrameError )
{
}


void PopH264::TDecoder::OnDecodedFrame(const SoyPixelsImpl& Pixels,FrameNumber_t FrameNumber,const json11::Json& Meta)
{
	mOnDecodedFrame( Pixels, FrameNumber, Meta );
}

void PopH264::TDecoder::OnDecodedFrame(const SoyPixelsImpl& Pixels,FrameNumber_t FrameNumber)
{
	json11::Json::object Meta;
	mOnDecodedFrame( Pixels, FrameNumber, Meta );
}


void PopH264::TDecoder::OnFrameError(const std::string& Error,FrameNumber_t FrameNumber)
{
	mOnFrameError( Error, &FrameNumber );
}


void PopH264::TDecoder::OnDecodedEndOfStream()
{
	SoyPixels Null;
	json11::Json::object Meta;
	mOnDecodedFrame(Null,0,Meta);
}

void PopH264::TDecoder::PushEndOfStream()
{
	//	send an explicit end of stream nalu
	//	todo: overload this for implementation specific flushes
	auto Eos = H264::EncodeNaluByte(H264NaluContent::EndOfStream,H264NaluPriority::Important);
	uint8_t EndOfStreamNalu[]{ 0,0,0,1,Eos };
	auto DataArray = GetRemoteArray( EndOfStreamNalu );
	
	//	mark pending data as finished
	mPendingDataFinished = true;
	Decode( GetArrayBridge(DataArray), 0 );
}

void PopH264::TDecoder::Decode(ArrayBridge<uint8_t>&& PacketData,FrameNumber_t FrameNumber)
{
	//	gr: maybe we should split when we PopNalu to move this work away from caller thread
	auto PushNalu = [&](const ArrayBridge<uint8_t>&& Nalu)
	{
		std::lock_guard<std::mutex> Lock(mPendingDataLock);
		std::shared_ptr<TInputNaluPacket> pPacket( new TInputNaluPacket() );
		pPacket->mData.Copy(Nalu);
		pPacket->mFrameNumber = FrameNumber;
		mPendingDatas.PushBack(pPacket);
	};
	H264::SplitNalu( PacketData, PushNalu );
	
	while ( true )
	{
		//	keep decoding until no more data to process
		if ( !DecodeNextPacket() )
			break;
	}
}


void PopH264::TDecoder::UnpopNalu(ArrayBridge<uint8_t>&& Nalu,FrameNumber_t FrameNumber)
{
	//	put-back data at the start of the queue
	//auto PushNalu = [&](const ArrayBridge<uint8_t>&& Nalu)
	{
		std::lock_guard<std::mutex> Lock(mPendingDataLock);
		std::shared_ptr<TInputNaluPacket> pPacket( new TInputNaluPacket() );
		pPacket->mData.Copy(Nalu);
		pPacket->mFrameNumber = FrameNumber;
		mPendingDatas.PushBack(pPacket);
	};
}


bool PopH264::TDecoder::PopNalu(ArrayBridge<uint8_t>&& Buffer,FrameNumber_t& FrameNumber)
{
	//	gr:could returnthis now and avoid the copy & alloc at caller
	std::shared_ptr<TInputNaluPacket> NextPacket;
	{
		std::lock_guard<std::mutex> Lock( mPendingDataLock );
		if ( mPendingDatas.IsEmpty() )
		{
			//	expecting more data to come
			if ( !mPendingDataFinished )
				return false;
		
			//	no more data ever
			return false;
		}
		NextPacket = mPendingDatas.PopAt(0);
	}
	Buffer.Copy( NextPacket->mData );
	FrameNumber = NextPacket->mFrameNumber;
	return true;
}


void PopH264::TDecoder::PeekHeaderNalus(ArrayBridge<uint8_t>&& SpsBuffer,ArrayBridge<uint8_t>&& PpsBuffer)
{
	std::lock_guard<std::mutex> Lock( mPendingDataLock );
	
	for ( auto pd=0;	pd<mPendingDatas.GetSize();	pd++ )
	{
		auto& PendingNaluData = mPendingDatas[pd]->mData;
		auto NaluType = H264::GetPacketType(GetArrayBridge(PendingNaluData));
		if (NaluType == H264NaluContent::SequenceParameterSet)
		{
			SpsBuffer.Copy(PendingNaluData);
		}
		if (NaluType == H264NaluContent::PictureParameterSet)
		{
			PpsBuffer.Copy(PendingNaluData);
		}
		if ( !SpsBuffer.IsEmpty() && !PpsBuffer.IsEmpty() )
			return;
	}

	if ( !SpsBuffer.IsEmpty() && !PpsBuffer.IsEmpty() )
		return;

	std::stringstream Debug;
	Debug << __PRETTY_FUNCTION__ << " failed to find sps(x" << SpsBuffer.GetDataSize() << ") or Pps(x" << PpsBuffer.GetDataSize() << ")";
	throw Soy::AssertException(Debug);
}



