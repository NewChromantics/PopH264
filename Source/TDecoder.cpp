#include "TDecoder.h"
#include "SoyH264.h"
#include <span>
#include "json11.hpp"
#include "FileReader.hpp"

namespace Jpeg
{
	static bool	IsJpegHeader(std::span<uint8_t> FileData);
}




PopH264::TDecoder::TDecoder(const TDecoderParams& Params,PopH264::OnDecodedFrame_t OnDecodedFrame,PopH264::OnFrameError_t OnFrameError) :
	mOnDecodedFrame	( OnDecodedFrame ),
	mOnFrameError	( OnFrameError ),
	mParams			( Params )
{
}

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

void PopH264::TDecoder::OnDecoderError(const std::string& Error)
{
	mOnFrameError( Error, nullptr );
}


void PopH264::TDecoder::OnDecodedEndOfStream()
{
	SoyPixels Null;
	json11::Json::object Meta;
	mOnDecodedFrame(Null,0,Meta);
}

void PopH264::TDecoder::PushEndOfStream()
{
	Decode( {}, FrameNumberInvalid, ContentType::EndOfFile );
}

void PopH264::TDecoder::CheckDecoderUpdates()
{
	CheckUpdates();
}



void PopH264::TDecoder::Decode(std::span<uint8_t> PacketData,FrameNumber_t FrameNumber,ContentType::Type ContentType)
{
	//	todo? if this is the first data, detect non-poph264/nalu input formats (eg. jpeg), that we dont want to split

	if ( ContentType == ContentType::EndOfFile )
	{
		//	mark that we expect no more data after this
		mPendingDataFinished = true;
	}
	
	//	now split when popped, in case this data isn't actually H264 data
	{
		std::scoped_lock Lock(mPendingDataLock);
		std::shared_ptr<TInputNaluPacket> pPacket( new TInputNaluPacket() );
		std::copy( PacketData.begin(), PacketData.end(), std::back_inserter(pPacket->mData) );
		pPacket->mFrameNumber = FrameNumber;
		pPacket->mContentType = ContentType;
		mPendingDatas.PushBack(pPacket);
	}
	
	while ( true )
	{
		//	keep decoding until no more data to process
		if ( !DecodeNextPacket() )
			break;
	}
}


void PopH264::TDecoder::UnpopPacket(std::shared_ptr<TInputNaluPacket> Packet)
{
	std::lock_guard<std::mutex> Lock(mPendingDataLock);
	mPendingDatas.PushBack(Packet);
}


bool Jpeg::IsJpegHeader(std::span<uint8_t> FileData)
{
	//	extra fast, extra loose jpeg header detector
	if ( FileData.size() < 10 )
		return false;
	
	using namespace PopH264;
	FileReader_t Reader( FileData );
	
	//	start of information
	auto Soi = Reader.Read16Reverse();
	if ( Soi != 0xffd8 )
		return false;
	
	//	sections follow this format
	auto MarkerStart = Reader.Read8();
	auto MarkerNumber = Reader.Read8();
	auto MarkerSize = Reader.Read16();
	
	if ( MarkerStart != 0xff )
		return false;
	//	marker number expected to be 0xe0+ for exif header
	
	//	too small to be exif
	if ( MarkerSize < 4 )
		return false;
	
	//	gr: this COULD be a different marker
	//auto ExifHeader = Reader.ReadFourcc('EXIF');
	auto ExifHeader = Reader.Read32Reverse();

	switch ( ExifHeader )
	{
		case 'JFIF':
		case 'Exif':
			return true;
			
		default:
		{
			//auto Fourcc = GetFourccString(ExifHeader,true);
			//std::cerr << "Failed to detect as jpeg, as first marker fourcc is " << Fourcc << std::endl;
			return false;
		}
	}
}

std::shared_ptr<PopH264::TInputNaluPacket> PopH264::TDecoder::PopNextPacket()
{
	//	gr:could returnthis now and avoid the copy & alloc at caller
	std::shared_ptr<TInputNaluPacket> NextPacket;
	{
		std::scoped_lock Lock( mPendingDataLock );
		if ( mPendingDatas.IsEmpty() )
		{
			//	expecting more data to come
			if ( !mPendingDataFinished )
				return nullptr;
		
			//	no more data ever
			return nullptr;
		}
		
		NextPacket = mPendingDatas.PopAt(0);
		std::span<uint8_t> NextPacketData( NextPacket->mData );
		
		//	todo: detect non h264 packets here (if first)
		if ( Jpeg::IsJpegHeader(NextPacketData) )
		{
			//	detected jpeg, dont attempt split
			std::Debug << "Detected Jpeg packet, skipping nalu split" << std::endl;
			NextPacket->mContentType = ContentType::Jpeg;
		}
		else if ( NextPacket->mData.empty() )
		{
			//	do nothing with empty packets (just retaining content type)
		}
		else
		{
			//	if this packet contains multiple nalu packets, split it here
			std::vector<std::shared_ptr<TInputNaluPacket>> SplitPackets;
			
			auto OnSplitNalu = [&](std::span<uint8_t> Nalu)
			{
				std::shared_ptr<TInputNaluPacket> pPacket( new TInputNaluPacket() );
				std::copy( Nalu.begin(), Nalu.end(), std::back_inserter(pPacket->mData) );
				pPacket->mFrameNumber = NextPacket->mFrameNumber;
				SplitPackets.push_back( pPacket );
			};
			H264::SplitNalu( NextPacket->GetData(), OnSplitNalu );
			
			//	if we split multiple re-insert back into the list
			NextPacket = SplitPackets[0];
			for ( auto i=1;	i<SplitPackets.size();	i++ )
				mPendingDatas.PushBack(SplitPackets[i]);
		}
	}
	
	return NextPacket;
}

bool PopH264::TDecoder::PopNalu(ArrayBridge<uint8_t>&& Buffer,FrameNumber_t& FrameNumber)
{
	auto NextPacket = PopNextPacket();
	if ( !NextPacket )
		return false;

	auto NextPacketData = NextPacket->GetData();
	FixedRemoteArray<uint8_t> NextPacketArray( NextPacketData.data(), NextPacketData.size() );
	Buffer.Copy( NextPacketArray );
	FrameNumber = NextPacket->mFrameNumber;
	
	//	anything calling this funciton is legacy that's popping h264 packets
	//	so if this is an EOF packet, re-insert h264 eos marker
	if ( NextPacket->mContentType == ContentType::EndOfFile )
	{
		auto Eos = H264::EncodeNaluByte(H264NaluContent::EndOfStream,H264NaluPriority::Important);
		uint8_t EndOfStreamNalu[]{ 0,0,0,1,Eos };
		Buffer.Copy( GetRemoteArray( EndOfStreamNalu ) );
	}
	
	return true;
}

void PopH264::TDecoder::PeekHeaderNalus(ArrayBridge<uint8_t>&& SpsBuffer,ArrayBridge<uint8_t>&& PpsBuffer)
{
	std::lock_guard<std::mutex> Lock( mPendingDataLock );
	
	for ( auto pd=0;	pd<mPendingDatas.GetSize();	pd++ )
	{
		auto PendingNaluData = mPendingDatas[pd]->GetData();
		auto NaluType = H264::GetPacketType(PendingNaluData);
		if (NaluType == H264NaluContent::SequenceParameterSet)
		{
			FixedRemoteArray Data( PendingNaluData.data(), PendingNaluData.size() );
			SpsBuffer.Copy(Data);
		}
		if (NaluType == H264NaluContent::PictureParameterSet)
		{
			FixedRemoteArray Data( PendingNaluData.data(), PendingNaluData.size() );
			PpsBuffer.Copy(Data);
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



