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

ContentType::Type DetectPacketContentType(std::span<uint8_t> PacketData)
{
	if ( Jpeg::IsJpegHeader(PacketData) )
		return ContentType::Jpeg;
	
	throw std::runtime_error("todo: detect nalu and h264 vs hevc here");
	
	return ContentType::Unknown;
}


void PopH264::TDecoder::Decode(std::span<uint8_t> PacketData,FrameNumber_t FrameNumber,ContentType::Type ContentType)
{
	//	if unknown content, try and detect it here
	//	gr: if we have previously detected it... keep detecting here?
	if ( ContentType == ContentType::Unknown )
	{
		ContentType = DetectPacketContentType( PacketData );
	}
	
	if ( ContentType == ContentType::EndOfFile )
	{
		//	mark that we expect no more data after this
		mPendingDataFinished = true;
	}
	
	
	std::vector<std::span<uint8_t>> NewPacketDatas;
	NewPacketDatas.push_back( PacketData );
	
	//	gr: we DO want to split SPS & PPS & SEI...
	if ( ContentType == ContentType::Unknown /*|| ContentType == ContentType::H264 */)
	{
		try
		{
			auto SubPackets = H264::SplitNalu(PacketData);
			NewPacketDatas = SubPackets;
		}
		catch (std::exception& e)
		{
			//	 probably not h264?
			std::Debug << "SplitNalu() error; " << e.what() << std::endl;
		}
	}
	
	//	now split when popped, in case this data isn't actually H264 data
	{
		std::vector<std::shared_ptr<TInputNaluPacket>> NewPackets;
		for ( auto& NewPacketData : NewPacketDatas )
		{
			std::shared_ptr<TInputNaluPacket> pPacket( new TInputNaluPacket() );
			std::copy( NewPacketData.begin(), NewPacketData.end(), std::back_inserter(pPacket->mData) );
			pPacket->mFrameNumber = FrameNumber;
			pPacket->mContentType = ContentType;
			NewPackets.push_back(pPacket);
		}
		
		std::scoped_lock Lock(mPendingDataLock);
		for ( auto& NewPacket : NewPackets )
			mPendingDatas.PushBack(NewPacket);
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
	/*auto MarkerNumber =*/ Reader.Read8();
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
		
		//	gr: don't split nalus. input data should be frame-seperated anyway
		//		we sometimes get IDR packets, which are 2 packets, for one frame (eg, top & bottom half)
		//		if we try and decode seperately, some decoders (apple) will fail with bad-data
		//	gr: https://developer.apple.com/forums/thread/14212
		//		we need to NOT split packets if theyre the same frame. IDRs can be multiple packets, but apple needs them together
		//	gr: but why did we split it in the first place... easier debugging?...
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

void PopH264::TDecoder::PeekHeaderNalus(std::vector<uint8_t>& SpsBuffer,std::vector<uint8_t>& PpsBuffer)
{
	std::lock_guard<std::mutex> Lock( mPendingDataLock );
	
	for ( auto pd=0;	pd<mPendingDatas.GetSize();	pd++ )
	{
		auto PendingNaluData = mPendingDatas[pd]->GetData();
		auto NaluType = H264::GetPacketType(PendingNaluData);
		if (NaluType == H264NaluContent::SequenceParameterSet)
		{
			std::copy( PendingNaluData.begin(), PendingNaluData.end(), std::back_inserter(SpsBuffer) );
		}
		if (NaluType == H264NaluContent::PictureParameterSet)
		{
			std::copy( PendingNaluData.begin(), PendingNaluData.end(), std::back_inserter(PpsBuffer) );
		}
		if ( !SpsBuffer.empty() && !PpsBuffer.empty() )
			return;
	}

	if ( !SpsBuffer.empty() && !PpsBuffer.empty() )
		return;

	std::stringstream Debug;
	Debug << __PRETTY_FUNCTION__ << " failed to find sps(x" << SpsBuffer.size() << ") or Pps(x" << PpsBuffer.size() << ")";
	throw Soy::AssertException(Debug);
}



