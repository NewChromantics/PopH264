#include "SoyH264.h"



size_t H264::GetNaluLength(const ArrayBridge<uint8_t>& Packet)
{
	//	todo: test for u8/u16/u32 size prefix
	if ( Packet.GetSize() < 4 )
		return 0;
	
	auto p0 = Packet[0];
	auto p1 = Packet[1];
	auto p2 = Packet[2];
	auto p3 = Packet[3];
	
	if ( p0 == 0 && p1 == 0 && p2 == 1 )
		return 3;
	
	if ( p0 == 0 && p1 == 0 && p2 == 0 && p3 == 1)
		return 4;
	
	//	couldn't detect, possibly no prefx and it's raw data
	//	could parse packet type to verify
	return 0;
}


size_t H264::GetNaluAnnexBLength(const ArrayBridge<uint8_t>& Packet)
{
	//	todo: test for u8/u16/u32 size prefix
	if ( Packet.GetSize() < 4 )
		throw Soy::AssertException("Packet not long enough for annexb");

	auto Data0 = Packet[0];
	auto Data1 = Packet[1];
	auto Data2 = Packet[2];
	auto Data3 = Packet[3];
	if (Data0 != 0 || Data1 != 0)
		throw Soy::AssertException("Data is not bytestream NALU header (leading zeroes)");
	if (Data2 == 1)
		return 3;
	if (Data2 == 0 && Data3 == 1)
		return 4;

	throw Soy::AssertException("Data is not bytestream NALU header (suffix)");
}

H264NaluContent::Type H264::GetPacketType(const ArrayBridge<uint8_t>&& Data)
{
	auto HeaderLength = GetNaluLength(Data);
	auto TypeAndPriority = Data[HeaderLength];
	auto Type = TypeAndPriority & 0x1f;
	auto Priority = TypeAndPriority >> 5;

	auto TypeEnum = static_cast<H264NaluContent::Type>(Type);
	return TypeEnum;
}


void ReformatDeliminator(ArrayBridge<uint8>& Data,
						 std::function<size_t(ArrayBridge<uint8>& Data,size_t Position)> ExtractChunk,
						 std::function<void(size_t ChunkLength,ArrayBridge<uint8>& Data,size_t& Position)> InsertChunk)
{
	size_t Position = 0;
	while ( true )
	{
		auto ChunkLength = ExtractChunk( Data, Position );
		if ( ChunkLength == 0 )
			break;
		{
			std::stringstream Error;
			Error << "Extracted NALU length of " << ChunkLength << "/" << Data.GetDataSize();
			Soy::Assert( ChunkLength <= Data.GetDataSize(), Error.str() );
		}
		
		InsertChunk( ChunkLength, Data, Position );
		Position += ChunkLength;
	}
}

void H264::DecodeNaluByte(uint8 Byte,H264NaluContent::Type& Content,H264NaluPriority::Type& Priority)
{
	uint8 Zero = (Byte >> 7) & 0x1;
	uint8 Idc = (Byte >> 5) & 0x3;
	uint8 Content8 = (Byte >> 0) & (0x1f);
	Soy::Assert( Zero==0, "Nalu zero bit non-zero");
	//	catch bad cases. look out for genuine cases, but if this is zero, NALU delin might have been read wrong
	Soy::Assert( Content8!=0, "Nalu content type is invalid (zero)");
	
	//	swich this for magic_enum
	//Priority = H264NaluPriority::Validate( Idc );
	//Content = H264NaluContent::Validate( Content8 );
	Priority = static_cast<H264NaluPriority::Type>( Idc );
	Content = static_cast<H264NaluContent::Type>( Content8 );
}

uint8 H264::EncodeNaluByte(H264NaluContent::Type Content,H264NaluPriority::Type Priority)
{
	//	uint8 Idc_Important = 0x3 << 5;	//	0x60
	//	uint8 Idc = Idc_Important;	//	011 XXXXX
	uint8 Idc = Priority;
	Idc <<= 5;
	uint8 Type = Content;
	
	uint8 Byte = Idc|Type;
	return Byte;
}


void H264::ConvertNaluPrefix(ArrayBridge<uint8_t>& Nalu,H264::NaluPrefix::Type NaluPrefixType)
{
	//	assuming annexb, this will throw if not
	auto PrefixLength = H264::GetNaluAnnexBLength(Nalu);
	
	//	quick implementation for now
	if ( NaluPrefixType != H264::NaluPrefix::ThirtyTwo )
		Soy_AssertTodo();
	
	auto NewPrefixSize = static_cast<int>(NaluPrefixType);
	
	//	pad if prefix was 3 bytes
	if ( PrefixLength == 3 )
		Nalu.InsertAt(0,0);
	else if ( PrefixLength != 4)
		throw Soy::AssertException("Expecting nalu size of 4");
	
	//	write over prefix
	uint32_t Size32 = Nalu.GetDataSize() - NewPrefixSize;
	uint8_t* Size8s = reinterpret_cast<uint8_t*>(&Size32);
	Nalu[0] = Size8s[3];
	Nalu[1] = Size8s[2];
	Nalu[2] = Size8s[1];
	Nalu[3] = Size8s[0];
}

size_t H264::GetNextNaluOffset(const ArrayBridge<uint8_t>&& Data, size_t StartFrom)
{
	//	detect 001
	auto* DataPtr = Data.GetArray();

	for (int i = StartFrom; i < Data.GetDataSize()-3; i++)
	{
		if (DataPtr[i + 0] != 0)	continue;
		if (DataPtr[i + 1] != 0)	continue;
		if (DataPtr[i + 2] != 1)	continue;

		//	check i-1 for 0 in case it's 0001 rather than 001
		if (DataPtr[i - 1] == 0)
			return i - 1;
		
		return i;
	}

	return 0;
}
