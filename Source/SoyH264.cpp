#include "SoyH264.h"


size_t H264::GetNaluLength(const ArrayBridge<uint8_t>& Data)
{
	auto Data0 = Data[0];
	auto Data1 = Data[1];
	auto Data2 = Data[2];
	auto Data3 = Data[3];
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


