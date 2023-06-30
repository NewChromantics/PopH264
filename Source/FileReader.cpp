#include "FileReader.hpp"
#include <sstream>


void FileReader_t::CheckRemaning(size_t Bytes)
{
	auto Remaining = RemainingBytes();
	if ( Bytes <= Remaining )
		return;
	
	std::stringstream Error;
	Error << "Attempting to read x" << Bytes << "bytes but only " << Remaining << "/" << size() << "bytes remaning.";
	throw std::runtime_error(Error.str());
}

std::span<uint8_t> FileReader_t::ReadBytes(size_t Size)
{
	CheckRemaning(Size);
	auto* Start = &mData[mPosition];
	std::span<uint8_t> Bytes( Start, Size );
	mPosition += Size;
	return Bytes;
}

uint8_t FileReader_t::Read8()
{
	CheckRemaning(1);
	auto Int = mData[mPosition];
	mPosition += 1;
	return Int;
}

uint16_t FileReader_t::Read16()
{
	CheckRemaning(2);
	uint16_t Int = 0;
	Int |= mData[mPosition+0] << 0;
	Int |= mData[mPosition+1] << 8;
	mPosition += 2;
	return Int;
}

uint16_t FileReader_t::Read16Reverse()
{
	CheckRemaning(2);
	uint16_t Int = 0;
	Int |= mData[mPosition+1] << 0;
	Int |= mData[mPosition+0] << 8;
	mPosition += 2;
	return Int;
}

uint32_t FileReader_t::Read32()
{
	CheckRemaning(4);
	uint32_t Int = 0;
	Int |= mData[mPosition+0] << 0;
	Int |= mData[mPosition+1] << 8;
	Int |= mData[mPosition+2] << 16;
	Int |= mData[mPosition+3] << 24;
	mPosition += 4;
	return Int;
}

uint32_t FileReader_t::Read32Reverse()
{
	CheckRemaning(4);
	uint32_t Int = 0;
	Int |= mData[mPosition+3] << 0;
	Int |= mData[mPosition+2] << 8;
	Int |= mData[mPosition+1] << 16;
	Int |= mData[mPosition+0] << 24;
	mPosition += 4;
	return Int;
}

void FileReader_t::ReadFourcc(uint32_t ExpectedFourcc)
{
	auto Fourcc = Read32();
	if ( Fourcc != ExpectedFourcc )
	{
		std::stringstream Error;
		//Error << "Wav chunk fourcc expected [" << GetFourccString(ExpectedFourcc) << "] found [" << GetFourccString(Fourcc) << "]";
		Error << "read incorrect fourcc";
		throw std::runtime_error( Error.str() );
	}
};

float FileReader_t::Read16AsFloat()
{
	auto Unsigned = Read16();
	auto Signed = static_cast<int16_t>(Unsigned);
	float Max = std::numeric_limits<int16_t>::max();
	//	gr: numeric limits is 32767, previously we used 32768.0f
	float Float = static_cast<float>(Signed) / Max;
	return Float;
}

float FileReader_t::Read8AsFloat()
{
	auto Unsigned = Read8();
	auto Signed = static_cast<int8_t>(Unsigned);
	float Max = std::numeric_limits<int8_t>::max();
	float Float = static_cast<float>(Signed) / Max;
	return Float;
}

float FileReader_t::Read32AsFloat()
{
	auto Unsigned = Read32();
	auto Signed = static_cast<int32_t>(Unsigned);
	//	android has a strict conversion error here, so hardcode for now and fix it in a bit
	//float Max = std::numeric_limits<int32_t>::max();
	float Max = 2147483647.0f;
	float Float = static_cast<float>(Signed) / Max;
	return Float;
}

std::string_view FileReader_t::ReadNullTerminatedString()
{
	//	must have at least a terminator!
	CheckRemaning(1);
	auto Remaining = RemainingBytes();
	
	for ( int i=0;	i<Remaining;	i++ )
	{
		auto Char = mData[mPosition+i];
		
		//	todo: check ascii?
		if ( Char != 0 )
			continue;
		
		//	found terminator
		auto Length = i;
		auto StringBytes = ReadBytes( Length );
		auto Terminator = Read8();
		if ( Terminator != 0 )
			throw std::runtime_error("Read terminator wrong");
		
		auto* StringChars = reinterpret_cast<char*>( StringBytes.data() );
		return std::string_view( StringChars, Length );
	}
	throw std::runtime_error("Failed to find terminator for string");
}
