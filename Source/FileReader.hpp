#pragma once

#include "std_span.hpp"
#include <string>



//	to avoid symbol clash when static lib is linked, (as symbol visibility doesn't work very well cross platform)
//	hide the clashin symbols in a namespace
namespace PopH264
{
	class FileReader_t;
}

//	gr: I have a more mature class in SoyLib somewhere for this that is async-compatible
class PopH264::FileReader_t
{
public:
	FileReader_t(std::span<uint8_t> Data) :
		mData	( Data )
	{
	}
	
	size_t				size()		{	return mData.size();	}
	
	int					RemainingBytes()	{	return mData.size() - mPosition;	}
	std::span<uint8_t>	RemainingData()		{	return ReadBytes( RemainingBytes() );	}
	uint8_t				Read8();
	uint16_t			Read16();
	uint16_t			Read16Reverse();
	uint32_t			Read32();
	uint32_t			Read32Reverse();
	float				Read8AsFloat();
	float				Read16AsFloat();
	float				Read32AsFloat();
	void				ReadFourcc(uint32_t ExpectedFourcc);
	void				ReadFourccReverse(uint32_t ExpectedFourcc);
	std::span<uint8_t>	ReadBytes(size_t Size);
	template<typename TYPE>
	TYPE				ReadObject()
	{
		auto Bytes = ReadBytes( sizeof(TYPE) );
		auto* pObject = reinterpret_cast<TYPE*>( Bytes.data() );
		return *pObject;
	}
	std::string_view	ReadNullTerminatedString();	//	expecting chars until null terminator (todo: ascii checking?)
	
	size_t				GetReadPosition()	{	return mPosition;	}
	void				CheckRemaning(size_t Bytes);	//	throw if there aren't this many bytes remaining to be read
	
private:
	size_t				mPosition = 0;
	std::span<uint8_t>	mData;
};
