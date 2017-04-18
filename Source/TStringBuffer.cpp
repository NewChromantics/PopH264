#include "TStringBuffer.hpp"
#include <sstream>




const char* TString::Get()
{
	return mData;
}

void TString::Set(const std::string& String)
{
	
}

void TString::Clear()
{
	mData[0] = '\0';
}
	




void TStringBuffer::Push(const std::string& String)
{
	std::lock_guard<std::mutex> Lock( mBufferLock );
	
	//mBuffer.push_back( String );
}

void TStringBuffer::Push(std::stringstream& String)
{
	Push( String.str() );
}

const char* TStringBuffer::Pop()
{
	std::lock_guard<std::mutex> Lock( mBufferLock );

	if ( mBuffer.empty() )
		return nullptr;
	
	auto& First = mBuffer[0];
	return First.Get();
}

void TStringBuffer::Release(const char* String)
{
	std::lock_guard<std::mutex> Lock( mBufferLock );
	
	for ( size_t i=0;	i<mBuffer.size();	i++ )
	{
		
	}
}

