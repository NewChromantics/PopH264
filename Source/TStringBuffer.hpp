#pragma once
#include <vector>
#include <mutex>



class TString
{
public:
	TString()
	{
		Clear();
	}
	
	const char*	Get();
	void		Set(const std::string& String);
	void		Clear();
	
public:
	char		mData[200];
};




class TStringBuffer
{
public:
	void						Push(const std::string& String);
	void						Push(std::stringstream& String);
	const char*					Pop();							//	get pointer to the FIFO string. null if none
	void						Release(const char* String);
	
public:
	std::mutex					mBufferLock;
	std::vector<TString>		mBuffer;
};


