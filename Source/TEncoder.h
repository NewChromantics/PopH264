#pragma once

#include "Array.hpp"
#include "HeapArray.hpp"
#include "SoyTime.h"
#include <functional>
#include <span>

class SoyPixelsImpl;


namespace PopH264
{
	class TEncoder;
	class TPacket;
	class TEncoderFrameMeta;
}

class PopH264::TPacket
{
public:
	std::span<uint8_t>		GetData()	{	return mData ? std::span<uint8_t>( mData->data(), mData->size() ) : std::span<uint8_t>();	}
public:
	std::shared_ptr<std::vector<uint8_t>>	mData;
	std::string								mInputMeta;	//	original input meta json
	bool									mEndOfStream = false;
	std::string								mError;
};


//	as packets are popped asynchronously to input, we need to keep meta
//	associated with frames we use an arbritry number for frame (presentation
//	time)
class PopH264::TEncoderFrameMeta
{
public:
	size_t		mFrameNumber = 0;
	std::string	mMeta;
};


class PopH264::TEncoder
{
public:
	TEncoder(std::function<void(TPacket&)> OnOutputPacket);
	
	//	two overloads in case there's optimised versions for different encoders
	virtual void	Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe) = 0;
	virtual void	Encode(const SoyPixelsImpl& Pixels,const std::string& Meta,bool Keyframe)=0;
	virtual void	FinishEncoding()=0;
	
	virtual std::string	GetEncoderName()	{	return {};	}

protected:
	void			OnOutputPacket(TPacket& Packet);
	void			OnError(std::string_view Error);
	void			OnFinished();

	//	returns frame number used as PTS and stores meta
	size_t			PushFrameMeta(const std::string& Meta);
	//	gr: SOME frames will yield multiple packets (eg SPS & PPS) so some we need to keep around...
	//		gotta work out a way to figure out what we can discard
	std::string		GetFrameMeta(size_t FrameNumber);

	bool			HasEncodingFinished()	{	return mHasOutputEndOfStream || mHasOutputError;	}
	
private:
	std::function<void(TPacket&)>	mOnOutputPacket;
	bool						mHasOutputEndOfStream = false;	//	we've sent out an EOF frame
	bool						mHasOutputError = false;

	std::mutex					mFrameMetaLock;
	size_t						mFrameCount = 0;
	Array<TEncoderFrameMeta>	mFrameMetas;
};
