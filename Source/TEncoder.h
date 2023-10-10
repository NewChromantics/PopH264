#pragma once

#include "Array.hpp"
#include "HeapArray.hpp"
#include "SoyTime.h"
#include <functional>
#include <span>
#include "TDecoderInstance.h"	//	EventTime_t
#include <unordered_map>

class SoyPixelsImpl;


namespace PopH264
{
	class TEncoder;
	class TPacket;
	class TEncoderFrameMeta;
}


//	as packets are popped asynchronously to input, we need to keep meta
//	associated with frames we use an arbritry number for frame (presentation
//	time), we can also store other encoder per-frame meta here (timing)
class PopH264::TEncoderFrameMeta
{
public:
	std::string	mInputMeta;	//	meta provided by user to keep with frame
	EventTime_t	mPushTime = EventTime_t::min();
	EventTime_t	mEncodedTime = EventTime_t::min();

	//	write encoded time if it hasn't been set
	void		OnEncoded()
	{
		//	already set
		if ( mEncodedTime != EventTime_t::min() )
			return;
		mEncodedTime = EventTime_t::clock::now();
	}
	
	std::chrono::milliseconds	GetEncodeDurationMs()
	{
		if ( mPushTime == EventTime_t::min() || mEncodedTime == EventTime_t::min() )
			return std::chrono::milliseconds(0);
		auto Delta = mEncodedTime - mPushTime;
		return std::chrono::duration_cast<std::chrono::milliseconds>( Delta );
	}
};


class PopH264::TPacket
{
public:
	std::span<uint8_t>		GetData()	{	return mData ? std::span<uint8_t>( mData->data(), mData->size() ) : std::span<uint8_t>();	}
	std::string_view		GetInputMeta()	{	return mEncodeMeta.mInputMeta;	}
public:
	std::shared_ptr<std::vector<uint8_t>>	mData;
	TEncoderFrameMeta						mEncodeMeta;	//	includes original input meta
	bool									mEndOfStream = false;
	std::string								mError;
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
	FrameNumber_t		PushFrameMeta(const std::string& Meta);
	//	gr: SOME frames will yield multiple packets (eg SPS & PPS) so some we need to keep around...
	//		gotta work out a way to figure out what we can discard
	TEncoderFrameMeta	GetFrameMeta(FrameNumber_t FrameNumber);

	bool				HasEncodingFinished()	{	return mHasOutputEndOfStream || mHasOutputError;	}
	
private:
	std::function<void(TPacket&)>	mOnOutputPacket;
	bool						mHasOutputEndOfStream = false;	//	we've sent out an EOF frame
	bool						mHasOutputError = false;

	std::mutex					mFrameMetaLock;
	FrameNumber_t				mFrameCount = 0;
	std::unordered_map<FrameNumber_t,TEncoderFrameMeta>	mFrameMetas;
};
