#pragma once

#include "TCameraDevice.hpp"
#include "MfDecoder.h"
#include <SoyAutoReleasePtr.h>


namespace MediaFoundation
{
	void		EnumCaptureDevices(std::function<void(const std::string&)> AppendName);

	class TCaptureExtractor;
	class TCamera;
}


class MediaFoundation::TCamera : public TCameraDevice
{
public:
	TCamera(const std::string& DeviceName);

	void		PushLatestFrame(size_t StreamIndex);

	std::shared_ptr<TMediaExtractor>	mExtractor;
};


class MediaFoundation::TCaptureExtractor : public MfExtractor
{
public:
	TCaptureExtractor(const TMediaExtractorParams& Params);
	~TCaptureExtractor();

protected:
	virtual void		AllocSourceReader(const std::string& Filename) override;
	virtual bool		CanSeek() override				{	return false;	}
	virtual void		FilterStreams(ArrayBridge<TStreamMeta>& Streams) override;
	virtual void		CorrectIncomingTimecode(TMediaPacket& Timecode) override;

public:
	Soy::AutoReleasePtr<IMFMediaSource>		mMediaSource;
};


