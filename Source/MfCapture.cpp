#include "MfCapture.h"
#include <SortArray.h>


namespace MediaFoundation
{
	Soy::AutoReleasePtr<IMFMediaSource>	FindCaptureDevice(const std::string& Match);
}


//	gr: remvoe this and use Soy::AutoReleasePtr
template<typename T>
void SafeRelease(T*& Ptr)
{
	if ( !Ptr )
		return;
	Ptr->Release();
	Ptr = nullptr;
}

std::string GetDeviceName(IMFActivate& Device)
{
	WCHAR Buffer[1024];
	uint32 BufferSize = 0;
	auto Result = Device.GetString( MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, Buffer, sizeofarray(Buffer), &BufferSize );
	auto Name = Soy::WStringToString( Buffer );

	return Name;
}


void EnumCaptureDevices(std::function<void(IMFActivate&)> OnFoundDevice,GUID DeviceType)
{
	IMFAttributes* pAttributes = nullptr;
	auto Result = MFCreateAttributes( &pAttributes, 1 );
	MediaFoundation::IsOkay( Result, "MFCreateAttributes" );

	// Source type: video capture devices
	Result = pAttributes->SetGUID( MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, DeviceType );
	MediaFoundation::IsOkay( Result, "Set attribs for vidcap" );

    UINT32 DeviceCount = 0;
	IMFActivate **ppDevices = nullptr;

	auto Cleanup = [&]
	{
		for ( DWORD i=0;	ppDevices&&i<DeviceCount;	i++)
		{
			SafeRelease( ppDevices[i] );
		}

		if ( ppDevices )
		{
			CoTaskMemFree(ppDevices);
			ppDevices = nullptr;
		}

		SafeRelease(pAttributes);
	};

	try
	{
	    Result = MFEnumDeviceSources(pAttributes, &ppDevices, &DeviceCount );
	 	MediaFoundation::IsOkay( Result, "Set attribs for vidcap" );

		if ( DeviceCount == 0 )
		{
			Cleanup();
			return;
		}
		Soy::Assert( ppDevices != nullptr, "MFEnumDeviceSources returned null" );

		for ( int i=0;	i<DeviceCount;	i++ )
		{
			auto* pDevice = ppDevices[i];
			//	not expecting null devices?
			if ( !pDevice )
			{
				std::Debug << "Null device found whilst enumerating capture devices (" << DeviceType << ")" << std::endl;
				continue;
			}
			OnFoundDevice( *pDevice );
		}

		Cleanup();
	}
	catch(...)
	{
		Cleanup();
		throw;
	}

}



void MediaFoundation::EnumCaptureDevices(std::function<void(const std::string&)> AppendName)
{
	auto OnFoundDevice = [&](IMFActivate& Device)
	{
		auto Name = GetDeviceName( Device );
		AppendName( Name );
	};

	try
	{
		EnumCaptureDevices( OnFoundDevice, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID );
		EnumCaptureDevices( OnFoundDevice, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_GUID );
	}
	catch(std::exception& e)
	{
		std::Debug << "Failed to enum capture devices " << e.what() << std::endl;
	}
}


int GetDeviceNameMatchScore(const std::string& Name,const std::string& Match)
{
	if ( Match == "*" )
		return 1;

	int ContainsScore = 100;
	int StartsWithScore = 10;

	int Score = 0;
	bool StartsWith = Soy::StringBeginsWith( Name, Match, false );
	bool Contains = StartsWith ? true : Soy::StringContains( Name, Match, false );

	Score += StartsWith * StartsWithScore;
	Score += Contains * ContainsScore;
	return Score;
}

Soy::AutoReleasePtr<IMFMediaSource> MediaFoundation::FindCaptureDevice(const std::string& Match)
{
	//	matching devices & score
	typedef std::pair<Soy::AutoReleasePtr<IMFActivate>,int> TDeviceAndScore;
	Array<TDeviceAndScore> DevicesAndScores;

	auto OnFoundDevice = [&](IMFActivate& Device)
	{
		auto Name = GetDeviceName( Device );
		int Score = GetDeviceNameMatchScore( Name, Match );
		if ( Score == 0 )
			return;

		//	increase refcount on this device as Enum will release them all
		Soy::AutoReleasePtr<IMFActivate> pDevice( &Device, true );
		DevicesAndScores.PushBack( std::make_pair( pDevice, Score ) );
	};

	try
	{
		EnumCaptureDevices( OnFoundDevice, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID );
		EnumCaptureDevices( OnFoundDevice, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_GUID );
	}
	catch(std::exception& e)
	{
		std::Debug << "Failed to enum capture devices " << e.what() << std::endl;
	}

	//	find best
	auto Compare = [](const TDeviceAndScore& a,const TDeviceAndScore& b)
	{
		auto Scorea = a.second;
		auto Scoreb = b.second;
		if ( Scorea > Scoreb )	return -1;
		if ( Scorea < Scoreb )	return 1;
		return 0;
	};
	SortArrayLambda<TDeviceAndScore> SortedDevices( GetArrayBridge(DevicesAndScores), Compare );
	SortedDevices.Sort();

	if ( SortedDevices.IsEmpty() )
		return Soy::AutoReleasePtr<IMFMediaSource>();

	//	explcit acquire
	auto pDevice = SortedDevices[0].first;
	IMFMediaSource* pSource = nullptr;
	IMFMediaSource** ppSource = &pSource;
	auto Result = pDevice->ActivateObject( IID_PPV_ARGS(ppSource) );
	::MediaFoundation::IsOkay( Result, "Device::ActivateObject");

	if ( !pSource )
	{
		std::stringstream Error;
		Error << "Error getting media source for device " << GetDeviceName( *pDevice ) << " but no error reported";
		throw Soy::AssertException( Error.str() );
	}

	Soy::AutoReleasePtr<IMFMediaSource> Source( pSource, true );
	return Source;
}



SoyPixelsFormat::Type GetMeta(IMFStreamDescriptor* Stream,bool VerboseDebug)
{
	Soy::Assert( Stream !=nullptr, "Stream descriptor expected");

	Soy::AutoReleasePtr<IMFMediaTypeHandler> MediaHandler;
	auto Result = Stream->GetMediaTypeHandler( &MediaHandler.mObject );
	MediaFoundation::IsOkay( Result, "MediaSource.stream.GetMediaTypeHandler");

	Soy::AutoReleasePtr<IMFMediaType> MediaType;
	Result = MediaHandler->GetCurrentMediaType( &MediaType.mObject );
	MediaFoundation::IsOkay( Result, "MediaSource.stream.GetMediaTypeHandler.GetCurrentMediaType");
	
	auto Meta = MediaFoundation::GetStreamMeta( *MediaType, VerboseDebug );
	return Meta.mPixelMeta.GetFormat();
}

Soy::AutoReleasePtr<IMFPresentationDescriptor> GetPresentationDescriptor(std::map<size_t,SoyPixelsFormat::Type>& StreamFormats,IMFMediaSource& MediaSource,bool VerboseDebug)
{
	//	https://msdn.microsoft.com/en-us/library/windows/desktop/ms702261(v=vs.85).aspx
	//	gr: this gets the DEFAULT format... or the current one?
	//		but we can change it, and gets affected when we call Start()

	IMFPresentationDescriptor* pDescriptor = nullptr;
	auto Result = MediaSource.CreatePresentationDescriptor( &pDescriptor );

	//	result already has a ref added
	Soy::AutoReleasePtr<IMFPresentationDescriptor> Descriptor( pDescriptor, false );

	//	failed to get a format
	MediaFoundation::IsOkay( Result, "MediaSource.CreatePresentationDescriptor");
	if ( !Descriptor )
		return Soy::AutoReleasePtr<IMFPresentationDescriptor>();

	//	extract each stream descriptor to get the format
	DWORD StreamCount = 0;
	Result = Descriptor->GetStreamDescriptorCount( &StreamCount );
	MediaFoundation::IsOkay( Result, "MediaSource.GetStreamDescriptorCount");

	for ( int i=0;	i<StreamCount;	i++ )
	{
		//	get stream descriptor
		try
		{
			BOOL Enabled = false;
			IMFStreamDescriptor* pStream = nullptr;
			Result = Descriptor->GetStreamDescriptorByIndex( i, &Enabled, &pStream );
			Soy::AutoReleasePtr<IMFStreamDescriptor> Stream( pStream, false );

			std::stringstream Error;
			Error << "GetStreamDescriptorByIndex(" << i << ")";
			MediaFoundation::IsOkay( Result, Error.str() );
			auto Meta = GetMeta(Stream, VerboseDebug);
			StreamFormats[i] = Meta;
		}
		catch(std::exception& e)
		{
			if ( VerboseDebug )
				std::Debug << "Failed to get descriptor stream " << i << ": " << e.what() << std::endl;
			continue;
		}    
	}


	/*
	//	enable/disable streams
	Descriptor->SelectStream();
	Descriptor->DeselectStream();

	*/

	return Descriptor;
}


MediaFoundation::TCaptureExtractor::TCaptureExtractor(const TMediaExtractorParams& Params) :
	MfExtractor	( Params )
{
	Init();
}

void MediaFoundation::TCaptureExtractor::AllocSourceReader(const std::string& Filename)
{
	//	find device
	mMediaSource = FindCaptureDevice( Filename );
	if ( !mMediaSource )
	{
		std::stringstream Error;
		Error << "Failed to find capture device matching " << Filename;
		throw Soy::AssertException( Error.str() );
	}

	//	setup format
	std::map<size_t,SoyPixelsFormat::Type> Formats;
	auto PresentationDescriptor = GetPresentationDescriptor( Formats, *mMediaSource, mParams.mVerboseDebug );
	
	//	pick best descriptor
	if ( Formats.empty() )
	{
		std::stringstream Error;
		Error << "Could not resolve any compatible formats for capture device";
		throw Soy::AssertException( Error.str() );
	}

	//	do we need to explicitly retain the descriptor? for the life time of the video?
	{
		const GUID* TimeFormat = nullptr;

		PROPVARIANT StartTime;
		PropVariantInit(&StartTime);

		//	dont set a seek, start from current pos
		StartTime.vt = VT_EMPTY;
		//StartTime.hVal.QuadPart = 10000000; // 10^7 = 1 second.

		auto Result = mMediaSource->Start( PresentationDescriptor, TimeFormat, &StartTime );
		::MediaFoundation::IsOkay( Result, "MediaSource.start" );
	}
	

	BufferArray<GUID,10> Attribs;
	if ( mParams.mForceNonPlanarOutput )
	{
		Attribs.PushBack( MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING );
	}

	Soy::AutoReleasePtr<IMFAttributes> Attributes;

	//	auto retained
	auto Result = MFCreateAttributes( &Attributes.mObject, 1 );
	if ( Result == S_OK )
	{
		//	setup
		for ( int i=0;	i<Attribs.GetSize();	i++ )
		{
			Result = Attributes->SetUINT32( Attribs[i], 1 );	//	set nonzero
			std::stringstream Error;
			Error << "Setting attribute " << Attribs[i];

			//	gr: abort if any failed?
			MediaFoundation::IsOkay( Result, Error.str() );
		}

		//	set async callback
		if ( mSourceReaderCallback )
		{
			Result = Attributes->SetUnknown( MF_SOURCE_READER_ASYNC_CALLBACK, mSourceReaderCallback.mObject );
			std::stringstream Error;
			Error << "Setting attribute MF_SOURCE_READER_ASYNC_CALLBACK";

			//	gr: abort if any failed?
			MediaFoundation::IsOkay( Result, Error.str() );
		}
	}
	else
	{
		Attributes.Release();
		MediaFoundation::IsOkay( Result, "MFCreateAttributes" );
	}

	//	create the source reader
	{
		auto Result = MFCreateSourceReaderFromMediaSource( mMediaSource, Attributes, &mSourceReader.mObject );
		MediaFoundation::IsOkay(Result, "MFCreateSourceReaderFromMediaSource");
	}
}


MediaFoundation::TCaptureExtractor::~TCaptureExtractor()
{
	WaitToFinish();

	if ( mMediaSource )
	{
		mMediaSource->Stop();
		mMediaSource.Release();
	}
}


void MediaFoundation::TCaptureExtractor::FilterStreams(ArrayBridge<TStreamMeta>& Streams)
{
	//	camera capture sometimes reports multiple video stream indexes (eg. apple isight), as well as the 400 streams of different formats
	//	if I enable 0 and 1, it gives me a few frames and stops... so work around that
	int FirstVideoStreamIndex = -1;

	for ( int i=0;	i<Streams.GetSize();	i++ )
	{
		auto& StreamMeta = Streams[i];
		if ( !SoyMediaFormat::IsVideo( StreamMeta.mCodec ) )
			continue;

		FirstVideoStreamIndex = StreamMeta.mStreamIndex;
		break;
	}

	//	no ivode stream indexes?
	if ( FirstVideoStreamIndex != -1 )
	{
		for ( int i=Streams.GetSize()-1;	i>=0;	i-- )
		{
			auto& StreamMeta = Streams[i];
			if ( !SoyMediaFormat::IsVideo( StreamMeta.mCodec ) )
				continue;

			//	index is okay
			if ( StreamMeta.mStreamIndex == FirstVideoStreamIndex )
				continue;
	
			//	ditch stream
			std::Debug << "Ditching video stream as stream index is different to first-found. Enabling multiple streams hangs MediaFoundation decoder. Report if you have a camera with multiple streams! " << StreamMeta << std::endl;
			Streams.RemoveBlock(i,1);
		}
	}
}


void MediaFoundation::TCaptureExtractor::CorrectIncomingTimecode(TMediaPacket& Packet)
{
	//	gr; maybe an option to ignore the hardwares clock is needed for a 3rd option here, rather than overriding it
	if ( mParams.mLiveUseClockTime )
	{
		static bool OverridePacketTimecode = false;
		if ( OverridePacketTimecode )
		{
			Packet.mTimecode = SoyTime(true);
		}
	}
	else
	{
		Packet.mTimecode = GetSeekTime();
	}
}



MediaFoundation::TCamera::TCamera(const std::string& DeviceName)
{
	TMediaExtractorParams Params(DeviceName);

	Params.mOnFrameExtracted = [this](const SoyTime,size_t StreamIndex)
	{
		PushLatestFrame(StreamIndex);
	};

	//	gr: pitch padding is crashing, my padding code might be wrong... but none of my cameras are giving out unaligned images...
	Params.mApplyHeightPadding = false;
	Params.mApplyWidthPadding = false;

	mExtractor.reset(new TCaptureExtractor(Params));
}

void MediaFoundation::TCamera::PushLatestFrame(size_t StreamIndex)
{
	//	get latest packet
	auto StreamBuffer = mExtractor->GetStreamBuffer(StreamIndex);
	if ( !StreamBuffer )
	{
		auto MaxBufferSize = 10;
		StreamBuffer = mExtractor->AllocStreamBuffer(StreamIndex, MaxBufferSize);
	}
	if ( !StreamBuffer )
	{
		std::Debug << "No stream buffer for stream " << StreamIndex << std::endl;
		return;
	}

	std::shared_ptr<TMediaPacket> LatestPacket;
	while ( true )
	{
		auto NextPacket = StreamBuffer->PopPacket();
		if ( !NextPacket )
			break;
		LatestPacket = NextPacket;
	}

	if ( !LatestPacket )
		return;

	this->PushFrame(LatestPacket->mPixelBuffer, LatestPacket->mMeta.mPixelMeta);
}
