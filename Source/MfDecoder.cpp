#include "MfDecoder.h"
#include <Mferror.h>
#include <Codecapi.h>
#include <SoyH264.h>
#include <Propvarutil.h>

GUID GUID_Invalid = {0};

class PropVariantTime
{
public:
	PropVariantTime(SoyTime Time) :
		mInitialised	( false )
	{
		LONGLONG Nano100Secs = Time.GetNanoSeconds() / 100;
		auto Result = InitPropVariantFromInt64( Nano100Secs, &mProp );

		std::stringstream Error;
		Error << "Initialising propvariant with time " << Time;
		MediaFoundation::IsOkay( Result, Error.str() );
		mInitialised = true;
	}
	~PropVariantTime()
	{
		if ( mInitialised )
			PropVariantClear( &mProp );
	}

	PROPVARIANT	mProp;
	bool		mInitialised;
};


namespace MediaFoundation
{
	void							EnumStreams(ArrayBridge<TStreamMeta>& Streams,IMFSourceReader& Reader,bool VerboseDebug);
	void							GetSupportedFormats(ArrayBridge<SoyPixelsFormat::Type>&& Formats);

	TStreamMeta						GetStreamMeta(IMFSample& Sample,bool VerboseDebug);
	TStreamMeta						GetStreamMeta(IMFMediaType& MediaType,bool VerboseDebug);
	TStreamMeta						GetStreamMeta(IMFMediaType& MediaType,size_t StreamIndex,size_t MediaTypeIndex,IMFSourceReader& Reader,bool VerboseDebug);

	void							GetSampleData(ArrayBridge<uint8>&& Data,IMFSample& Sample);
}



namespace MediaFoundation
{
	SoyPixelsFormat::Type	GetPixelFormat(GUID Format);
}

MFExtractorCallback::MFExtractorCallback(MfExtractor& Parent) :
	mParent		( Parent ),
	mRefCount	( 0 )
{
}

STDMETHODIMP MFExtractorCallback::QueryInterface(REFIID iid, void** ppv)
{
	static const QITAB qit[] =
	{
		QITABENT(MFExtractorCallback, IMFSourceReaderCallback),
		{ 0 },
	};
	return QISearch(this, qit, iid, ppv);
}

STDMETHODIMP_(ULONG) MFExtractorCallback::AddRef()
{
	auto Count = ++mRefCount;
	return Count;
}

STDMETHODIMP_(ULONG) MFExtractorCallback::Release()
{
	auto Count = --mRefCount;
	if ( Count == 0 )
	{
		//	yuck	
		delete this;
	}
	return Count;
}

STDMETHODIMP MFExtractorCallback::OnEvent(DWORD, IMFMediaEvent *)
{
	return S_OK;
}

STDMETHODIMP MFExtractorCallback::OnFlush(DWORD)
{
	return S_OK;
}




void MediaFoundation::GetSampleData(ArrayBridge<uint8>&& Data,IMFSample& Sample)
{	
	//	copy to contigious media buffer
	IMFMediaBuffer* pBuffer = nullptr;
	auto Result = Sample.ConvertToContiguousBuffer( &pBuffer );
	MediaFoundation::IsOkay(Result, "ConvertToContiguousBuffer" );
	Soy::Assert( pBuffer!=nullptr, "Missing Media buffer object" );

	//	ConvertToContiguousBuffer automatically retains
	Soy::AutoReleasePtr<IMFMediaBuffer> BufferRet;
	BufferRet.Set( pBuffer, false );
	auto& Buffer = *BufferRet.mObject;

	//	lock
	BYTE* Bytes = nullptr;
	DWORD ByteSize = 0;
	
	//	note: lock is garunteed to be contiguous
	Result = Buffer.Lock( &Bytes, nullptr, &ByteSize );
	MediaFoundation::IsOkay( Result, "MediaBuffer::Lock" );

	Soy::Assert( Bytes!=nullptr, "MediaBuffer::Lock returned null" );
	Soy::Assert( ByteSize>0, "MediaBuffer::GetCurrentLength returned 0 bytes" );

	auto LockedArray = GetRemoteArray( Bytes, ByteSize );
	Data.Copy( LockedArray );
	
	Buffer.Unlock();
}


bool ReadAttrib32(IMFAttributes& Attribs,uint32& Value,GUID Guid,const std::string& ErrorPrefix,const char* GuidName)
{
	auto Result = Attribs.GetUINT32( Guid, &Value );

	std::stringstream Error;
	Error << ErrorPrefix << " ";
	if ( GuidName )
		Error << GuidName;
	else 
		Error << Guid;

	try
	{
		MediaFoundation::IsOkay(Result, Error.str());
		return true;
	}
	catch(std::exception& e)
	{
		std::Debug << e.what() << std::endl;
		return false;
	}
}

template<typename TYPE>
bool ReadAttrib32(IMFAttributes& Attribs,TYPE& Value,GUID Guid,const std::string& ErrorPrefix,const char* GuidName)
{
	uint32 Value32;
	if ( !ReadAttrib32( Attribs, Value32, Guid, ErrorPrefix, GuidName ) )
		return false;
	Value = static_cast<TYPE>(Value32);
	return true;
}


TStreamMeta MediaFoundation::GetStreamMeta(IMFSample& Sample,bool VerboseDebug)
{
	//	initialise with base meta
	TStreamMeta Meta;
	std::string ErrorPrefix = "Sample";

	//	key H264 frame info
	{
		eAVEncH264PictureType Value;
		if ( ReadAttrib32( Sample, Value, MFSampleExtension_VideoEncodePictureType, ErrorPrefix, "MFSampleExtension_VideoEncodePictureType" ) )
		{
		}
	}
	
	return Meta;
}

void MediaFoundation::GetMediaFileExtensions(ArrayBridge<std::string>&& Extensions)
{
	//	formats listed here, but pretty much an arbritry list
	//	https://msdn.microsoft.com/en-us/library/windows/desktop/dd757927%28v=vs.85%29.aspx?f=255&MSPPError=-2147217396
	const char* Ext[] = {
		".3gp",
		".mp4",
		".m4a",
		".aac",
		".ts",
		".ts2",
		".flac",
		".mp3",
		".mid",
		".xmf",
		".mxmf",
		".rtttl",
		".rtx",
		".ota",
		".imy",
		".ogg",
		".mkv",
		".wav",
		".webp",
		
		//	not listed
		".wave",
		".mov",
		".mp2",
		".m4v",
		".h264",
		".mpg",
	};
	Soy::PushStringArray( Extensions, Ext );
}


TStreamMeta MediaFoundation::GetStreamMeta(IMFMediaType& MediaType,bool VerboseDebug)
{
	//	initialise with base meta
	TStreamMeta Meta;

	//	gr: need to grab these beforehand for PCM
	{
		//	block alignment is BytesPerSample*ChannelCount so need to divide
		ReadAttrib32( MediaType, Meta.mChannelCount, MF_MT_AUDIO_NUM_CHANNELS, "MF_MT_AUDIO_NUM_CHANNELS", "MF_MT_AUDIO_NUM_CHANNELS" );
		uint32 BytesPerSample = 0;
		ReadAttrib32( MediaType, BytesPerSample, MF_MT_AUDIO_BLOCK_ALIGNMENT, "MF_MT_AUDIO_BLOCK_ALIGNMENT", "MF_MT_AUDIO_BLOCK_ALIGNMENT" );
		if ( Meta.mChannelCount > 0 )
		{
			Meta.mAudioBitsPerChannel = (BytesPerSample / Meta.mChannelCount) * 8;
		}
	}

	//	determine format from codec/main type
	{
		GUID MajorType = {0};
		MediaType.GetGUID( MF_MT_MAJOR_TYPE, &MajorType );
		GUID SubType = {0};
		MediaType.GetGUID( MF_MT_SUBTYPE, &SubType );


		size_t H264NaluLengthSize = 0;

		//	todo: probe for this from somewhere!
		if ( MajorType == MFMediaType_Video && SubType == MFVideoFormat_H264 )
		{
			H264NaluLengthSize = H264::GetNaluLengthSize( SoyMediaFormat::H264_32 );
			if ( VerboseDebug )
				std::Debug << __func__ << " Warning guessing at H264 NALU length size(" << H264NaluLengthSize << ")" << std::endl;
		}

		//	probe PCM size
		if ( MajorType == MFMediaType_Audio && SubType == MFAudioFormat_PCM )
		{
			H264NaluLengthSize = Meta.mAudioBitsPerChannel / 8;
		}

		Meta.mCodec = MediaFoundation::GetFormat( MajorType, SubType, H264NaluLengthSize );

		//	if we don't know pixel format here, try and set it from codec
		if ( Meta.mPixelMeta.GetFormat() == SoyPixelsFormat::Invalid )
			Meta.mPixelMeta.DumbSetFormat( SoyMediaFormat::GetPixelFormat( Meta.mCodec ) );
	}

	std::string ErrorPrefix = Meta.mDescription + " stream";



	//	extract other meta
	//	todo: find all unhandled mediatetype attributes and put into description
	{
		UINT32 Width = 0;
		UINT32 Height = 0;
		//	assume format is not known here
		auto Format = Meta.mPixelMeta.GetFormat();
		auto Result = MFGetAttributeSize( &MediaType, MF_MT_FRAME_SIZE, &Width, &Height );
		try
		{
			MediaFoundation::IsOkay(Result, ErrorPrefix + " MF_MT_FRAME_SIZE");
			Meta.mPixelMeta = SoyPixelsMeta(Width, Height, Format);
			if ( VerboseDebug )
				std::Debug << ErrorPrefix << " MF_MT_FRAME_SIZE = " << Meta.mPixelMeta << std::endl;
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	//	for 2d buffers, extract their pitch/stride. This reveals the padded width (which we cannot get directly from an IMFMediaBuffer)
	//	if it's negative, it also means it's flipped, so we can use that here too for the transform matrix
	//	https://msdn.microsoft.com/en-us/library/windows/desktop/aa473821(v=vs.85).aspx
	{
		try
		{
			//The minimum stride might be stored in the MF_MT_DEFAULT_STRIDE attribute.
			//If the MF_MT_DEFAULT_STRIDE attribute is not set, call the MFGetStrideForBitmapInfoHeader function to calculate the stride for most common video formats.
			LONG Pitch = 0;
			Pitch = MFGetAttributeUINT32( &MediaType, MF_MT_DEFAULT_STRIDE, Pitch );

			//	try and extract from format
			if ( Pitch == 0 )
			{
				//	FOURCC code or D3DFORMAT value that specifies the video format. If you have a video subtype GUID, you can use the first DWORD of the subtype.
				GUID SubType = {0};
				MediaType.GetGUID( MF_MT_SUBTYPE, &SubType );
				DWORD FormatFourcc = SubType.Data1;
				auto Width = Meta.mPixelMeta.GetWidth();
				auto Result = MFGetStrideForBitmapInfoHeader( FormatFourcc, Width, &Pitch );
				MediaFoundation::IsOkay( Result, ErrorPrefix + " MFGetStrideForBitmapInfoHeader()" );
			}
			
			//	if upside down, then set the transform matrix to flip it
			if ( Pitch < 0 )
			{
				Pitch = -Pitch;
				Meta.mTransform = SoyMath::GetFlipMatrix3x3();
			}
			
			if ( Pitch < Meta.mPixelMeta.GetWidth() )
			{
				std::stringstream Error;
				Error << "Extracted pitch of " << Pitch << " which is smaller than the image's width " << Meta.mPixelMeta.GetWidth();
				throw Soy::AssertException( Error.str() );
			}

			//	we may or may not have pixel format here, so store row size, not padding
			Meta.mPixelRowSize = Pitch;
		}
		catch(std::exception& e)
		{
			//	no pitch
			//	todo from docs:
			//	If the MF_MT_DEFAULT_STRIDE attribute is not set, call the MFGetStrideForBitmapInfoHeader function to calculate the stride for most common video formats.
		}
	}


	{
		UINT32 Numerator = 0;
		UINT32 Denominator = 0;
		try
		{
			auto Result = MFGetAttributeSize( &MediaType, MF_MT_FRAME_RATE, &Numerator, &Denominator );
			MediaFoundation::IsOkay(Result, ErrorPrefix + " MF_MT_FRAME_RATE");
			if ( Numerator == 0 || Denominator == 0 )
			{
				std::stringstream Error;
				Error << "Failed to decode frame rate from Numerator=" << Numerator << " & " << "Denominator=" << Denominator << " (would get div/0)";
				throw Soy::AssertException(Error.str());
			}
			Meta.mFramesPerSecond = (float)Numerator / (float)Denominator;
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	//	only in win8 sdk AND with win8 ifdef... 
	#if WINDOWS_TARGET_SDK>=8
	ReadAttrib32( MediaType, Meta.mVideoClockWiseRotationDegrees, MF_MT_VIDEO_ROTATION, ErrorPrefix, "MF_MT_VIDEO_ROTATION" );
	#endif

	{
		MFVideoInterlaceMode Mode;
		if ( ReadAttrib32( MediaType, Mode, MF_MT_INTERLACE_MODE, ErrorPrefix, "MF_MT_INTERLACE_MODE" ) )
		{
			//	gr: lots of options here!
			if ( Mode == MFVideoInterlace_Unknown )
				Mode = MFVideoInterlace_Progressive;
			Meta.mInterlaced = (Mode!=MFVideoInterlace_Progressive);
		}
	}
	
	ReadAttrib32( MediaType, Meta.mAverageBitRate, MF_MT_AVG_BITRATE, ErrorPrefix, "MF_MT_AVG_BITRATE" );
	ReadAttrib32( MediaType, Meta.mMaxKeyframeSpacing, MF_MT_MAX_KEYFRAME_SPACING, ErrorPrefix, "MF_MT_MAX_KEYFRAME_SPACING" );
	ReadAttrib32( MediaType, Meta.mAudioSampleRate, MF_MT_AUDIO_SAMPLES_PER_SECOND, ErrorPrefix, "MF_MT_AUDIO_SAMPLES_PER_SECOND" );
	
	ReadAttrib32( MediaType, Meta.mAudioSamplesIndependent, MF_MT_ALL_SAMPLES_INDEPENDENT, ErrorPrefix, "MF_MT_ALL_SAMPLES_INDEPENDENT" );


	{
		MFVideoDRMFlags Value;
		if ( ReadAttrib32( MediaType, Value, MF_MT_DRM_FLAGS, ErrorPrefix, "MF_MT_DRM_FLAGS" ) )
		{
			//	note: not storing analog vs digital protection here
			Meta.mDrmProtected = ( Value != MFVideoDRMFlag_None );
		}
	}


	{
		MFVideoTransferMatrix Mode;
		if ( ReadAttrib32( MediaType, Mode, MF_MT_YUV_MATRIX, ErrorPrefix, "MF_MT_YUV_MATRIX" ) )
		{
			//	gr: codec may already have been set here, so we're changing the RANGE, not the format!
			//	gr: sometimes we have invalid pixel format (mjpeg!), so cannot convert pixel format (invalid) to Full/Ntsc/SMPTE etc
			try
			{
				if ( Mode == MFVideoTransferMatrix_BT709 )
				{
					auto NewFormat = SoyPixelsFormat::GetYuvFull( Meta.mPixelMeta.GetFormat() );
					Meta.mPixelMeta.DumbSetFormat( NewFormat );
				}
				else if ( Mode == MFVideoTransferMatrix_BT601 )
				{
					auto NewFormat = SoyPixelsFormat::GetYuvNtsc( Meta.mPixelMeta.GetFormat() );
					Meta.mPixelMeta.DumbSetFormat( NewFormat );
				}
				else if ( Mode == MFVideoTransferMatrix_SMPTE240M )
				{
					auto NewFormat = SoyPixelsFormat::GetYuvSmptec( Meta.mPixelMeta.GetFormat() );
					Meta.mPixelMeta.DumbSetFormat( NewFormat );
				}
				else if ( Mode != MFVideoTransferMatrix_Unknown )
				{
					throw Soy::AssertException("Unknown matrix format");
				}
			}
			catch(std::exception& e)
			{
				if ( VerboseDebug )
					std::Debug << "YUV mode ignored when converting from " << Meta.mPixelMeta.GetFormat() << " to " << Mode << ": " << e.what() << std::endl;
			}
		}
	}

	return Meta;
}


TStreamMeta MediaFoundation::GetStreamMeta(IMFMediaType& MediaType,size_t StreamIndex,size_t MediaTypeIndex,IMFSourceReader& Reader,bool VerboseDebug)
{
	auto Meta = GetStreamMeta( MediaType, VerboseDebug );

	Meta.mStreamIndex = StreamIndex;
	Meta.mMediaTypeIndex = MediaTypeIndex;

	std::string ErrorPrefix = Meta.mDescription + " stream";

	{
		PROPVARIANT Prop;
		auto Result = Reader.GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &Prop );
		try
		{
			MediaFoundation::IsOkay(Result, ErrorPrefix + " MF_PD_DURATION");
			auto Duration100Ns = Prop.uhVal.QuadPart;
			Meta.mDuration.SetNanoSeconds( Duration100Ns * 100 );
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	if ( SoyMediaFormat::IsAudio( Meta.mCodec ) )
	{
		PROPVARIANT Prop;
		auto Result = Reader.GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_PD_AUDIO_ENCODING_BITRATE, &Prop );
		try
		{
			MediaFoundation::IsOkay(Result, ErrorPrefix + " MF_PD_AUDIO_ENCODING_BITRATE");
			Meta.mEncodingBitRate = Prop.uhVal.QuadPart;
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	if ( SoyMediaFormat::IsVideo( Meta.mCodec ) )
	{
		PROPVARIANT Prop;
		auto Result = Reader.GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_PD_VIDEO_ENCODING_BITRATE, &Prop );
		try
		{
			MediaFoundation::IsOkay(Result, ErrorPrefix + " MF_PD_VIDEO_ENCODING_BITRATE");
			Meta.mEncodingBitRate = Prop.uhVal.QuadPart;
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	return Meta;
}


void MediaFoundation::EnumStreams(ArrayBridge<TStreamMeta>& Streams,IMFSourceReader& Reader,bool VerboseDebug)
{
	//	https://msdn.microsoft.com/en-us/library/dd389281(VS.85).aspx
	DWORD StreamIndex = 0;
	DWORD MediaTypeIndex = 0;

	//	in case we get stuck in some loop, stream count is large for devices
	static DWORD MaxStreamIndex = 200;
	static DWORD MaxMediaType = 200;

	while ( StreamIndex < MaxStreamIndex && MediaTypeIndex < MaxMediaType )
	{
		Soy::AutoReleasePtr<IMFMediaType> MediaType;
        auto Result = Reader.GetNativeMediaType( StreamIndex, MediaTypeIndex, &MediaType.mObject );

		//	no more streams
		if ( Result == MF_E_INVALIDSTREAMNUMBER )
			break;

		//	no more entries for this stream
		if ( Result == MF_E_NO_MORE_TYPES )
		{
			MediaTypeIndex = 0;
			StreamIndex++;
			continue;
		}

		//	some error. if device loses power/disconnected during this loop
		if ( Result != S_OK )
		{
			auto Error = Platform::GetErrorString( Result );
			std::Debug << "Error getting streams meta " << StreamIndex << "," << MaxMediaType << ": " << Error << std::endl;
			break;
		}

		if ( MediaType )
		{
			//	gr: drop streams we fail to parse
			try
			{
				TStreamMeta Meta = GetStreamMeta( *MediaType.mObject, StreamIndex, MediaTypeIndex, Reader, VerboseDebug );
				Streams.PushBack( Meta );
			}
			catch( std::exception& e)
			{
				if ( VerboseDebug )
					std::Debug << "Failed to parse stream meta for StreamIndex=" << StreamIndex << ", MediaTypeIndex=" << MediaTypeIndex << ". Skipping stream" << std::endl;
			}
		}
		MediaTypeIndex++;
    }
}



MfByteStream::MfByteStream(const std::string& Filename) :
	mFilename(Filename)
{
	MF_FILE_ACCESSMODE Access = MF_ACCESSMODE_READ;
	MF_FILE_OPENMODE Open = MF_OPENMODE_FAIL_IF_NOT_EXIST;
	MF_FILE_FLAGS Flags = MF_FILEFLAGS_NONE;
	auto FilenameW = Soy::StringToWString( Filename );
	auto Result = MFCreateFile(Access, Open, Flags, FilenameW.c_str(), &mByteStream.mObject );
	MediaFoundation::IsOkay(Result, "MFCreateFile");
}


MfExtractor::MfExtractor(const TMediaExtractorParams& Params) :
	TMediaExtractor			( Params ),
	mMediaFoundationContext	( MediaFoundation::GetContext() ),
	mFilename				( Params.mFilename ),
	mAsyncReadSampleRequests	( 0 )
{
	Soy::Assert( mMediaFoundationContext != nullptr, "Missing Media foundation context");
}

void MfExtractor::Init()
{
	CreateSourceReader( mFilename );
	Start();
	TriggerAsyncRead();
}

MfExtractor::~MfExtractor()
{
	WaitToFinish();

	//	release last context
	//	gr: I'm sure I did this inside the unity/plugin code before?
	mMediaFoundationContext.reset();
	MediaFoundation::Shutdown();
}

HRESULT GetSourceFlags(IMFSourceReader *pReader, ULONG *pulFlags)
{
    ULONG flags = 0;

    PROPVARIANT var;
    PropVariantInit(&var);

    HRESULT hr = pReader->GetPresentationAttribute(
        MF_SOURCE_READER_MEDIASOURCE, 
        MF_SOURCE_READER_MEDIASOURCE_CHARACTERISTICS, 
        &var);

    if (SUCCEEDED(hr))
    {
		flags = var.ulVal;
      //  hr = PropVariantToUInt32(var, &flags);
    }
    if (SUCCEEDED(hr))
    {
        *pulFlags = flags;
    }

    PropVariantClear(&var);
    return hr;
}


bool MfExtractor::CanSeek()
{
	//	allow arbritry setting if we have no reader
	if ( !mSourceReader )
		return true;

	if ( !mParams.mAllowReseek )
		return false;

	bool CanSeek = false;
	ULONG flags = 0;
    if ( S_OK == GetSourceFlags(mSourceReader, &flags) )
    {
		CanSeek = ((flags & MFMEDIASOURCE_CAN_SEEK) == MFMEDIASOURCE_CAN_SEEK);
    }
    return CanSeek;
}

bool MfExtractor::OnSeek()
{
	if ( !CanSeek() )
		return false;

	static bool DebugSeek = false;
	Soy::Assert( mSourceReader!=nullptr, "Source reader missing");

	//	this may need to be at least KeyFrameDifference ms
	static int MinSeekDiffForwardMs = 1000;
	static int MinSeekDiffBackwardMs = -1000;
	auto SeekTime = GetSeekTime();

	auto SeekRealTime = GetExtractorRealTimecode( SeekTime );
	auto Diff = SeekRealTime.GetDiff( SoyTime(mLastReadTime) );

	if ( Diff == 0 )
		return false;
	if ( Diff > MinSeekDiffBackwardMs && Diff < MinSeekDiffForwardMs )
	{
		if ( DebugSeek )
			std::Debug << "Skipping MFExtractor seek, difference only " << Diff << "/(" << MinSeekDiffBackwardMs << "..." << MinSeekDiffForwardMs << ")" << std::endl;
		return false;
	}
	
	mPendingSeek = SeekRealTime;
	try
	{
		ProcessPendingSeek();

		//	need to start reading again otherwise nothing triggers it
		TriggerAsyncRead();
	}
	catch(std::exception& e)
	{
		std::Debug << "Seek to " << SeekRealTime << " deffered: " << e.what() << std::endl;
	}
	return true;
}


void MediaFoundation::GetSupportedFormats(ArrayBridge<SoyPixelsFormat::Type>&& Formats)
{
	//	gr: this is more to do with the parent decoder and it's shader support... so this applies to all platforms...
	Formats.PushBack( SoyPixelsFormat::Yuv_8_8_8_Full );
	Formats.PushBack( SoyPixelsFormat::Yuv_8_8_8_Ntsc );
	Formats.PushBack( SoyPixelsFormat::Yuv_8_8_8_Smptec );
	Formats.PushBack( SoyPixelsFormat::RGBA );
	Formats.PushBack( SoyPixelsFormat::RGB );
	Formats.PushBack( SoyPixelsFormat::Yuv_8_88_Full );
	Formats.PushBack( SoyPixelsFormat::Yuv_8_88_Ntsc );
	Formats.PushBack( SoyPixelsFormat::Yuv_8_88_Smptec );

	Formats.PushBack( SoyPixelsFormat::YYuv_8888_Full );
	Formats.PushBack( SoyPixelsFormat::YYuv_8888_Ntsc );
	Formats.PushBack( SoyPixelsFormat::YYuv_8888_Smptec );
}


void MfExtractor::CreateSourceReader(const std::string& Filename)
{
	//	alloc the async callback
	if ( !mSourceReaderCallback )
	{
		mSourceReaderCallback.Set( new MFExtractorCallback(*this), true );
	}

	AllocSourceReader( Filename );
	Soy::Assert( mSourceReader != nullptr, "Failed to allocate source reader" );

	//	enum the streams so we try and convert the right one
	Array<TStreamMeta> Streams;
	MediaFoundation::EnumStreams( GetArrayBridge(Streams), *mSourceReader, mParams.mVerboseDebug );
	FilterStreams( GetArrayBridge(Streams) );

	//	configure the output we want
	//	capture errors in case we come out with no streams at all
	std::stringstream CreateStreamsError;
	for ( int s=0;	s<Streams.GetSize();	s++ )
	{
		try
		{
			ConfigureStream( Streams[s] );
		}
		catch(std::exception& e)
		{
			CreateStreamsError << "Error creating stream " << Streams[s] << ": " << e.what() << ";";
		}
	}

	//	throw an error if we fail to create any streams at all
	if ( mStreams.empty() )
	{
		std::stringstream Error;
		Error << "Created 0/" << Streams.GetSize() << " streams; " << CreateStreamsError.str();
		throw Soy::AssertException( Error.str() );
	}

	//	gr: changed this to disable all then enable as configuration succeeds
	{
		auto Result = mSourceReader->SetStreamSelection( MF_SOURCE_READER_ALL_STREAMS, false );
		MediaFoundation::IsOkay( Result, "SetStreamSelection MF_SOURCE_READER_ALL_STREAMS = false" );
	}

	//	enable streams after we've resolved the final ones we want
	for ( auto it=mStreams.begin();	it!=mStreams.end();	it++ )
	{
		auto& StreamIndex = it->first;
		auto& StreamMeta = it->second;

		auto Result = mSourceReader->SetStreamSelection( size_cast<DWORD>(StreamIndex), true );
		std::stringstream Error;
		Error << "SetStreamSelection(" << StreamIndex << ")=true";
		MediaFoundation::IsOkay( Result, Error.str() );
	}

	{
		Array<TStreamMeta> Streams;
		GetStreams( GetArrayBridge(Streams) );
		OnStreamsChanged( GetArrayBridge(Streams) );
	}
}

void MfExtractor::ConfigureStream(TStreamMeta Stream)
{
	//	if we've already picked a stream for this stream index then skip it
	{
		auto Existing = mStreams.find( Stream.mStreamIndex );
		if ( Existing != mStreams.end() )
		{
			std::Debug << "Already configured stream for stream index " << Stream.mStreamIndex << std::endl;
			return;
		}
	}

	if ( SoyMediaFormat::IsVideo( Stream.mCodec ) )
	{
		ConfigureVideoStream( Stream );
	}
	else if ( SoyMediaFormat::IsAudio( Stream.mCodec ) )
	{
		ConfigureAudioStream( Stream );
	}
	else
	{
		ConfigureOtherStream( Stream );
	}
}

void MfExtractor::ConfigureVideoStream(TStreamMeta& Stream)
{
	{
		std::stringstream Error;
		Error << "Trying to use video stream with invalid dimensions: " << Stream.mPixelMeta;
		Soy::Assert( Stream.mPixelMeta.IsValidDimensions(), Error.str() );
	}

	//	https://msdn.microsoft.com/en-us/library/dd389281(VS.85).aspx
	//	To get data directly from the source without decoding it, use one of the types returned by GetNativeMediaType.
	//	To decode the stream, create a new media type that describes the desired uncompressed format.

	//	try different formats until we find one it accepts conversion to
	//	gr: this may need to change to get Soy formats, then get ALL the MF formats that converts to, then try THEM
	Array<SoyPixelsFormat::Type> Formats;
	MediaFoundation::GetSupportedFormats( GetArrayBridge(Formats) );
	
	//	start with it's current format for least conversion! (as long as we support it)
	//if ( Formats.Find( Stream.mPixelMeta.GetFormat() ) )
	//	*Formats.InsertBlock( 0, 1 ) = Stream.mPixelMeta.GetFormat();

	auto FinalFormat = SoyPixelsFormat::Invalid;
	
	while ( FinalFormat == SoyMediaFormat::Invalid && !Formats.IsEmpty() )
	{
		auto TryFormat = Formats.PopAt(0);
		auto OutputType = MediaFoundation::GetPlatformFormat( TryFormat );
		auto Result = mSourceReader.mObject->SetCurrentMediaType( size_cast<DWORD>(Stream.mStreamIndex), nullptr, OutputType.mObject );

		std::stringstream Error;
		Error << "SetCurrentMediaType (" << TryFormat << "); ";
		if ( Result == MF_E_TOPO_CODEC_NOT_FOUND )
			Error << "Decoder codec not found";
		else if ( Result == MF_E_INVALIDMEDIATYPE )
			Error << "Decoder found, but cannot convert";

		try
		{
			MediaFoundation::IsOkay(Result, Error.str());
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
			continue;
		}

		//	meta might have changed?
		TStreamMeta NewStreamMeta = MediaFoundation::GetStreamMeta( *OutputType.mObject, Stream.mStreamIndex, Stream.mMediaTypeIndex, *mSourceReader, mParams.mVerboseDebug );

		//	force override
		if ( mParams.mForceYuvColourFormat != SoyPixelsFormat::Invalid )
		{
			try
			{
				auto NewTryFormat = SoyPixelsFormat::ChangeYuvColourRange( TryFormat, mParams.mForceYuvColourFormat );
				TryFormat = NewTryFormat;
				std::Debug << "Forcing YUV format from " << TryFormat << " to " << NewTryFormat << std::endl;
			}
			catch(std::exception& e)
			{
				std::Debug << "Failed to forcing YUV format; " << e.what() << std::endl;
			}
		}

		//	stream's format has changed with SetCurrentMediaType
		//	gr: verify against NewStreamMeta?
		Stream.mPixelMeta.DumbSetFormat( TryFormat );
		FinalFormat = TryFormat;
	}

	//	gr: do we fail here, or let it fall through with the "unsupported format"
	static bool AllowUnsupportedFormat = false;
	if ( !AllowUnsupportedFormat )
	{
		//	this should set to invalid, or just set again to what we determined in the loop
		Stream.mPixelMeta.DumbSetFormat( FinalFormat );
	}

	Soy::Assert( Stream.mPixelMeta.IsValid(), "Failed to find pixel format decoder accepts" );

	//	on success, save the meta we've used
	mStreams[Stream.mStreamIndex] = Stream;
}

class TAudioTryFormat
{
public:
	TAudioTryFormat(const GUID& Guid,size_t FormatSize,size_t SampleRate,size_t ChannelCount) :
		mGuid			( Guid ),
		mFormatSize		( FormatSize ),
		mSampleRate		( SampleRate ),
		mChannelCount	( ChannelCount )
	{
	}
	TAudioTryFormat(){}
	
	GUID	mGuid;
	size_t	mFormatSize;
	size_t	mSampleRate;
	size_t	mChannelCount;
};

void MfExtractor::ConfigureAudioStream(TStreamMeta& Stream)
{
	if ( !mParams.mExtractAudioStreams )
		return;

	//	try and get the extractor to convert to a nicer format (and the format we desire)
	BufferArray<TAudioTryFormat,20> TryFormats;

	//	if we're splitting audio, don't decode to the desired number of streams
	bool Split = mParams.mSplitAudioChannelsIntoStreams;
	size_t NewChannelCount = Split ? Stream.mChannelCount : mParams.mAudioChannelCount;
	size_t BackupChannelCount = Split ? mParams.mAudioChannelCount : Stream.mChannelCount;

	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_Float,0,mParams.mAudioSampleRate,NewChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_Float,0,mParams.mAudioSampleRate,BackupChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_Float,0,0,NewChannelCount) );

	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,4,mParams.mAudioSampleRate,NewChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,2,mParams.mAudioSampleRate,NewChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,1,mParams.mAudioSampleRate,NewChannelCount) );

	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,4,mParams.mAudioSampleRate,BackupChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,2,mParams.mAudioSampleRate,BackupChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,1,mParams.mAudioSampleRate,BackupChannelCount) );

	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,4,0,mParams.mAudioChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,2,0,mParams.mAudioChannelCount) );
	TryFormats.PushBack( TAudioTryFormat(MFAudioFormat_PCM,1,0,mParams.mAudioChannelCount) );


	auto FinalFormat = SoyMediaFormat::Invalid;
	
	while ( FinalFormat == SoyMediaFormat::Invalid && !TryFormats.IsEmpty() )
	{
		auto TryFormat = GetArrayBridge(TryFormats).PopAt(0);
		try
		{
			//	gr: we are telling it which format to use, so... not sure. guess we'll find out after
			//	gr: hacky! sort this out
			auto FormatSize = TryFormat.mFormatSize;
			auto TryFormatType = MediaFoundation::GetFormat( MFMediaType_Audio, TryFormat.mGuid, FormatSize );

			bool IsSameFormat = (TryFormatType == Stream.mCodec) && (TryFormat.mChannelCount==Stream.mChannelCount) && (TryFormat.mSampleRate==Stream.mAudioSampleRate);

			static bool ConvertIfSameFormat = false;
			if ( !IsSameFormat || ConvertIfSameFormat )
			{
				auto OutputType = MediaFoundation::GetPlatformFormat( MFMediaType_Audio, TryFormat.mGuid );

				//	set some additional properties
				if ( TryFormat.mChannelCount != 0 )
				{
					auto Result = OutputType.mObject->SetUINT32( MF_MT_AUDIO_NUM_CHANNELS, TryFormat.mChannelCount );
					MediaFoundation::IsOkay( Result, "set MF_MT_AUDIO_NUM_CHANNELS" );
				}

				if ( TryFormat.mSampleRate != 0 )
				{
					auto Result = OutputType.mObject->SetUINT32( MF_MT_AUDIO_SAMPLES_PER_SECOND, TryFormat.mSampleRate );
					MediaFoundation::IsOkay( Result, "set MF_MT_AUDIO_SAMPLES_PER_SECOND" );
				}

				auto Result = mSourceReader.mObject->SetCurrentMediaType( size_cast<DWORD>(Stream.mStreamIndex), nullptr, OutputType.mObject );

				std::stringstream Error;
				Error << "SetCurrentMediaType (" << TryFormatType << "); ";
				if ( Result == MF_E_TOPO_CODEC_NOT_FOUND )
					Error << "Decoder codec not found";
				else if ( Result == MF_E_INVALIDMEDIATYPE )
					Error << "Decoder found, but cannot convert";

				try
				{
					MediaFoundation::IsOkay(Result, Error.str());
				}
				catch ( std::exception& e )
				{
					std::Debug << e.what() << std::endl;
					continue;
				}

				//	meta might have changed?
				//	gr: changing audio format & channels should change meta
				//		seeing as we used this to get the old stream, it shouldn't lose any data. Look out for this!
				try
				{
					TStreamMeta NewStreamMeta = MediaFoundation::GetStreamMeta( *OutputType.mObject, Stream.mStreamIndex, Stream.mMediaTypeIndex, *mSourceReader, mParams.mVerboseDebug );
					Stream = NewStreamMeta;
				}
				catch(std::exception& e)
				{
					std::Debug << "Error fetching new audio meta: " << e.what() << std::endl;
				}
				//	stream's format has changed with SetCurrentMediaType
				//	gr: verify against NewStreamMeta?
				Stream.mCodec = TryFormatType;
			}

			FinalFormat = TryFormatType;
		}
		catch(std::exception& e)
		{
			std::Debug << "Exception trying to setup stream (" << Stream.mCodec << ") as format " << TryFormat.mGuid << "(rate=" << TryFormat.mSampleRate << ", channels=" << TryFormat.mChannelCount << ")" << std::endl;
		}
	}

	//	gr: do we fail here, or let it fall through with the "unsupported format"
	static bool AllowUnsupportedFormat = false;
	if ( !AllowUnsupportedFormat )
	{
		//	this should set to invalid, or just set again to what we determined in the loop
		Stream.mCodec = FinalFormat;
	}

	Soy::Assert( Stream.mCodec != SoyMediaFormat::Invalid, "Failed to find audio format decoder accepts" );

	//	on success, save the meta we've used
	mStreams[Stream.mStreamIndex] = Stream;
}

void MfExtractor::ConfigureOtherStream(TStreamMeta& Stream)
{
	std::Debug << Stream.mCodec << " stream is currently unsupported" << std::endl;
}


void MfExtractor::PushPacket(std::shared_ptr<TMediaPacket>& Sample)
{
	Soy::Assert( Sample != nullptr, "MfExtractor::PushPacket Expected non-null sample");

	{
		std::lock_guard<std::mutex> Lock( mPacketQueueLock );
		CorrectIncomingTimecode( *Sample );
		mPacketQueue.PushBack( Sample );
	}
	OnPacketExtracted( Sample );
}

void MfExtractor::TriggerAsyncRead()
{
	if ( !mSourceReader )
		return;

	try
	{
		//	if the pending seek fails, don't read another sample
		ProcessPendingSeek();
		
		DWORD ReadFlags = 0;
		auto Result = mSourceReader->ReadSample( MF_SOURCE_READER_ANY_STREAM, ReadFlags, nullptr, nullptr, nullptr, nullptr );
		MediaFoundation::IsOkay( Result, "TriggerAsyncRead");
		mAsyncReadSampleRequests++;
	}
	catch(std::exception& e)
	{
		std::Debug << "TriggerAsyncRead failed:" << e.what() << std::endl;
	}
}

void MfExtractor::ProcessPendingSeek()
{
	//	no seek required
	if ( !mPendingSeek.IsValid() )
		return;

	PropVariantTime Time( mPendingSeek );

	const GUID* TimeFormat = nullptr;
	//if ( !mSourceReaderLock.try_lock();
	auto Result = mSourceReader->SetCurrentPosition( GUID_NULL, Time.mProp );

	//	sample request still pending https://msdn.microsoft.com/en-us/library/windows/desktop/dd374668(v=vs.85).aspx
	if ( Result == MF_E_INVALIDREQUEST )
	{
		throw Soy::AssertException("Cannot seek, sample request still pending");
	}

	//mSourceReaderLock.unlock();
	std::stringstream Error;
	Error << "Seeking MFExtractor to (real)" << mPendingSeek;
	MediaFoundation::IsOkay( Result, Error.str() );

	//	reset read time
	//	really need a "unknown, don't seek yet" value...
	mLastReadTime = mPendingSeek.GetMilliSeconds();

	mPendingSeek = SoyTime();

	//	flush packets in the queue
	if ( !mPacketQueue.IsEmpty() )
	{
		std::lock_guard<std::mutex> Lock( mPacketQueueLock );
		mPacketQueue.Clear();
	}
}


bool MfExtractor::CanSleep()
{
	if ( !mPacketQueue.IsEmpty() )
		return false;

	return TMediaExtractor::CanSleep();
}

std::shared_ptr<TMediaPacket> MfExtractor::ReadNextPacket()
{
	bool DoPostSyncRead = mPendingSeek.IsValid();
	try
	{
		ProcessPendingSeek();
	}
	catch(std::exception& e)
	{
		//	pending seek so... flush packet queue?
		std::Debug << "ReadNextPacket ProcessPending seek failed. (" << e.what() << ") Flush x" << mPacketQueue.GetSize() << " queue?" << std::endl;
		return nullptr;
	}

	//	not ready... or error?
	if ( !mSourceReader )
		return nullptr;

	std::lock_guard<std::mutex> Lock( mPacketQueueLock );

	if ( mPacketQueue.IsEmpty() )
	{
		//	if we've resync'd we NEED to trigger another read... (because we drop pending-seek frames?)
		if ( DoPostSyncRead )
			TriggerAsyncRead();
		
		//	seem to be getiting stuck with capture camera even though there's one pending..
		static int MaxQueuedRequests = 1;
		if ( mAsyncReadSampleRequests < MaxQueuedRequests )
			TriggerAsyncRead();

		return nullptr;
	}

	//	this won't block, but request another packet, so okay here
	TriggerAsyncRead();
	return mPacketQueue.PopAt(0);
}


HRESULT MFExtractorCallback::OnReadSample(HRESULT hrStatus,DWORD dwStreamIndex,
        DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample *pSample)
{
	//	Returns an HRESULT value. Currently, the source reader ignores the return value.
	//	https://msdn.microsoft.com/en-us/library/windows/desktop/dd374658(v=vs.85).aspx

	mParent.mAsyncReadSampleRequests--;

	bool Eof = bool_cast( dwStreamFlags & MF_SOURCE_READERF_ENDOFSTREAM );
	auto Tick = bool_cast( dwStreamFlags & MF_SOURCE_READERF_STREAMTICK );

	//	pure tick
	if ( !pSample && !Eof && Tick )
	{
		//	web cam sends a null sample tick, gotta trigger parent to request another
		mParent.TriggerAsyncRead();
		return S_OK;
	}

	//	dropped frame but not eof
	if ( !pSample && !Eof )
	{
		//	dropped frame, parent may just sit here...
		std::Debug << "Dropped frame: flags=" << dwStreamFlags << std::endl;
		return S_OK;
	}
	
	//	gr: if eof... send it anyway?
	if ( hrStatus != S_OK )
		return S_OK;

	//	retain sample
	Soy::AutoReleasePtr<IMFSample> Sample( pSample, true );

	SoyTime Timestamp;
	auto TimeStampNs100 = llTimestamp;
	Timestamp.SetNanoSeconds( TimeStampNs100*100 );
	
	try
	{
		mParent.PushPacket( Sample, Timestamp, Eof, dwStreamIndex );
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception pushing sample " << e.what() << std::endl;
	}

	return S_OK;
}


void MfExtractor::PushPacket(Soy::AutoReleasePtr<IMFSample> Sample,SoyTime Timestamp,bool Eof,size_t StreamIndex)
{
	if ( mPendingSeek.IsValid() )
	{
		std::Debug << "Pending seek " << mPendingSeek << " so dropping packet " << Timestamp << std::endl;
		return;
	}
	mLastReadTime = Timestamp.GetMilliSeconds();

	if ( Eof )
	{
		std::shared_ptr<TMediaPacket> pPacket( new TMediaPacket );
		auto& Packet = *pPacket;

		//	added try/catch around this as we DEFINTELY want to send this, just in case we've been giving a sample we're ignoring
		try
		{
			auto& StreamMeta = GetStreamMeta( StreamIndex );
			Packet.mMeta = StreamMeta;
		}
		catch(...)
		{
			Packet.mMeta.mStreamIndex = StreamIndex;
		}
		Packet.mEof = true;
		Packet.mTimecode = Timestamp;
		PushPacket( pPacket );
		return;
	}

	if ( !Sample )
	{
		//	 MF_SOURCE_READERF_ENDOFSTREAM flag in pdwStreamFlags and sets ppSample to NULL.
		//	if there is a gap in the stream, pdwStreamFlags receives the MF_SOURCE_READERF_STREAMTICK flag, ppSample is NULL, and pllTimestamp indicates the time when the gap occurred.
		std::Debug << "ReadSample return null sample (no error) (" << Timestamp << ") eof=" << Eof;
		return;
	}

	//	currently gives us nothing but missing attribs
	//auto SampleMeta = GetStreamMeta( *Sample.mObject );
	auto& StreamMeta = GetStreamMeta( StreamIndex );

	//	for debugging in case we get multiple planes or audio samples meta
	{
		DWORD BufferCount = 0;
		auto Result = Sample->GetBufferCount( &BufferCount );
		
		try
		{
			MediaFoundation::IsOkay(Result, "Sample::GetBufferCount");
			if ( BufferCount > 1 )
				std::Debug << "Sample buffer count ==" << BufferCount << "; for stream " << StreamMeta << std::endl;
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	//	gr: this is a bit hacky, what do we do on the other platforms?
	if ( SoyMediaFormat::IsVideo( StreamMeta.mCodec ) )
	{
		std::shared_ptr<TMediaPacket> pPacket( new TMediaPacket );
		auto& Packet = *pPacket;
		Packet.mMeta = StreamMeta;
		Packet.mPixelBuffer.reset( new MfPixelBuffer(Sample, StreamMeta, mParams.mApplyHeightPadding, mParams.mApplyWidthPadding, mParams.mWin7Emulation ) );
		Packet.mTimecode = Timestamp;
		PushPacket( pPacket );
		return;
	}
	else if ( SoyMediaFormat::IsAudio( StreamMeta.mCodec ) )
	{
		std::shared_ptr<TMediaPacket> pPacket( new TMediaPacket );
		auto& Packet = *pPacket;
		Packet.mMeta = StreamMeta;
		Packet.mTimecode = Timestamp;
		MediaFoundation::GetSampleData( GetArrayBridge( Packet.mData ), *Sample.mObject );
		PushPacket( pPacket );
		return;
	}

	std::stringstream Error;
	Error << "Don't know what to do with with stream sample from " << StreamMeta << std::endl;
	throw Soy::AssertException( Error.str() );
}

const TStreamMeta& MfExtractor::GetStreamMeta(size_t StreamIndex)
{
	auto StreamIt = mStreams.find(StreamIndex);
	if ( StreamIt == mStreams.end() )
	{
		std::stringstream Error;
		Error << "No stream #" << StreamIndex;
		throw Soy::AssertException( Error.str() );
	}
	return StreamIt->second;
}


void MfExtractor::GetStreams(ArrayBridge<TStreamMeta>&& Streams)
{
	for ( auto& Stream : mStreams )
	{
		Streams.PushBack( Stream.second );
	}
}
	
MfPixelBuffer::MfPixelBuffer(Soy::AutoReleasePtr<IMFSample>& Sample,const TStreamMeta& Meta,bool ApplyHeightPadding,bool ApplyWidthPadding,bool Win7Emulation) :
	mMeta				( Meta ),
	mSample				( Sample ),
	mApplyWidthPadding	( ApplyWidthPadding ),
	mApplyHeightPadding	( ApplyHeightPadding ),
	mWin7Emulation		( Win7Emulation )
{
	Soy::Assert( mSample, "Sample expected" );
}

MfPixelBuffer::~MfPixelBuffer()
{
	if ( mLockedMediaBuffer )
		std::Debug << __func__ << " mLockedMediaBuffer not released" << std::endl;
	if ( mLockedBuffer2D )
		std::Debug << __func__ << " mLockedBuffer2D not released" << std::endl;
	if ( !mLockedPixels.IsEmpty() )
		std::Debug << __func__ << " mLockedPixels not released" << std::endl;

	mSample.Release();
}

void MfPixelBuffer::GetMediaBuffer(Soy::AutoReleasePtr<IMFMediaBuffer>& Buffer)
{
	Soy::Assert( mSample, "Sample expected" );

	//	get buffer as contigious. if this ever fails, revert back to multiple buffer stitching (locally?)
	IMFMediaBuffer* pBuffer = nullptr;
	auto Result = mSample->ConvertToContiguousBuffer( &pBuffer );
	MediaFoundation::IsOkay(Result, "ConvertToContiguousBuffer" );
	Soy::Assert(pBuffer!=nullptr, "Missing Media buffer object");

	//	ConvertToContiguousBuffer automatically retains
	Buffer.Set( pBuffer, false );
	//MediaBuffer.Retain();
}


void MfPixelBuffer::Lock(ArrayBridge<Directx::TTexture>&& Textures,Directx::TContext& Context,float3x3& Transform)
{
	/*
	auto MediaBuffer = GetMediaBuffer();

	//	get dx surface
//	IDirect3DSurface9* pSurface = nullptr;
//	auto Result = MFGetService( MediaBuffer.mObject, MR_BUFFER_SERVICE, __uuidof(IDirect3DSurface9), (void**)&pSurface );

	//	gr: find out where this is
	IDirect3DSurface11* pSurface = nullptr;
	auto Result = MFGetService( MediaBuffer.mObject, MR_BUFFER_SERVICE, __uuidof(IDirect3DSurface11), (void**)&pSurface );
	MediaFoundation::IsOkay( Result, "Get Directx surface service");
	*/
}


size_t RescalePitch(size_t Pitch,const SoyPixelsImpl& Pixels)
{
	size_t ChannelCount = 0;

	//	see if the current format rescales (single planar YUV does, as does RGB. biplanar YUV wont)
	try
	{
		ChannelCount = Pixels.GetChannels();
	}
	catch(std::exception& e)
	{
		//	try splitting
		BufferArray<std::shared_ptr<SoyPixelsImpl>,4> PixelPlanes;
		auto& MutablePixels = const_cast<SoyPixelsImpl&>( Pixels );
		MutablePixels.SplitPlanes( GetArrayBridge(PixelPlanes) );

		if ( PixelPlanes.GetSize() == 0 || PixelPlanes[0] == nullptr )
		{
			std::stringstream Error;
			Error << "Failed to split planes of " << Pixels.GetMeta() << " to get pitch alignment";
			throw Soy::AssertException( Error.str() );
		}

		auto PitchMeta = PixelPlanes[0]->GetMeta();
		ChannelCount = PitchMeta.GetChannels();
	}

	if ( ChannelCount == 0 )
	{
		std::stringstream Error;
		Error << "Failed to get channel count for pitch(" << Pitch << ") rescaling of " << Pixels.GetMeta();
		throw Soy::AssertException( Error.str() );
	}
	
	//	check alignment, we have to do this with the first plane (no channel count for YUV formats)
	auto ChannelOverflow = Pitch % ChannelCount;
	if ( ChannelOverflow > 0 )
	{
		std::stringstream Error;
		Error << "Image pitch (" << Pitch << ") doesn't align to channel count(" << ChannelCount << ") for " << Pixels.GetMeta();
		throw Soy::AssertException( Error.str() );
	}
	
	auto NewPitch = Pitch / ChannelCount;
	return NewPitch;
}

void MfPixelBuffer::ApplyPadding(SoyPixelsMeta& Meta,float3x3& Transform,size_t Pitch,size_t DataSize)
{
	SoyPixelsMeta OrigMeta = Meta;
	//	gr: refactor this to find all possible meta combinations and find the best one that fits in datasize

	if ( mApplyHeightPadding )
	{
		//	gr: some NV12 formats (1080 height) aren't aligning (chroma is off)
		//	this page says IMC1 has to align to 16 lines, but maybe nv12 does too
		//	https://msdn.microsoft.com/en-us/library/windows/hardware/ff538197(v=vs.85).aspx
		//	may need to only apply this if the plane split doesn't align?
		static size_t HeightAlignment = 16;
		auto OldHeight = Meta.GetHeight();
		auto HeightOverflow = Meta.GetHeight() % HeightAlignment;
		auto HeightPadding = (HeightOverflow > 0) ? (HeightAlignment - HeightOverflow) : 0;
		if ( HeightPadding != 0 )
		{
			//	gr: vive camera (620x460x YYuv8888) doesn't align to 16 lines and overflows if we do...
			//	we'll still try it, but test the meta first
			try
			{
				SoyPixelsMeta NewMeta = Meta;
				NewMeta.DumbSetHeight( Meta.GetHeight() + HeightPadding );

				//	check alignment
				BufferArray<std::tuple<size_t,size_t,SoyPixelsMeta>,10> PlaneOffsetSizeAndMetas;
				NewMeta.SplitPlanes( DataSize, GetArrayBridge(PlaneOffsetSizeAndMetas) );
				Meta = NewMeta;
			}
			catch(std::exception& e)
			{
				std::Debug << "Aligning height from " << OldHeight << " to " << (OldHeight+HeightPadding) << " failed: " << e.what() << std::endl;
			}
		}

	}

	//	fix this in caller
	if ( Pitch == 0 )
	{
		std::stringstream Error;
		Error << "Pitch is zero (for " << Meta << ")";
		throw Soy::AssertException( Error.str() );
	}

	//	gr: isight camera on windows gives us 640x480 yuv844 but meta tells us its 740x568, so crop
	if ( Pitch < Meta.GetWidth() )
	{
		std::Debug << "Warning, pitch(" << Pitch << ") < width (" << Meta << ") - cropping width" << std::endl;
		//Pitch = Meta.GetWidth();
		Meta.DumbSetWidth( Pitch );

		//	as width is smaller, check data size in case we should clip the height too
		if ( Meta.GetDataSize() > DataSize )
		{
			try
			{
				auto ChannelCount = Meta.GetChannels();
				//	calc height
				auto Height = DataSize / ChannelCount / Meta.GetWidth();
				std::Debug << "Pitch/width/data size clipped height from " << Meta.GetHeight() << " to " << Height << std::endl;
				Meta.DumbSetHeight( Height );
			}
			catch(std::exception& e)
			{
				std::Debug << "Pitch/width/datasize mismatch height clip failed: " << e.what() << std::endl;
			}
		}
	}

	auto WidthPadding = Pitch - Meta.GetWidth();
	//Soy::Assert( WidthPadding == mMeta.mPixelWidthPadding, "Extracted pitch mismatch with stream meta");
	if ( WidthPadding > 0 )
	{
		//	clip padding with transform
		Meta.DumbSetWidth( Meta.GetWidth() + WidthPadding );
	}

	//	apply transform in case meta has changed
	//	gr: this should multiply against itself? test against a rotated vidoe
	Transform(0,0) = OrigMeta.GetWidth() / static_cast<float>(Meta.GetWidth() );
	Transform(1,1) = OrigMeta.GetHeight() / static_cast<float>(Meta.GetHeight() );

}


void MfPixelBuffer::LockPixelsMediaBuffer2D(ArrayBridge<SoyPixelsImpl*>& Textures,Soy::AutoReleasePtr<IMFMediaBuffer>& MediaBuffer,float3x3& Transform)
{
	Soy::Assert( MediaBuffer!=nullptr, "Expected media buffer" );

	Soy::AutoReleasePtr<IMF2DBuffer> Buffer2D;
	auto Result = MediaBuffer->QueryInterface(&Buffer2D.mObject);
	MediaFoundation::IsOkay( Result, "Get IMF2DBuffer interface" );
	Soy::Assert( Buffer2D, "Successfull QueryInterface(IMF2DBuffer) but missing object");

	//	gr: docs suggest I need to retain this buffer2D, but seems I dont, but it definitely needs releasing or we leak
	static bool RetainBuffer2D = false;
	if ( RetainBuffer2D )
		Buffer2D->AddRef();

	bool IsContiguous = true;
	{
		BOOL RealIsContiguous = IsContiguous;
		Result = Buffer2D->IsContiguousFormat(&RealIsContiguous);
		MediaFoundation::IsOkay( Result, "Could not determine if buffer2d is contiguous");
		IsContiguous = bool_cast(RealIsContiguous);

		//	gr: if not contiguous then maybe we should throw and revert back to Lock() which is garunteed to be contiguous
		//		BUT may incur a copy cost (though so far not noticably slow)
		//	note: with YUV multiple planes which aren't aligning, (which is reported as non-continugous) using lock isn't fixing that (we have more bytes than we need)
		//	https://msdn.microsoft.com/en-us/library/windows/desktop/ms699894(v=vs.85).aspx
		//Soy::Assert( IsContiguous, "Cannot handle non-contiguous buffers at the moment");
	}

	//	lock
	BYTE* Bytes = nullptr;
	LONG Pitch = 0;
	DWORD ByteSize = 0;
	Result = MediaBuffer->GetCurrentLength(&ByteSize);
	MediaFoundation::IsOkay( Result, "IMF2DBuffer::GetCurrentLength" );
	Soy::Assert( ByteSize>0, "IMF2DBuffer::GetCurrentLength returned 0 bytes" );
	
	Result = Buffer2D->Lock2D( &Bytes, &Pitch );
	MediaFoundation::IsOkay( Result, "IMF2DBuffer::Lock2D" );
	Soy::Assert( Bytes!=nullptr, "IMF2DBuffer::Lock2D returned null" );
	mLockedBuffer2D = Buffer2D;
	mLockedMediaBuffer = MediaBuffer;

	auto Meta = mMeta.mPixelMeta;

	//	gr: cannot currently get MF to pull out pitch in meta, so we go with whatever is biggest
	//	pitch info should have been pulled out into the meta earlier
	//	this also means the transform won't be corrected if the pitch is upside down...
	bool Flipped = (Pitch < 0);
	Pitch = abs(Pitch);
	if ( Pitch == 0 )
	{
		Pitch = Meta.GetWidth();
	}
	else
	{
		SoyPixelsRemote Pixels( Bytes, ByteSize, Meta );
		Pitch = RescalePitch( Pitch, Pixels );
	}

	if ( Flipped )
	{
		std::Debug << "Todo: Apply flip transform" << std::endl;
	}

	ApplyPadding( Meta, Transform, Pitch, ByteSize );

	//	if the format has multiple planes we need to split the image
	SoyPixelsRemote Pixels( Bytes, ByteSize, Meta );
	Array<std::shared_ptr<SoyPixelsImpl>> PixelPlanes;
	Pixels.SplitPlanes( GetArrayBridge(PixelPlanes) );
	for ( int p=0;	p<PixelPlanes.GetSize();	p++ )
	{
		auto& Plane = PixelPlanes[p];
		
		//	gr: temporary hack until transform is applied in DX... clip the meta
		//		using transform matrix's height (as the chroma plane will probably be 1/2 height, and 1/2 padding)
		auto NewHeight = Plane->GetHeight() * Transform(1,1);
		Plane->GetMeta().DumbSetHeight( NewHeight );

		mLockedPixels.PushBack( Plane );
		Textures.PushBack( Plane.get() );
	}
}


void MfPixelBuffer::LockPixelsMediaBuffer(ArrayBridge<SoyPixelsImpl*>& Textures,Soy::AutoReleasePtr<IMFMediaBuffer>& MediaBuffer,float3x3& Transform)
{
	Soy::Assert( MediaBuffer!=nullptr, "Expected media buffer" );
	auto& Buffer = *MediaBuffer.mObject;

	//	lock
	BYTE* Bytes = nullptr;
	DWORD ByteSize = 0;
	
	//	note: lock is garunteed to be contiguous
	auto Result = Buffer.Lock( &Bytes, nullptr, &ByteSize );
	MediaFoundation::IsOkay( Result, "MediaBuffer::Lock" );

	mLockedMediaBuffer.Set( MediaBuffer.mObject, true );

	Soy::Assert( Bytes!=nullptr, "MediaBuffer::Lock returned null" );
	Soy::Assert( ByteSize>0, "MediaBuffer::GetCurrentLength returned 0 bytes" );

	//	pitch info should have been pulled out into the meta earlier (we cannot get it any other way)
	auto Meta = mMeta.mPixelMeta;

	//	gr: the pixel row size is in bytes, not pixels, so need to adjust
	auto Pitch = mMeta.mPixelRowSize;
	if ( Pitch == 0 )
	{
		//	2048x1048 was being padded
		//	had a case of a video that was 1148x646 & smeared and didn't bytes dont align to w/h/format
		if ( mApplyWidthPadding )
		{
			auto Overflow = Meta.GetWidth() % 16;
			if ( Overflow > 0 )
			{
				auto Pad = 16 - Overflow;
				Pitch = Meta.GetWidth() + Pad;
				std::Debug << "Applying width padding to " << Meta << ". Pitch now " << Pitch << std::endl;
			}
			else
			{
				Pitch = Meta.GetWidth();
			}
		}
		else
		{
			Pitch = Meta.GetWidth();
		}
	}
	else
	{
		SoyPixelsRemote Pixels( Bytes, ByteSize, Meta );
		Pitch = RescalePitch(  Pitch, Pixels );
	}

	ApplyPadding( Meta, Transform, Pitch, ByteSize );

	//	split-plane now does our under/overrun checks for us
	SoyPixelsRemote Pixels( Bytes, ByteSize, Meta );

	BufferArray<std::shared_ptr<SoyPixelsImpl>,4> PixelPlanes;
	Pixels.SplitPlanes( GetArrayBridge(PixelPlanes) );
	for ( int p=0;	p<PixelPlanes.GetSize();	p++ )
	{
		auto& Plane = PixelPlanes[p];
		
		//	gr: temporary hack until transform is applied in DX... clip the meta
		//		using transform matrix's height (as the chroma plane will probably be 1/2 height, and 1/2 padding)
		auto NewHeight = Plane->GetHeight() * Transform(1,1);
		Plane->GetMeta().DumbSetHeight( NewHeight );

		mLockedPixels.PushBack( Plane );
		Textures.PushBack( Plane.get() );
	}
}


void MfPixelBuffer::Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform)
{
	Soy::AutoReleasePtr<IMFMediaBuffer> MediaBuffer;
	GetMediaBuffer(MediaBuffer);

	//	in case there are problems don't apply the transform twice 
	float3x3 TransformBackup = Transform;

	//	for debug, force to not use the special Buffer2D type
	//	... which doesn't exist in win7
	bool SkipBuffer2D = mWin7Emulation;
		
	//	first try and use the [more efficient according to the docs] 2D buffer type
	try
	{
		if ( !SkipBuffer2D )
		{
			LockPixelsMediaBuffer2D( Textures, MediaBuffer, Transform );
			return;
		}
	}
	catch(std::exception& e)
	{
		//	just in case something has been locked and not reset in code, unlock
		Unlock();
		Transform = TransformBackup;
		std::Debug << "Could not use 2D buffer: " << e.what() << std::endl;
	}
	
	//	try raw buffer mode
	try
	{
		LockPixelsMediaBuffer( Textures, MediaBuffer, Transform );
		return;
	}
	catch(std::exception& e)
	{
		//	just in case something has been locked and not reset in code, unlock
		Unlock();
		Transform = TransformBackup;
		std::Debug << "Could not use raw buffer: " << e.what() << std::endl;
	}

}

void MfPixelBuffer::Unlock()
{
	if ( mLockedBuffer2D )
	{
		auto Result = mLockedBuffer2D->Unlock2D();
		try
		{
			MediaFoundation::IsOkay(Result, "Unlocking 2D buffer");
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
		mLockedBuffer2D.Release();
	}
	else if ( mLockedMediaBuffer )
	{
		auto Result = mLockedMediaBuffer->Unlock();
		try
		{
			MediaFoundation::IsOkay( Result, "Unlocking media buffer" );
		}
		catch(std::exception& e)
		{
			std::Debug << e.what() << std::endl;
		}
	}

	//	release locked objects
	mLockedMediaBuffer.Release();
	mLockedPixels.Clear();
}


void MfFileExtractor::AllocSourceReader(const std::string& Filename)
{
	bool IsUrlStream = Soy::StringBeginsWith(Filename, "http", false );

	//	gr: note, not using this as we can do YUV in shader
	//	https://msdn.microsoft.com/en-us/library/windows/desktop/dd374667(v=vs.85).aspx
	//	If you set the MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING attribute to TRUE when you create
	//	the Source Reader, the Source Reader will convert YUV video to RGB-32. This conversion is 
	//	not optimized for real-time video playback.
	//	gr: ^^ some cases where we want that as we're not handling odd row stride. ForceNonPlanarOutput is the equivelent backup on osx

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


	if ( IsUrlStream )
	{
		std::wstring Url = Soy::StringToWString( Filename );

		auto Result = MFCreateSourceReaderFromURL( Url.c_str(), Attributes, &mSourceReader.mObject );
		MediaFoundation::IsOkay(Result, "MFCreateSourceReaderFromURL");
	}
	else
	{
		auto CoInitialiseResult = CoInitialize(nullptr);
		mByteStream.reset(new MfByteStream( Filename ) );
		auto Result = MFCreateSourceReaderFromByteStream( mByteStream->mByteStream.mObject, Attributes, &mSourceReader.mObject );
		MediaFoundation::IsOkay(Result, "MFCreateSourceReaderFromByteStream");
	}

}

