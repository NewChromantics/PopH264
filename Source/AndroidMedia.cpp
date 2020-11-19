#include "AndroidMedia.h"
#include <sstream>
#include <thread>
//#include <SoyH264.h>
//#include <SoyWave.h>
#include <SoyOpengl.h>


//	android java defines, see if we can find a JNI header with these
#define KEY_WIDTH			"width"
#define KEY_HEIGHT			"height"
#define KEY_DURATION		"durationUs"
#define KEY_COLOR_FORMAT	"color-format"
#define KEY_FRAME_RATE		"frame-rate"
#define KEY_MIME			"mime"
#define KEY_PROFILE			"profile"
#define KEY_CHANNEL_COUNT	"channel-count"
#define KEY_SAMPLE_RATE		"sample-rate"
#define KEY_BITS_FORMAT_UNDOCUMENTED	"bits-format"

#define BUFFER_FLAG_CODEC_CONFIG	(0x00000002)
#define BUFFER_FLAG_END_OF_STREAM	(0x00000004)
#define BUFFER_FLAG_KEY_FRAME		(0x00000001)

namespace Java
{
	SoyMediaFormat::Type	MediaCodecColourFormatIndexToPixelFormat(int ColourFormat);
}


enum
{
	//	http://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities.html
	COLOR_Format12bitRGB444	= 3,
	COLOR_Format16bitARGB1555 = 5,
	COLOR_Format16bitARGB4444 = 4,
	COLOR_Format16bitBGR565 = 7,
	COLOR_Format16bitRGB565 = 6,
	COLOR_Format18BitBGR666 = 41,
	COLOR_Format18bitARGB1665 = 9,
	COLOR_Format18bitRGB666 = 8,
	COLOR_Format19bitARGB1666 = 10,
	COLOR_Format24BitABGR6666 = 43,
	COLOR_Format24BitARGB6666 = 42,
	COLOR_Format24bitARGB1887 = 13,
	COLOR_Format24bitBGR888 = 12,
	COLOR_Format24bitRGB888 = 11,
	COLOR_Format32bitABGR8888 = 0x7f00a000,
	COLOR_Format32bitARGB8888 = 16,
	COLOR_Format32bitBGRA8888 = 15,
	COLOR_Format8bitRGB332 = 2,
	COLOR_FormatCbYCrY = 27,
	COLOR_FormatCrYCbY = 28,
	COLOR_FormatL16 = 36,
	COLOR_FormatL2 = 33,
	COLOR_FormatL32 = 38,
	COLOR_FormatL4 = 34,
	COLOR_FormatL8 = 35,
	COLOR_FormatMonochrome = 1,
	COLOR_FormatRGBAFlexible = 0x7f36a888,
	COLOR_FormatRGBFlexible = 0x7f36b888,
	COLOR_FormatRawBayer10bit = 31,
	COLOR_FormatRawBayer8bit = 30,
	COLOR_FormatRawBayer8bitcompressed = 32,
	COLOR_FormatSurface = 0x7f000789,
	COLOR_FormatYCbYCr = 25,
	COLOR_FormatYCrYCb = 26,
	COLOR_FormatYUV411PackedPlanar = 18,
	COLOR_FormatYUV411Planar = 17,
	COLOR_FormatYUV420Flexible = 0x7f420888,
	COLOR_FormatYUV420PackedPlanar = 20,
	COLOR_FormatYUV420PackedSemiPlanar = 39,
	COLOR_FormatYUV420Planar = 19,
	COLOR_FormatYUV420SemiPlanar = 21,
	COLOR_FormatYUV422Flexible = 0x7f422888,
	COLOR_FormatYUV422PackedPlanar = 23,
	COLOR_FormatYUV422PackedSemiPlanar = 40,
	COLOR_FormatYUV422Planar = 22,
	COLOR_FormatYUV422SemiPlanar = 24,
	COLOR_FormatYUV444Flexible = 0x7f444888,
	COLOR_FormatYUV444Interleaved = 29,
	COLOR_QCOM_FormatYUV420SemiPlanar = 0x7fa30c00,
	COLOR_TI_FormatYUV420PackedSemiPlanar = 0x7f000100,
	
	//	gr: mystery format (when using surface texture)
	COLOR_FORMAT_UNKNOWN_MAYBE_SURFACE = 261,
	
	//	note4 is giving us a pixel format of this, even when we have a surface texture. Renders okay.
	//	gr; in non-opengl mode, the colour space is broken!
	//		it is not the same as Yuv_8_8_8 (but close). The name is a big hint of this.
	//		probably more like Yuv_8_88 but line by line.
	OMX_QCOM_COLOR_FormatYVU420SemiPlanarInterlace = 0x7FA30C04,	//	2141391876
};





void Android::IsOkay(MLResult Result,std::stringstream& Context)
{
Soy_AssertTodo();
/*
	if ( Result == MLResult_Ok )
		return;
	
	auto Str = Context.str();
	IsOkay( Result, Str.c_str() );
	*/
}

const char* Android::GetErrorString(MLResult Result)
{
Soy_AssertTodo();
/*
	//	specific media errors
	auto HasPrefix = [&](MLResult Prefix)
	{
		auto And = Result & Prefix;
		return And == Prefix;
	};

	if ( HasPrefix(MLResultAPIPrefix_MediaGeneric) ||
		HasPrefix(MLResultAPIPrefix_Media) ||
		HasPrefix(MLResultAPIPrefix_MediaDRM) ||
		HasPrefix(MLResultAPIPrefix_MediaOMX) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXExtensions) ||
		HasPrefix(MLResultAPIPrefix_MediaOMXVendors) ||
		HasPrefix(MLResultAPIPrefix_MediaPlayer) )
	{
		return MLMediaResultGetString( Result );
	}

	return MLGetResultString(Result);
	*/
}

void Android::IsOkay(MLResult Result,const char* Context)
{
	Soy_AssertTodo();
/*if ( Result == MLResult_Ok )
		return;

	auto ResultString = GetErrorString(Result);
	
	//	gr: sometimes we get unknown so, put error nmber in
	std::stringstream Error;
	Error << "Error in " << Context << ": " << ResultString;
	throw Soy::AssertException( Error );
	*/
}



//	note: this currently casts away const-ness (because of GetRemoteArray)
template<typename NEWTYPE,typename OLDTYPE>
inline FixedRemoteArray<NEWTYPE> CastArray(const ArrayBridge<OLDTYPE>&& Array)
{
	auto OldDataSize = Array.GetDataSize();
	auto OldElementSize = Array.GetElementSize();
	auto NewElementSize = sizeof(NEWTYPE);
	
	auto NewElementCount = OldDataSize / NewElementSize;
	auto* NewData = reinterpret_cast<const NEWTYPE*>( Array.GetArray() );
	return GetRemoteArray( NewData, NewElementCount );
}



void Android::GetMediaFileExtensions(ArrayBridge<std::string>&& Extensions)
{
	//	formats listed here, but pretty much an arbritry list
	//	http://developer.android.com/guide/appendix/media-formats.html

	const char* AndroidExtensions[] = {
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
	Soy::PushStringArray( Extensions, AndroidExtensions );
}




TSurfaceTexture::TSurfaceTexture(Opengl::TContext& Context,SoyPixelsMeta DesiredBufferMeta,Soy::TSemaphore* Semaphore,bool SingleBufferMode)
{
#if defined(ENABLE_OPENGL)
	auto AllocateTexture = [this,DesiredBufferMeta,SingleBufferMode]
	{
		Opengl::FlushError("TSurfaceTexture::AllocateTexture");
		
		//	alloc special texture
		glGenTextures( 1, &mTexture.mTexture.mName );
		mTexture.mType = GL_TEXTURE_EXTERNAL_OES;
		{
			std::stringstream Error;
			Error << "glGenTextures Surface texture GL_TEXTURE_EXTERNAL_OES allocation: " << mTexture.mTexture.mName;
			Opengl::IsOkay( Error.str() );
		}
		Soy::Assert( mTexture.IsValid(), std::string(__func__)+" allocated invalid texture" );
		
		//	gr: use proper funcs!
		glBindTexture(mTexture.mType, mTexture.mTexture.mName);
		glTexParameterf(mTexture.mType, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(mTexture.mType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(mTexture.mType, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(mTexture.mType, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(mTexture.mType, 0);
		Opengl::IsOkay("Surface texture GL_TEXTURE_EXTERNAL_OES init");
		
		//	this isn't the real dimensions... not sure how we get this. We never query the texture (can't on ES) and we just re-use the texture on the GPU so never deal with buffer
		//	just setting it to something so it's valid
		mTexture.mMeta = DesiredBufferMeta;
		if ( !mTexture.mMeta.IsValid() )
		{
			std::Debug << "DesiredBufferMeta of SurfaceTexture invalid; " << DesiredBufferMeta << std::endl;
			mTexture.mMeta = SoyPixelsMeta( 123, 456, SoyPixelsFormat::RGBA );
		}

		//	gr: does this need to be in a java thread?
		std::Debug << "Creating surface texture; SingleBufferMode=" << SingleBufferMode << "...." << std::endl;
		
		//	gr: single buffer mode causes big updateTexImage() pauses (locks?)
		//	create java surface texture
		mSurfaceTexture.reset( new JSurfaceTexture( mTexture, SingleBufferMode ) );

		//	initialise buffer size
		//	note; video & camera override this;
		//	http://developer.android.com/reference/android/graphics/SurfaceTexture.html#setDefaultBufferSize(int, int)
		//	The width and height parameters must be no greater than the minimum of GL_MAX_VIEWPORT_DIMS and GL_MAX_TEXTURE_SIZE (see glGetIntegerv). An error due to invalid dimensions might not be reported until updateTexImage() is called.
		if ( DesiredBufferMeta.IsValidDimensions() )
			mSurfaceTexture->SetBufferSize( DesiredBufferMeta.GetWidth(), DesiredBufferMeta.GetHeight() );

		//	create the surface
		mSurface.reset( new JSurface(*mSurfaceTexture) );
		
		return true;
	};
	
	if ( Semaphore )
		Context.PushJob( AllocateTexture, *Semaphore );
	else
		Context.PushJob( AllocateTexture );
#else
	throw Soy::AssertException("Opengl not supported");
#endif
}

TSurfaceTexture::~TSurfaceTexture()
{
	if ( mTexture )
	{
#if defined(ENABLE_OPENGL)
		mTexture->Delete();
#else
		throw Soy::AssertException("Opengl not supported");
#endif
	}
	mTexture.reset();
}
	
bool TSurfaceTexture::Update(SoyTime& Timestamp,bool& Changed)
{
	Changed = false;
	//	gr: hold onto ptr for thread safety
	auto pSurfaceTexture = mSurfaceTexture;
	if ( !pSurfaceTexture )
		return false;

	auto LastTimestampNs = pSurfaceTexture->GetTimestampNano();
	Soy::TScopeTimerPrint UpdateSurfaceTimer("TSurfaceTexture::Update updatetexture",5);
	//	gr: this can be quite slow (>9ms), but I can't see any way to get control over this. Delays I presume are down to the GPU & os :/
	pSurfaceTexture->UpdateTexture();
	UpdateSurfaceTimer.Stop();
	auto NewTimestampNs = pSurfaceTexture->GetTimestampNano();
	
	mCurrentContentsTimestamp.mTime = NewTimestampNs / 1000000;
	Timestamp = mCurrentContentsTimestamp;
	Changed = (LastTimestampNs != NewTimestampNs);

	return true;
}





void TSurfacePixelBuffer::Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform)
{
	ofScopeTimerWarning InitTimer("CopyToGlTexture init",4);
	if ( !mSurfaceTexture )
		return;

	if ( !mSurfaceTexture->IsValid() )
	{
		std::Debug << "surfacetexture not valid" << std::endl;
		return;
	}

#if defined(ENABLE_OPENGL)
	auto Texture = mSurfaceTexture->mTexture;
	if ( Texture && Texture->IsValid() )
		Textures.PushBack( *Texture );
#else
	throw Soy::AssertException("Opengl not supported");
#endif
}

void TSurfacePixelBuffer::Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform)
{
	std::Debug << "Android doesn't currently support slow pixel copy" << std::endl;
}

void TSurfacePixelBuffer::Unlock()
{
}


bool TSurfaceTexture::IsValid() const
{
	if ( !mSurfaceTexture )
	{
		std::Debug << "surface texture jtexture not valid " << std::endl;
		return false;
	}
	
#if defined(OPENGL_OPENGL)
	if ( !mTexture || !mTexture->IsValid() )
	{
		std::Debug << "surface texture texture not valid " << std::endl;
		return false;
	}
#else
	throw Soy::AssertException("Opengl not supported");
#endif
	
	return true;
}


SoyMediaFormat::Type Java::MediaCodecColourFormatIndexToPixelFormat(int ColourFormat)
{
	switch ( ColourFormat )
	{
		case COLOR_Format24bitRGB888:	return SoyMediaFormat::RGB;
		case COLOR_Format24bitBGR888:	return SoyMediaFormat::BGR;
		case COLOR_Format32bitBGRA8888:	return SoyMediaFormat::BGRA;
		case COLOR_Format32bitARGB8888:	return SoyMediaFormat::RGBA;
		case COLOR_FormatL8:			return SoyMediaFormat::Greyscale;
		case COLOR_FormatYUV420Planar:	return SoyMediaFormat::Yuv_8_8_8;
		case COLOR_FormatYUV420SemiPlanar:	return SoyMediaFormat::Yuv_8_88;

		case OMX_QCOM_COLOR_FormatYVU420SemiPlanarInterlace:	return SoyMediaFormat::Yuv_8_8_8;
			
		//	handle this to remove the debug
		case COLOR_FORMAT_UNKNOWN_MAYBE_SURFACE:	return SoyMediaFormat::Invalid;
			
		default:
			std::Debug << "Unhandled ColourFormat->PixelFormat conversion; " << ColourFormat << std::endl;
			return SoyMediaFormat::Invalid;
	}
}


TStreamMeta GetStreamFromMediaFormat(JniMediaFormat& Track,size_t StreamIndex)
{
	//	free some of the locals used in this func as they're quite self contained (we extract all the data we want)
	Java::TLocalRefStack LocalRefStack;
	
	TStreamMeta Meta;
	Meta.mStreamIndex = StreamIndex;
	
	std::Debug << "GetStreamFromMediaFormat(addr " << &Track << ") StreamIndex=" << StreamIndex << std::endl;
	
	auto AllMeta = Track.CallStringMethod("toString");
	//std::Debug << "Track toString=" << AllMeta << std::endl;

	try
	{
		auto Mime = Track.CallStringMethod("getString", KEY_MIME );
		//std::Debug << "Track Mime=" << Mime << std::endl;

		//	if mime is raw, we expect colour format to exist -
		//	though docs say the key is only for encoders... but we get it from output format of a decoder
		if ( Mime == "video/raw" )
		{
			auto ColourFormatIndex = Track.CallIntMethod("getInteger", KEY_COLOR_FORMAT );
			//std::Debug << "Colour format KEY_COLOR_FORMAT index=" << ColourFormatIndex << std::endl;
			Meta.mCodec = Java::MediaCodecColourFormatIndexToPixelFormat( ColourFormatIndex );
		}
		else
		{
			Meta.SetMime( Mime );
		}
 	}
	catch(...){}
	
	try
	{
		auto Profile = Track.CallStringMethod("getString", KEY_PROFILE );
		//std::Debug << "Track Profile=" << Profile << std::endl;
	}
	catch(...){}
	
	try
	{
		Meta.mPixelMeta.DumbSetHeight( Track.CallIntMethod("getInteger", KEY_HEIGHT) );
		Meta.mPixelMeta.DumbSetWidth( Track.CallIntMethod("getInteger", KEY_WIDTH) );
	}
	catch(...){}

	Meta.mPixelMeta.DumbSetFormat( SoyMediaFormat::GetPixelFormat(Meta.mCodec) );

	try
	{
		jlong DurationMicroSecs = Track.CallLongMethod("getLong", KEY_DURATION );
		Meta.mDuration.SetMicroSeconds( DurationMicroSecs );
	}
	catch(...){}
	
	try
	{
		Meta.mFramesPerSecond = Track.CallIntMethod("getInteger", KEY_FRAME_RATE );	//	docs say integer OR float... but OS(s6) just says "cannot cast integer to float"
	}
	catch(...){}

	try
	{
		Meta.mChannelCount = Track.CallIntMethod("getInteger", KEY_CHANNEL_COUNT );
	}
	catch(...){}
	
	try
	{
		Meta.mAudioSampleRate = Track.CallIntMethod("getInteger", KEY_SAMPLE_RATE );
	}
	catch(...){}
	
	try
	{
		Meta.mAudioBitsPerChannel = Track.CallIntMethod("getInteger", KEY_BITS_FORMAT_UNDOCUMENTED );
	}
	catch(...){}
	
	

	static bool DebugSPSPPS = false;
	if ( DebugSPSPPS )
	{
		Array<uint8> SPS;
		Array<uint8> PPS;
		
		//	grab codec data
		try
		{
			auto CsdByteBuffer = Track.CallObjectMethod("getByteBuffer","java.nio.ByteBuffer","csd-0");
			Java::BufferToArray( CsdByteBuffer, GetArrayBridge(SPS), "Get CSD-0" );
		}
		catch(std::exception& e)
		{
			//std::Debug << "Error getting format CSD-0 buffer; " << e.what() << std::endl;
		}
		
		try
		{
			auto CsdByteBuffer = Track.CallObjectMethod("getByteBuffer","java.nio.ByteBuffer","csd-1");
			Java::BufferToArray( CsdByteBuffer, GetArrayBridge(PPS), "Get CSD-1" );
		}
		catch(std::exception& e)
		{
			//std::Debug << "Error getting format CSD-1 buffer; " << e.what() << std::endl;
		}

		std::Debug << "csd-0 buffer x" << SPS.GetSize() << ": ";
		for ( int i=0;	i<SPS.GetSize();	i++ )
			std::Debug << (int)SPS[i] << " ";
		std::Debug << std::endl;
	
		std::Debug << "csd-1 buffer x" << PPS.GetSize() << ": ";
		for ( int i=0;	i<PPS.GetSize();	i++ )
			std::Debug << (int)PPS[i] << " ";
		std::Debug << std::endl;
	}
	
	std::stringstream Description;
	//Description << "Mime=" << Mime << " ";
	//Description << "Profile=" << Profile << " ";
	Description << "All=" << AllMeta << " ";
	Meta.mDescription = Description.str();
	
	return Meta;
}



/*
AndroidMediaExtractor::AndroidMediaExtractor(const TMediaExtractorParams& Params) :
	TMediaExtractor		( Params ),
	mDoneInitialAdvance	( false )
{
	auto Filename = Params.mFilename;
	std::Debug << "AndroidMediaExtractor(" << Filename << ")" << std::endl;
	mExtractor.reset( new JniMediaExtractor() );
	auto& Extractor = *mExtractor;
	
	if ( Soy::StringTrimLeft( Filename, TVideoDecoderParams::gProtocol_AndroidAssets, false ) )
	{
		Extractor.SetDataSourceAssets( Filename );
	}
	else if ( Soy::StringTrimLeft( Filename, TVideoDecoderParams::gProtocol_AndroidJar, false ) )
	{
		Extractor.SetDataSourceJar( Filename );
	}
	else if ( Soy::StringTrimLeft( Filename, TVideoDecoderParams::gProtocol_AndroidSdCard, false ) )
	{
		Extractor.SetDataSourceSdCard( Filename );
	}
	else
	{
		Extractor.SetDataSourcePath( Filename );
	}
	
	auto TrackFilter = [&](TStreamMeta& Meta)
	{
		if ( SoyMediaFormat::IsAudio( Meta.mCodec ) )
		{
			if ( !Params.mExtractAudioStreams )
				return false;
		}
		
		return true;
	};
	
	//	extract track metas and enable them in the filter
	auto TrackCount = Extractor.GetTrackCount();
	std::Debug << "Android extractor found " << TrackCount << " tracks" << std::endl;
	for ( int t=0;	t<TrackCount;	t++ )
	{
		try
		{
			auto TrackFormat = GetStreamFormat(t);
			auto Stream = GetStreamFromMediaFormat( TrackFormat->mFormat, t );
			
			if ( !TrackFilter( Stream ) )
			{
				std::Debug << __func__ << " skipping track " << Stream << std::endl;
				continue;
			}
		
			mStreams.PushBack( Stream );
			//	add to list of tracks to extract
			Extractor.CallVoidMethod("selectTrack", static_cast<int>(Stream.mStreamIndex) );
		}
		catch(std::exception& e)
		{
			std::Debug << "Failed to get track " << t << " from extractor: " << e.what() << std::endl;
		}
	}
	
	Soy::Assert( !mStreams.IsEmpty(), "Failed to extract any streams from movie" );

	
	
	//	allocate a buffer
	try
	{
		TJniClass ByteBufferClass("java.nio.ByteBuffer");
		int BufferMaxSize = 1024 * 1024 * 2;
		auto Buffer = ByteBufferClass.CallStaticObjectMethod("allocateDirect", ByteBufferClass.GetClassName(), BufferMaxSize );
		Soy::Assert( Buffer, "Failed to allocate buffer");
		
		//	copy the object we allocated
		mJavaBuffer.reset( new TJniObject(Buffer) );
	}
	catch (std::exception& e)
	{
		std::stringstream Error;
		Error << "Failed to allocate buffer; " << e.what();
		throw Soy::AssertException( Error.str() );
	}
	Soy::Assert( mJavaBuffer!=nullptr, "JavaBuffer expected");
	
	//	start the extractor thread automatically so we get first frames asap
	//	gr: maybe not?
	Start();
}

AndroidMediaExtractor::~AndroidMediaExtractor()
{
	std::Debug << __func__ << std::endl;
	WaitToFinish();
	
	if ( mExtractor )
	{
		std::Debug << __func__ << " release extractor" << std::endl;
		mExtractor->CallVoidMethod("release");
		mExtractor.reset();
	}
}

std::shared_ptr<Platform::TMediaFormat> AndroidMediaExtractor::GetStreamFormat(size_t StreamIndex)
{
	Soy::Assert( mExtractor!=nullptr, "Extractor expected" );
	
	JniMediaFormat Format = mExtractor->GetTrack( StreamIndex );
	return std::shared_ptr<Platform::TMediaFormat>( new Platform::TMediaFormat( Format ) );
}

void AndroidMediaExtractor::GetStreams(ArrayBridge<TStreamMeta>&& Streams)
{
	Streams.PushBackArray( mStreams );
}

std::shared_ptr<TMediaPacket> AndroidMediaExtractor::ReadNextPacket()
{
	//	thread safe ptr copies
	auto pExtractor = mExtractor;
	auto pBuffer = mJavaBuffer;
	Soy::Assert( pBuffer!=nullptr, "Expected java buffer" );
	Soy::Assert( pExtractor!=nullptr, "Expected extractor" );
	auto& Buffer = *pBuffer;
	auto& Extractor = *pExtractor;
	
	/ *	gr: skips first keyframe!
	//	soemtimes this throws (odd data? shark video!) an illegalArgumentException.
	//	http://stackoverflow.com/questions/33148629/android-mediaextractor-readsampledata-illegalargumentexception
	//	this post suggests we need to call advance at least once at the start
	if ( !mDoneInitialAdvance )
	{
		auto DidAdvanced = Extractor.CallBoolMethod("advance");
		if ( !DidAdvanced )
		{
			std::Debug << __func__ << " Failed to do initial advance(). Fatal?" << std::endl;
		}
		mDoneInitialAdvance = true;
	}
	 * /

	
	int Offset = 0;
	//std::Debug << __func__ << " readSampleData" << std::endl;
	auto BufferSize = 0;
	
	//	this video plays many frames, then I get an illegal argument error
	//	http://downloads.4ksamples.com/downloads/4K-Chimei-inn-60mbps%20(4ksamples)%20.mp4
	//	this stackoverflow post suggests the buffer is OOM. Increased buffer from 1mb to 2mb and problem has gone
	//	http://stackoverflow.com/questions/33148629/android-mediaextractor-readsampledata-illegalargumentexception/35160271#35160271
	//	consider replacing this with an explicit, larger reallocation
	try
	{
		BufferSize = Extractor.CallIntMethod( "readSampleData", Buffer, Offset );
	}
	catch(std::exception& e)
	{
		std::Debug << "readSampleData exception; " << e.what() << ". Possibly buffer (" << Soy::FormatSizeBytes(Java::GetBufferSize(Buffer)) << ") is too small? Skipping frame." << std::endl;
		if ( !Extractor.CallBoolMethod("advance") )
		{
			std::Debug << "Failed to advance, fatal? EOF?" << std::endl;
		}
		return nullptr;
	}
	
	//	eof
	if ( BufferSize < 0)
	{
		//std::Debug << "Got sample size " << BufferSize << " EOF" << std::endl;
		return nullptr;
	}
	

	//	get meta for current frame
	//std::Debug << __func__ << " getmeta" << std::endl;
#define SAMPLE_FLAG_ENCRYPTED	0x2
#define SAMPLE_FLAG_SYNC		0x1
	auto SampleStream = Extractor.CallIntMethod("getSampleTrackIndex");
	auto SamplePresentationTimeMicrosecs = Extractor.CallLongMethod("getSampleTime");
	auto SampleFlags = Extractor.CallIntMethod("getSampleFlags");
	bool IsKeyframe = bool_cast(SampleFlags & SAMPLE_FLAG_SYNC);
	

	//std::Debug << "Extracted sample; " << BufferSize << " bytes, flags=" << SampleFlags <<  std::endl;
	
	//	skip packets in the past
	//	todo: don't skip the keyframe before TIME as it's required for infra frames
	SoyTime PresentationTimecode;
	PresentationTimecode.SetMicroSeconds( SamplePresentationTimeMicrosecs );
	if ( !CanPushPacket( PresentationTimecode, SampleStream, IsKeyframe) )
	{
		//	todo: make sure thread doesn't sleep
		std::Debug << "Skipping extracted frame at " << PresentationTimecode << std::endl;
		auto DidAdvanced = Extractor.CallBoolMethod("advance");
		Soy::Assert( DidAdvanced, "Failed to advance() to next frame. Fatal?");
		return nullptr;
	}

	
	
	std::shared_ptr<TMediaPacket> pPacket( new TMediaPacket );
	auto& Packet = *pPacket;
	
	Packet.mTimecode = PresentationTimecode;
	Packet.mEncrypted = SampleFlags & SAMPLE_FLAG_ENCRYPTED;
	Packet.mIsKeyFrame = IsKeyframe;
	//std::Debug << __func__ << " Java::BufferToArray" << std::endl;
	Java::BufferToArray( Buffer, GetArrayBridge(Packet.mData), "Extractor read packet", BufferSize );
	
	//std::Debug << __func__ << " GetStreamFormat" << std::endl;
	auto Format = GetStreamFormat( SampleStream );
	//std::Debug << __func__ << " GetStreamFromMediaFormat stream #" << SampleStream << std::endl;
	Packet.mMeta = GetStreamFromMediaFormat( Format->mFormat, SampleStream );

	//	determine real h264 format
	if ( SoyMediaFormat::IsH264(Packet.mMeta.mCodec) )
	{
		H264::ResolveH264Format( Packet.mMeta.mCodec, GetArrayBridge(Packet.mData) );
	}
	
	//std::Debug << __func__ << " advance" << std::endl;
	auto DidAdvanced = Extractor.CallBoolMethod("advance");
	Soy::Assert( DidAdvanced, "Failed to advance() to next frame. Fatal?");
	
	OnPacketExtracted(pPacket);
	return pPacket;
}
*/
/*
AndroidEncoderBuffer::AndroidEncoderBuffer(int OutputBufferIndex,const std::shared_ptr<TJniObject>& Codec,const std::shared_ptr<TSurfaceTexture>& Surface) :
	mOutputBufferIndex	( OutputBufferIndex ),
	mCodec				( Codec ),
	mSurfaceTexture		( Surface )
{
}

AndroidEncoderBuffer::AndroidEncoderBuffer(int OutputBufferIndex,const std::shared_ptr<TJniObject>& Codec,const std::shared_ptr<SoyPixelsImpl>& ByteBufferPixels) :
	mOutputBufferIndex	( OutputBufferIndex ),
	mCodec				( Codec ),
	mByteBufferPixels	( ByteBufferPixels )
{
}


AndroidEncoderBuffer::~AndroidEncoderBuffer()
{
	//	release buffer if still acquired
	try
	{
		ReleaseBuffer(false);
	}
	catch(std::exception& e)
	{
		std::Debug << __func__ << " exception; " << e.what() << std::endl;
	}
}

void AndroidEncoderBuffer::ReleaseBuffer(bool Render)
{
	if ( mOutputBufferIndex != -1 && mCodec )
	{
		ofScopeTimerWarning Timer("releaseOutputBuffer",2);
		mCodec->CallVoidMethod("releaseOutputBuffer", mOutputBufferIndex, Render );
		mOutputBufferIndex = -1;
	}
}

void AndroidEncoderBuffer::Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform)
{
	//	no opengl backing, assume we're reading buffer directly instead
	if ( !mSurfaceTexture )
		return;

	//	deffered creation of surface's texture hasn't occurred yet... but it should have by now?
	Soy::Assert( mSurfaceTexture->mTexture.IsValid(false), "Surface texture isn't valid yet" );

	//std::Debug << __func__ << " baking surface texture " << mSurfaceTexture->mTexture.mTexture.mName << " x " << mSurfaceTexture->mTexture.GetMeta() << std::endl;
	
	//	update the surface
	//	http://developer.android.com/reference/android/media/MediaCodec.html#dequeueOutputBuffer(android.media.MediaCodec.BufferInfo, long)
	//	Do not render the buffer: Call releaseOutputBuffer(bufferId, false).
	//	Render the buffer with the default timestamp: Call releaseOutputBuffer(bufferId, true).
	//	Render the buffer with a specific timestamp: Call releaseOutputBuffer(bufferId, timestamp).
	//	gr: consider using the timestamp version to specify when in the OS (nearest to nanoTime()) to present the buffer
	//		this would be good as it would mean our side we relesae the buffer asap
	//		but removes our entire buffering system
	ReleaseBuffer( true );
	
	//	copy from surface to texture
	//	gr: use mSurfaceTexture->Update( Frame.mTimestamp, Changed ?
	SoyTime SurfaceTextureTimestamp;
	bool SurfaceTextureChanged = false;
	mSurfaceTexture->Update( SurfaceTextureTimestamp, SurfaceTextureChanged );

	//	gr: this is reporting 0 and unchanged... presumably that's because the surfacetexture has this set externally which we're not doing, we just watch updateTexImage to be called...
//	std::Debug << "Surface texture update; " << SurfaceTextureTimestamp << " changed=" << SurfaceTextureChanged << std::endl;
	
	//	surface is now updated, return it
	Textures.PushBack( mSurfaceTexture->mTexture );
}

void AndroidEncoderBuffer::Lock(ArrayBridge<SoyPixelsImpl*>&& Textures,float3x3& Transform)
{
	Textures.PushBack( mByteBufferPixels.get() );
}

void AndroidEncoderBuffer::Unlock()
{
	//	don't release buffer here! if texture fails, and we go to pixels, this will release early. Fine to leave it in the destructor
}
*/


/*

AndroidMediaDecoder::AndroidMediaDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer>& InputBuffer,std::shared_ptr<TPixelBufferManager>& OutputBuffer,std::shared_ptr<Platform::TMediaFormat>& Format,const TVideoDecoderParams& Params,std::shared_ptr<Opengl::TContext> OpenglContext) :
	TMediaDecoder		( ThreadName, InputBuffer, OutputBuffer ),
	mCodecStarted		( false )
{
	//	see main thread/same thread comments
	//	http://stackoverflow.com/questions/32772854/android-ndk-crash-in-androidmediacodec#
	//	after getting this working, appears we can start on one thread and iterate on another.
	//	keep this code here in case other phones/media players can't handle it...
	//	gr: we use the deffered alloc so we can wait for the surface to be created
	//		maybe we can mix creating stuff before the configure...
	static bool DefferedAlloc = true;
	
	SoyPixelsMeta SurfaceMeta;
	if ( Params.mDecoderUseHardwareBuffer )
	{
		SurfaceMeta = Params.mAssumedTargetTextureMeta;

		if ( !SurfaceMeta.IsValidDimensions() )
		{
			std::stringstream Error;
			Error << "Creating surface texture but params from VideoDecoderParams are invalid" << SurfaceMeta << std::endl;
			throw Soy::AssertException( Error.str() );
		}
	}
	
	if ( DefferedAlloc )
	{
		//	gr: create all this on the same thread as the buffer queue
		auto InvokeAlloc = [=](bool&)
		{
			try
			{
				Alloc( SurfaceMeta, Format, OpenglContext, Params.mAndroidSingleBufferMode );
			}
			catch(std::exception& e)
			{
				//	gr: get extended info here. Sometimes fails (and something then abort()'s the app) with
				//	I/Pop     (12487): pop: Failed to allocate encoder Java exception in void TJniObject::CallVoidMethod(const std::string &, TJniObject &, TJniObject &, TJniObject &, int)configure: android.media.MediaCodec$CodecException: Error 0xffffffea
				//	F/libc    (12487): Fatal signal 6 (SIGABRT), code -6 in tid 13250 (Thread-443)
				//	https://developer.android.com/reference/android/media/MediaCodec.CodecException.html
				std::Debug << "Failed to allocate encoder " << e.what() << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				WaitToFinish();
				std::Debug << "mCodec.reset()... " << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				mCodec.reset();
				std::Debug << "mCodec.reset()... finished "<< std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
			}
		};
		this->mOnStart.AddListener( InvokeAlloc );
	}
	else
	{
		Alloc( SurfaceMeta, Format, OpenglContext, Params.mAndroidSingleBufferMode );
	}

	//	start thread
	Start();
}


AndroidMediaDecoder::AndroidMediaDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer>& InputBuffer,std::shared_ptr<TAudioBufferManager>& OutputBuffer,std::shared_ptr<Platform::TMediaFormat>& Format,const TVideoDecoderParams& Params) :
	TMediaDecoder		( ThreadName, InputBuffer, OutputBuffer ),
	mCodecStarted		( false )
{
	Soy::Assert( Format!=nullptr, "Audio media decoder requires format");
	
	//	see main thread/same thread comments
	//	http://stackoverflow.com/questions/32772854/android-ndk-crash-in-androidmediacodec#
	//	after getting this working, appears we can start on one thread and iterate on another.
	//	keep this code here in case other phones/media players can't handle it...
	//	gr: we use the deffered alloc so we can wait for the surface to be created
	//		maybe we can mix creating stuff before the configure...
	static bool DefferedAlloc = true;
	
	if ( DefferedAlloc )
	{
		//	gr: create all this on the same thread as the buffer queue
		auto InvokeAlloc = [=](bool&)
		{
			try
			{
				AllocAudio( *Format );
			}
			catch(std::exception& e)
			{
				//	gr: get extended info here. Sometimes fails (and something then abort()'s the app) with
				//	I/Pop     (12487): pop: Failed to allocate encoder Java exception in void TJniObject::CallVoidMethod(const std::string &, TJniObject &, TJniObject &, TJniObject &, int)configure: android.media.MediaCodec$CodecException: Error 0xffffffea
				//	F/libc    (12487): Fatal signal 6 (SIGABRT), code -6 in tid 13250 (Thread-443)
				//	https://developer.android.com/reference/android/media/MediaCodec.CodecException.html
				std::Debug << "Failed to allocate encoder " << e.what() << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				WaitToFinish();
				std::Debug << "mCodec.reset()... " << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
				mCodec.reset();
				std::Debug << "mCodec.reset()... finished "<< std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(200) );
			}
		};
		this->mOnStart.AddListener( InvokeAlloc );
	}
	else
	{
		AllocAudio( *Format );
	}
	
	//	start thread
	Start();
}


void AndroidMediaDecoder::Alloc(SoyPixelsMeta SurfaceMeta,std::shared_ptr<Platform::TMediaFormat> Format,std::shared_ptr<Opengl::TContext> OpenglContext,bool SingleBufferMode)
{
	//	generate format if not provided
	//	gr: on some phones, our generated format is not producing output frames (dequeueoutputbuffer never gives us a result)
	if ( !Format )
	{
		Soy::Assert( Format!=nullptr, "gr: this needs re-writing..." );
		/*
		JniMediaFormat OldFormatj;
		//auto Stream = GetStreamFromMediaFormat( OldFormatj, 999 );

		//	create format from scratch
		TJniClass MediaFormatClass("android.media.MediaFormat");
		TJniObject Formatj = MediaFormatClass.CallStaticObjectMethod( "createVideoFormat", MediaFormatClass.GetClassName(), Stream.GetMime(), FormatjMeta.mPixelMeta.GetWidth(), FormatjMeta.mPixelMeta.GetHeight() );
		
		long DurationMicroSecs = Stream.mDuration.GetMicroSeconds();
		Formatj.CallVoidMethod("setLong", KEY_DURATION, DurationMicroSecs );
		
		auto SPS = OldFormatj.CallObjectMethod("getByteBuffer","java.nio.ByteBuffer","csd-0");
		auto PPS = OldFormatj.CallObjectMethod("getByteBuffer","java.nio.ByteBuffer","csd-1");
		
		Formatj.CallVoidMethod("setByteBuffer", "csd-0", SPS );
		Formatj.CallVoidMethod("setByteBuffer", "csd-1", PPS );
		//byte[] header_sps = { 0, 0, 0, 1, 103, 100, 0, 40, -84, 52, -59, 1, -32, 17, 31, 120, 11, 80, 16, 16, 31, 0, 0, 3, 3, -23, 0, 0, -22, 96, -108 };
		//byte[] header_pps = { 0, 0, 0, 1, 104, -18, 60, -128 };
		//format.setByteBuffer("csd-0", ByteBuffer.wrap(header_sps));
		//format.setByteBuffer("csd-1", ByteBuffer.wrap(header_pps));
		//format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1920 * 1080);
		//format.setInteger("durationUs", 63446722);
		
		//	set colour format
#define COLOR_FormatYUV420Planar	19
		Formatj.CallVoidMethod("setInteger", KEY_COLOR_FORMAT, COLOR_FormatYUV420Planar );
		//	=6291456, durationUs=30000000,
		
		//	http://stackoverflow.com/questions/15105843/mediacodec-jelly-bean#_=_
		static bool ApplyMaxInputSize = true;
		if ( ApplyMaxInputSize )
		{
#define KEY_MAX_INPUT_SIZE	"max-input-size"
			int MaxInputSize = 0;
			Formatj.CallVoidMethod("setInteger", KEY_MAX_INPUT_SIZE, MaxInputSize );
		}
		
		Format.reset( new Platform::TMediaFormat( JniMediaFormat( Formatj ) ) );
		std::Debug << "Decoder format meta " << GetStreamFromMediaFormat( DecoderFormat,888) << std::endl;
		 * /
	}
					 
	Alloc( SurfaceMeta, *Format, OpenglContext, SingleBufferMode );
}

void AndroidMediaDecoder::Alloc(SoyPixelsMeta SurfaceMeta,Platform::TMediaFormat& Format,std::shared_ptr<Opengl::TContext> OpenglContext,bool SingleBufferMode)
{
	//	gr: attempt to create this from the stream meta to remove the android dependency
	//		note: when we do this, don't forget to set SPS and PPS http://stackoverflow.com/questions/19742047/how-to-use-mediacodec-without-mediaextractor-for-h264

	std::Debug << __func__ << std::endl;
	
	//	create decoder
	std::Debug << "Alloc Copy format... format=" << &Format << std::endl;
	std::this_thread::sleep_for( std::chrono::milliseconds(100) );
	std::Debug << "Alloc Copy format... format.mFormat" << &Format.mFormat << std::endl;
	std::this_thread::sleep_for( std::chrono::milliseconds(100) );
	auto Formatj = Format.mFormat;

	std::Debug << "Alloc GetStreamFromMediaFormat" << std::endl;
	auto Stream = GetStreamFromMediaFormat( Formatj, 999 );
	
	std::Debug << "Creating decoder with mime " << Stream.GetMime() << std::endl;
	TJniClass MediaCodecClass("android/media/MediaCodec");
	auto MediaCodec = MediaCodecClass.CallStaticObjectMethod("createDecoderByType","android/media/MediaCodec", Stream.GetMime() );
	Soy::Assert( MediaCodec, "Failed to create decoder for track" );

	//	store codec once we've reached a point when it needs to be released
	mCodec.reset( new TJniObject( MediaCodec ) );


	
	//	create a surface if params specified
	//	gr: now we create it here, we can block instead of this hacky sleep.
	if ( SurfaceMeta.IsValidDimensions() && OpenglContext )
	{
		std::Debug << "Creating surface texture " << SurfaceMeta << " (and blocking...)" << std::endl;
		Soy::TSemaphore Semaphore;
		mSurfaceTexture.reset( new TSurfaceTexture( *OpenglContext, SurfaceMeta, &Semaphore, SingleBufferMode ) );
		Semaphore.Wait();
	}
	else
	{
		std::Debug << "X Skipped creating surface texture: Context=" << (OpenglContext?"not-null":"null") << " SurfaceMeta=" << SurfaceMeta << std::endl;
	}
	
	std::shared_ptr<JSurface> pSurface = mSurfaceTexture ? mSurfaceTexture->mSurface : nullptr;
	if ( !pSurface )
		std::Debug << "Configuring media codec without surface" << std::endl;
	else
		std::Debug << "Configuring media codec WITH surface: " << mSurfaceTexture->GetTexture().GetMeta() << std::endl;
	auto Surfacej = pSurface ? *pSurface : TJniObject::Null("android.view.Surface");

	
	auto MediaCryptoj = TJniObject::Null("android.media.MediaCrypto");
	int Flags = 0x0;	//	CONFIGURE_FLAG_ENCODE
	
	//	format cannot be null
	//	see https://dxr.mozilla.org/mozilla-central/source/dom/media/platforms/android/AndroidDecoderModule.cpp
	//	for a "working" implementation. Maybe buffers need resetting, maybe deque input & output needs to be on same thread?
	MediaCodec.CallVoidMethod("configure", Formatj, Surfacej, MediaCryptoj, Flags );

	//auto OutputFormat = MediaCodec.CallObjectMethod("getOutputFormat", "android.media.MediaFormat");

	std::Debug << "MediaCodec.Start()" << std::endl;
	MediaCodec.CallVoidMethod("start");
	mCodecStarted = true;
	
	std::Debug << __func__ << " finished" << std::endl;
}


void AndroidMediaDecoder::AllocAudio(Platform::TMediaFormat& Format)
{
	std::Debug << __func__ << std::endl;
	
	//	create decoder
	std::Debug << "AllocAudio Copy format... format=" << &Format << std::endl;
	std::this_thread::sleep_for( std::chrono::milliseconds(100) );
	std::Debug << "AllocAudio Copy format... format.mFormat" << &Format.mFormat << std::endl;
	std::this_thread::sleep_for( std::chrono::milliseconds(100) );

	auto Formatj = Format.mFormat;
	auto Stream = GetStreamFromMediaFormat( Formatj, 999 );
	std::Debug << "Creating audio decoder with stream " << Stream << std::endl;
	TJniClass MediaCodecClass("android/media/MediaCodec");
	auto MediaCodec = MediaCodecClass.CallStaticObjectMethod("createDecoderByType","android/media/MediaCodec", Stream.GetMime() );
	Soy::Assert( MediaCodec, "Failed to create decoder for track" );
	
	//	store codec once we've reached a point when it needs to be released
	mCodec.reset( new TJniObject( MediaCodec ) );
	
	auto Surfacej =  TJniObject::Null("android.view.Surface");
	auto MediaCryptoj = TJniObject::Null("android.media.MediaCrypto");
	int Flags = 0x0;	//	CONFIGURE_FLAG_ENCODE
	
	//	see https://dxr.mozilla.org/mozilla-central/source/dom/media/platforms/android/AndroidDecoderModule.cpp
	//	for a "working" implementation. Maybe buffers need resetting, maybe deque input & output needs to be on same thread?
	MediaCodec.CallVoidMethod("configure", Formatj, Surfacej, MediaCryptoj, Flags );
	
	//auto OutputFormat = MediaCodec.CallObjectMethod("getOutputFormat", "android.media.MediaFormat");
	
	std::Debug << "MediaCodec.Start()" << std::endl;
	MediaCodec.CallVoidMethod("start");
	mCodecStarted = true;
	
	std::Debug << __func__ << " finished" << std::endl;
}


AndroidMediaDecoder::~AndroidMediaDecoder()
{
	std::Debug << __func__ << std::endl;

	//	end the thread
	WaitToFinish();
	
	if ( mCodec )
	{
		std::Debug << __func__ << " stop codec"  << std::endl;
		mCodec->CallVoidMethod("stop");
		mCodecStarted = false;
		std::Debug << __func__ << " release codec"  << std::endl;
		mCodec->CallVoidMethod("release");
		std::Debug << __func__ << " delete codec"  << std::endl;
		mCodec.reset();
	}
}

void AndroidMediaDecoder::ProcessOutputPacket(std::function<bool(std::shared_ptr<TJniObject>,size_t,int,const TStreamMeta&,SoyTime)> HandleBufferFunc,TMediaBufferManager& Output)
{
	Soy::Assert( mCodec!=nullptr, "Codec expected" );
	
	bool Block = true;
	//long TimeoutUs = Block ? -1 : 0;
	long TimeoutUs = 1000;
	//std::Debug << "dequeueOutputBuffer with timeout=" << TimeoutUs << std::endl;
	
	//	look for pending output buffer
	//	instance a meta object that gets updated
	TJniObject BufferInfo("android.media.MediaCodec$BufferInfo");
	
	int outputBufferId;
	try
	{
		outputBufferId = mCodec->CallIntMethod( "dequeueOutputBuffer", BufferInfo, TimeoutUs );
	}
	catch (std::exception& e)
	{
		std::Debug << "dequeueOutputBuffer caused exception; " << e.what() << ", resetting codec..."  << std::endl;
		//	mCodec->CallVoidMethod("reset");
		//	mCodec->CallVoidMethod("start");
		return;
	}
	
	if (outputBufferId < 0 )
	{
		int INFO_OUTPUT_BUFFERS_CHANGED = -3;
		int INFO_OUTPUT_FORMAT_CHANGED = -2;
		int INFO_TRY_AGAIN_LATER = -1;
		
		if ( outputBufferId == INFO_OUTPUT_BUFFERS_CHANGED )
		{
			std::Debug << "ProcessOutputPacket INFO_OUTPUT_BUFFERS_CHANGED" << std::endl;
		}
		else if ( outputBufferId == INFO_OUTPUT_FORMAT_CHANGED )
		{
			std::Debug << "ProcessOutputPacket INFO_OUTPUT_FORMAT_CHANGED" << std::endl;
		}
		else if ( outputBufferId == INFO_TRY_AGAIN_LATER )
		{
			std::Debug << "ProcessOutputPacket INFO_TRY_AGAIN_LATER" << std::endl;
		}
		else
		{
			std::stringstream Error;
			Error << "unkown output buffer id error " << outputBufferId;
			throw Soy::AssertException( Error.str() );
		}
		Java::IsOkay("No output buffer");
		return;
	}
	std::Debug << "Got output buffer=" << outputBufferId << std::endl;
	
	
	//	if we error, release the buffer or we'll block the decoder
	try
	{
		//	this func should be nice and fast, unless we hit the slow path
		Soy::TScopeTimerPrint Timer("Handle decompressed frame", 2);
		
		//	get meta
		auto presentationTimeUs = BufferInfo.GetLongField("presentationTimeUs");
		SoyTime PresentationTime;
		PresentationTime.SetMicroSeconds( presentationTimeUs );
		Output.mOnFrameDecoded.OnTriggered( PresentationTime );
		
		auto Flags = BufferInfo.GetIntField("flags");
		auto offset = BufferInfo.GetIntField("offset");
		auto size = BufferInfo.GetIntField("size");
		auto MediaFormat = JniMediaFormat( mCodec->CallObjectMethod("getOutputFormat","android.media.MediaFormat",outputBufferId) );
		
		auto OutputMeta = GetStreamFromMediaFormat( MediaFormat, 999 );
		
		
		
		//	before we do any work, see if we want to skip this frame
		if ( !Output.PrePushBuffer( PresentationTime ) )
		{
			std::Debug << "Skipped frame " << PresentationTime << " with PrePushPixelBuffer" << std::endl;
			mCodec->CallVoidMethod("releaseOutputBuffer", outputBufferId, false );
			return;
		}
		
		/*
		 std::Debug << "output buffer: ";
		 std::Debug << " timeUs=" << presentationTimeUs;
		 std::Debug << " Flags=" << Flags;
		 std::Debug << " size=" << size;
		 std::Debug << " offset=" << offset;
		 std::Debug << " timestamp=" << Frame.mTimestamp;
		 std::Debug << " outputformat=" << OutputMeta;
		 std::Debug << std::endl;
		 * /
		
		//	work out which path to use
		std::shared_ptr<TJniObject> pByteBuffer;
		
		//	grab byte buffer
		try
		{
			auto ByteBuffer = mCodec->CallObjectMethod("getOutputBuffer", "java.nio.ByteBuffer", outputBufferId );
			pByteBuffer.reset( new TJniObject( ByteBuffer ) );
		}
		catch(std::exception& e)
		{
		}
		
		//	if this returns true, release the buffer
		if ( HandleBufferFunc( pByteBuffer, size, outputBufferId, OutputMeta, PresentationTime ) )
		{
			//std::Debug << "Release output buffer..." << std::endl;
			mCodec->CallVoidMethod("releaseOutputBuffer", outputBufferId, false );
		}
	}
	catch ( std::exception& e)
	{
		std::Debug << __func__ << " " << e.what() << std::endl;
		mCodec->CallVoidMethod("releaseOutputBuffer", outputBufferId, false );
		throw;
	}
	catch (...)
	{
		mCodec->CallVoidMethod("releaseOutputBuffer", outputBufferId, false );
		throw;
	}
	
}


void AndroidMediaDecoder::ProcessOutputPacket(TPixelBufferManager& Output)
{
	auto HandleBuffer = [&Output,this](std::shared_ptr<TJniObject> pByteBuffer,size_t ByteBufferSize,int OutputBufferId,const TStreamMeta& OutputMeta,SoyTime Timestamp)
	{
		std::shared_ptr<SoyPixelsImpl> pByteBufferPixels;
		TPixelBufferFrame Frame;
		//	make psuedo pixel buffer which references this packet
		Frame.mTimestamp = Timestamp;

		//	try and make direct pixels reference to the byte buffer
		if ( !mSurfaceTexture && OutputMeta.mPixelMeta.IsValid() )
		{
			try
			{
				//	if this doens't throw then we can use it
				auto BufferArray = Java::GetBufferArray( *pByteBuffer, ByteBufferSize );
				pByteBufferPixels.reset( new SoyPixelsRemote( GetArrayBridge(BufferArray), OutputMeta.mPixelMeta ) );
			}
			catch(std::exception& e)
			{
			}
		}
		
		bool ReleaseBuffer = false;
		
		if ( mSurfaceTexture )
		{
			//	special buffer with texture
			Soy::TScopeTimerPrint Timer( "Making surface backed pixel buffer...", 0 );
			//std::Debug << "Making surface backed pixel buffer..." << std::endl;
			Frame.mPixels.reset( new AndroidEncoderBuffer( OutputBufferId, mCodec, mSurfaceTexture ) );
			ReleaseBuffer = false;
		}
		else if ( pByteBufferPixels )
		{
			//	special buffer with access to byte buffer
			std::Debug << "Making byte-buffer backed pixel buffer..." << std::endl;
			Frame.mPixels.reset( new AndroidEncoderBuffer( OutputBufferId, mCodec, pByteBufferPixels ) );
			ReleaseBuffer = false;
		}
		else
		{
			std::Debug << "Making buffer copy(slow!) dumb pixel buffer..." << std::endl;
			//	if using pixel buffer directly, make a dumb pixel buffer and release the buffer immediately
			if ( !OutputMeta.mPixelMeta.IsValid() )
			{
				std::stringstream Error;
				Error << "Using dumb pixel buffer for decoder output, but invalid meta; " << OutputMeta.mPixelMeta;
				throw Soy::AssertException( Error.str() );
			}
			
			//std::Debug << "Allocating dumb pixel buffer..." << std::endl;
			std::shared_ptr<TDumbPixelBuffer> pPixelBuffer( new TDumbPixelBuffer(OutputMeta.mPixelMeta) );
			auto& PixelBuffer = *pPixelBuffer;
			auto& PixelBufferArray = PixelBuffer.mPixels.GetPixelsArray();
			
			//std::Debug << "Copying buffer to pixel buffer..." << std::endl;
			auto ByteBuffer = mCodec->CallObjectMethod("getOutputBuffer", "java.nio.ByteBuffer", OutputBufferId );
			
			//	copy pixels
			//	gr: this is very slow
			//	Encoder CopyFromOutputBuffer BufferSize=6291456/6291456 took 29ms/5ms to execute
			//	check if we can access memory directly from the JVM and if we can, use AndroidEncoderBuffer and use remote SoyPixels when uploading to avoid the 6mb copy
			Java::BufferToArray( ByteBuffer, GetArrayBridge( PixelBufferArray ), "Encoder CopyFromOutputBuffer", ByteBufferSize );
			//std::Debug << "Extracted " << PixelBufferArray.GetSize() << " bytes / " << size << std::endl;
			
			//	gr: emulate note4
			//std::this_thread::sleep_for( std::chrono::milliseconds(30) );
			
			Frame.mPixels = pPixelBuffer;

			//	can release now we've copied them
			ReleaseBuffer = true;
		}
		
		auto BlockPush = [this]
		{
			std::Debug << "Frame buffer block test: " << IsWorking() << std::endl;
			return IsWorking();
		};
		
		//std::Debug << "PushPixelBuffer()..." << std::endl;
		Output.PushPixelBuffer( Frame, BlockPush );
		
		return ReleaseBuffer;
	};
	
	ProcessOutputPacket( HandleBuffer, Output );
}


void AndroidMediaDecoder::ProcessOutputPacket(TAudioBufferManager& Output)
{
	auto HandleBuffer = [&](std::shared_ptr<TJniObject> pByteBuffer,size_t ByteBufferSize,int OutputBufferId,const TStreamMeta& Meta,SoyTime Timestamp)
	{
		OnDecodeFrameSubmitted( Timestamp );

		//	gr: replace all this with the pass through encoder (or at least re-use the code)

		//	gr: look out for this
		//	http://developer.android.com/reference/android/media/MediaFormat.html
		//	KEY_IS_ADTS	Integer	optional, if decoding AAC audio content, setting this key to 1 indicates that each audio frame is prefixed by the ADTS header.
		
		std::Debug << "Handle audio data; " << Timestamp << "; " << Meta << std::endl;
		TAudioBufferBlock AudioBlock;
		AudioBlock.mChannels = Meta.mChannelCount;
		AudioBlock.mFrequency = Meta.mAudioSampleRate;
		AudioBlock.mStartTime = Timestamp;

		//	if this doens't throw then we can use it
		auto BufferArray = Java::GetBufferArray( *pByteBuffer, ByteBufferSize );
		static bool DebugAudioData = false;
		if ( DebugAudioData )
		{
			auto Data16 = CastArray<sint16>( GetArrayBridge(BufferArray) );
			std::Debug << "Audio data x" << Data16.GetSize() << ": " ;
			for ( int i=0;	i<Data16.GetSize();	i++ )
			{
				std::Debug << " " << (int)Data16[i];
			}
			std::Debug << std::endl;
		}
		
		auto Format = Meta.mCodec;

		//	android raw data is always 16 bit?
		//	http://stackoverflow.com/questions/23529654/does-mediacodec-always-give-16-bit-audio-output
		//	here (and comment on above) suggests it MAY NOT be...
		//	http://stackoverflow.com/questions/30246737/mediacodec-and-24-bit-pcm
		if ( Format == SoyMediaFormat::PcmAndroidRaw )
			Format = SoyMediaFormat::PcmLinear_16;
		
		//	convert to float audio
		if ( Format == SoyMediaFormat::PcmLinear_16  )
		{
			auto Data16 = CastArray<sint16>( GetArrayBridge(BufferArray) );
			Wave::ConvertSamples( GetArrayBridge(Data16), GetArrayBridge(AudioBlock.mData) );
			Output.mOnFrameDecoded.OnTriggered( Timestamp );
			Output.PushAudioBuffer( AudioBlock );
		}
		else if ( Format == SoyMediaFormat::PcmLinear_8 )
		{
			auto Data8 = CastArray<sint8>( GetArrayBridge(BufferArray) );
			Wave::ConvertSamples( GetArrayBridge(Data8), GetArrayBridge(AudioBlock.mData) );
			Output.mOnFrameDecoded.OnTriggered( Timestamp );
			Output.PushAudioBuffer( AudioBlock );
		}
		else if ( Format == SoyMediaFormat::PcmLinear_float )
		{
			auto Dataf = CastArray<float>( GetArrayBridge(BufferArray) );
			Wave::ConvertSamples( GetArrayBridge(Dataf), GetArrayBridge(AudioBlock.mData) );
			Output.mOnFrameDecoded.OnTriggered( Timestamp );
			Output.PushAudioBuffer( AudioBlock );
		}
		else
		{
			Output.mOnFrameDecodeFailed.OnTriggered( Timestamp );
			std::stringstream Error;
			Error << __func__ << " cannot handle " << Meta.mCodec;
			throw Soy::AssertException( Error.str() );
		}
		
	
		/*
		if ( mSurfaceTexture )
		{
			//	special buffer with texture
			std::Debug << "Making surface backed pixel buffer..." << std::endl;
			Frame.mPixels.reset( new AndroidEncoderBuffer( outputBufferId, mCodec, mSurfaceTexture ) );
		}
		else if ( pByteBufferPixels )
		{
			//	special buffer with access to byte buffer
			std::Debug << "Making byte-buffer backed pixel buffer..." << std::endl;
			Frame.mPixels.reset( new AndroidEncoderBuffer( outputBufferId, mCodec, pByteBufferPixels ) );
		}
		else
		{
			std::Debug << "Making buffer copy(slow!) dumb pixel buffer..." << std::endl;
			//	if using pixel buffer directly, make a dumb pixel buffer and release the buffer immediately
			if ( !OutputMeta.mPixelMeta.IsValid() )
			{
				std::stringstream Error;
				Error << "Using dumb pixel buffer for decoder output, but invalid meta; " << OutputMeta.mPixelMeta;
				throw Soy::AssertException( Error.str() );
			}
			
			//std::Debug << "Allocating dumb pixel buffer..." << std::endl;
			std::shared_ptr<TDumbPixelBuffer> pPixelBuffer( new TDumbPixelBuffer(OutputMeta.mPixelMeta) );
			auto& PixelBuffer = *pPixelBuffer;
			auto& PixelBufferArray = PixelBuffer.mPixels.GetPixelsArray();
			
			//std::Debug << "Copying buffer to pixel buffer..." << std::endl;
			auto ByteBuffer = mCodec->CallObjectMethod("getOutputBuffer", "java.nio.ByteBuffer", outputBufferId );
			
			//	copy pixels
			//	gr: this is very slow
			//	Encoder CopyFromOutputBuffer BufferSize=6291456/6291456 took 29ms/5ms to execute
			//	check if we can access memory directly from the JVM and if we can, use AndroidEncoderBuffer and use remote SoyPixels when uploading to avoid the 6mb copy
			Java::BufferToArray( ByteBuffer, GetArrayBridge( PixelBufferArray ), "Encoder CopyFromOutputBuffer", size );
			//std::Debug << "Extracted " << PixelBufferArray.GetSize() << " bytes / " << size << std::endl;
			
			//	gr: emulate note4
			//std::this_thread::sleep_for( std::chrono::milliseconds(30) );
			
			//	can release now we've copied them
			//std::Debug << "Release output buffer..." << std::endl;
			mCodec->CallVoidMethod("releaseOutputBuffer", outputBufferId, false );
			
			Frame.mPixels = pPixelBuffer;
		}
		
		auto BlockPush = [this]
		{
			std::Debug << "Frame buffer block test: " << IsWorking() << std::endl;
			return IsWorking();
		};
		
		//std::Debug << "PushPixelBuffer()..." << std::endl;
		FrameBuffer.PushPixelBuffer( Frame, BlockPush );
		* /
		return true;
	};
	
	ProcessOutputPacket( HandleBuffer, Output );
}




bool AndroidMediaDecoder::ProcessPacket(const TMediaPacket& Packet)
{
	//std::Debug << "Encoding packet " << Packet << std::endl;
	Soy::Assert( mCodec!=nullptr, "Codec expected" );
	if ( !mCodecStarted )
	{
		std::Debug << "Codec not started" << std::endl;
		return false;
	}
	
	//	get an input buffer to write input data (packets) to
	//	gr: don't block so that we don't deadlock the thread on destruction
	//		but also wait for a bit so we don't repeatedly call this and the packet re-insertion
	//	gr: 16ms tends to return once per-input-packet, so presumably these buffers are free'd up when the encoder
	//		is done with it, SO it's either tied to the pop & release our side, or the time it takes to decode
	SoyTime Timeout( std::chrono::milliseconds(30) );
	bool Block = false;
	bool Immediate = false;
	long TimeoutUs = Timeout.GetMicroSeconds();
	if ( Block )
		TimeoutUs = -1;
	if ( Immediate )
		TimeoutUs = 0;
	
	//	this throws when not started... okay, but maybe cleaner not to. or as it's multithreaded...wait
	int InputBufferId = mCodec->CallIntMethod("dequeueInputBuffer", TimeoutUs );
	if ( InputBufferId < 0 )
	{
		Java::IsOkay("Failed to get codec input buffer");
		//std::Debug << "Failed to get codec input buffer" << std::endl;
		return false;
	}

	//	added to debug detecting if we get to queue ANY frames for note-4 with surface
	std::Debug << "dequeueInputBuffer bufferid=" << InputBufferId << " for " << Packet << std::endl;
	
	try
	{
		auto InputBuffer = mCodec->CallObjectMethod("getInputBuffer", "java.nio.ByteBuffer", InputBufferId );
		Java::ArrayToBuffer( GetArrayBridge(Packet.mData), InputBuffer, "Encoder copy packet to buffer" );
	}
	catch(std::exception& e)
	{
		std::Debug << "Exception after dequeueInputBuffer(), " << e.what() << "; returning dummy packet to queue." << std::endl;
		int Offset = 0;
		int Size = 0;		//	will cause eof?
		int Flags = 0;
		long PresentationTimeMicrosecs = 0;
		mCodec->CallVoidMethod("queueInputBuffer", InputBufferId, Offset, Size, PresentationTimeMicrosecs, Flags );
		throw;
	}
	//	put input buffer back in the queue

	bool Eof = false;	//	gr: need to put this in media packet?
	int Offset = 0;
	int Size = Eof ? 0 : Packet.mData.GetDataSize();
	int Flags = 0;
	if ( Eof )
		Flags |= BUFFER_FLAG_END_OF_STREAM;
	if ( Packet.mIsKeyFrame )
		Flags |= BUFFER_FLAG_KEY_FRAME;
	long PresentationTimeMicrosecs = Packet.mTimecode.GetMicroSeconds();
	
	/*
	std::Debug << "Encoder queue input buffer ";
	std::Debug << " InputBufferId=" << InputBufferId;
	std::Debug << " Eof=" << Eof;
	std::Debug << " Packet.mIsKeyFrame=" << Packet.mIsKeyFrame;
	std::Debug << " PresentationTimeMicrosecs=" << PresentationTimeMicrosecs;
	std::Debug << " Offset=" << Offset;
	std::Debug << " Size=" << Size;
	std::Debug << " Packet decode time=" << Packet.mDecodeTimecode;
	std::Debug << std::endl;
	 * /
	mCodec->CallVoidMethod("queueInputBuffer", InputBufferId, Offset, Size, PresentationTimeMicrosecs, Flags );

	if ( mPixelOutput )
	{
		mPixelOutput->mOnFrameDecodeSubmission.OnTriggered( Packet.mTimecode );
	}
	if ( mAudioOutput )
	{
		mAudioOutput->mOnFrameDecodeSubmission.OnTriggered( Packet.mTimecode );
	}
	
	return true;
}
*/

/*
AndroidAudioDecoder::AndroidAudioDecoder(const std::string& ThreadName,const TStreamMeta& Stream,std::shared_ptr<TMediaPacketBuffer> InputBuffer,std::shared_ptr<TAudioBufferManager> OutputBuffer) :
	TMediaDecoder	( ThreadName, InputBuffer, OutputBuffer )
{
	Start();
}



bool AndroidAudioDecoder::ProcessPacket(const TMediaPacket& Packet)
{
	auto& Output = GetAudioBufferManager();
	
	TAudioBufferBlock AudioBlock;
	
	//	convert to float audio
	if ( Packet.mMeta.mCodec == SoyMediaFormat::PcmLinear_16 )
	{
		auto Data16 = CastArray<sint16>( GetArrayBridge(Packet.mData) );
		Wave::ConvertSamples( GetArrayBridge(Data16), GetArrayBridge(AudioBlock.mData) );
		Output.mOnFrameDecoded.OnTriggered( Packet.mTimecode );
		Output.PushAudioBuffer( AudioBlock );
	}
	else if ( Packet.mMeta.mCodec == SoyMediaFormat::PcmLinear_8 )
	{
		auto Data8 = CastArray<sint8>( GetArrayBridge(Packet.mData) );
		Wave::ConvertSamples( GetArrayBridge(Data8), GetArrayBridge(AudioBlock.mData) );
		Output.mOnFrameDecoded.OnTriggered( Packet.mTimecode );
		Output.PushAudioBuffer( AudioBlock );
	}
	else
	{
		Output.mOnFrameDecodeFailed.OnTriggered( Packet.mTimecode );
		std::stringstream Error;
		Error << __func__ << " cannot handle " << Packet.mMeta.mCodec;
		throw Soy::AssertException( Error.str() );
	}
	
	return true;
}

void AndroidAudioDecoder::ConvertPcmLinear16ToPcmFloat(const ArrayBridge<sint16>&& Input,ArrayBridge<float>&& Output)
{
	for ( int i=0;	i<Input.GetSize();	i++ )
	{
		float Samplef;
		Wave::ConvertSample( Input[i], Samplef );
		static float Gain = 1.f;
		Samplef *= Gain;
		Output.PushBack( Samplef );
	}
}

*/
