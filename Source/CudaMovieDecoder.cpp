#include "CudaMovieDecoder.h"


SoyPixelsFormat::Type Cuda::GetPixelFormat(cudaVideoChromaFormat Format)
{
	switch ( Format )
	{
		case cudaVideoChromaFormat_Monochrome:	return SoyPixelsFormat::LumaFull;
		case cudaVideoChromaFormat_420:			return SoyPixelsFormat::Yuv420_Biplanar_Video;
		case cudaVideoChromaFormat_422:			return SoyPixelsFormat::Yuv422_Biplanar_Full;
		case cudaVideoChromaFormat_444:			return SoyPixelsFormat::Yuv444_Biplanar_Full;
			
		default:
			return SoyPixelsFormat::Invalid;
	}
}


std::string Cuda::GetEnumString(cudaVideoChromaFormat Format)
{
#define CASE_ENUM_STRING(e)	case (e):	return #e;
	switch (Format)
	{
			CASE_ENUM_STRING(cudaVideoChromaFormat_Monochrome);
			CASE_ENUM_STRING(cudaVideoChromaFormat_420);
			CASE_ENUM_STRING(cudaVideoChromaFormat_422);
			CASE_ENUM_STRING(cudaVideoChromaFormat_444);
	};
#undef CASE_ENUM_STRING
	std::stringstream Unknown;
	Unknown << "Unknown cudaVideoChromaFormat " << Format;
	return Unknown.str();
}


std::string Cuda::GetEnumString(cudaVideoCodec Codec)
{
#define CASE_ENUM_STRING(e)	case (e):	return #e;
	switch (Codec)
	{
			CASE_ENUM_STRING(cudaVideoCodec_MPEG1);
			CASE_ENUM_STRING(cudaVideoCodec_MPEG2);
			CASE_ENUM_STRING(cudaVideoCodec_MPEG4);
			CASE_ENUM_STRING(cudaVideoCodec_VC1);
			CASE_ENUM_STRING(cudaVideoCodec_H264);
			CASE_ENUM_STRING(cudaVideoCodec_JPEG);
			CASE_ENUM_STRING(cudaVideoCodec_H264_SVC);
			CASE_ENUM_STRING(cudaVideoCodec_H264_MVC);
			CASE_ENUM_STRING(cudaVideoCodec_HEVC);
			CASE_ENUM_STRING(cudaVideoCodec_NumCodecs);
			CASE_ENUM_STRING(cudaVideoCodec_YUV420);
			CASE_ENUM_STRING(cudaVideoCodec_YV12);
			CASE_ENUM_STRING(cudaVideoCodec_NV12);
			CASE_ENUM_STRING(cudaVideoCodec_YUYV);
			CASE_ENUM_STRING(cudaVideoCodec_UYVY);
	};
#undef CASE_ENUM_STRING
	std::stringstream Unknown;
	Unknown << "Unknown cudaVideoCodec " << Codec;
	return Unknown.str();
}


SoyTime Cuda::GetTime(CUvideotimestamp Timestamp)
{
	//	"10mhz clock"
	auto TimeScalar = 10000;
	auto TimeMs = size_cast<uint64>( Timestamp / TimeScalar );
	return SoyTime( TimeMs );
}



std::ostream& operator<<(std::ostream &out,cudaVideoCodec& in)
{
	out << Cuda::GetEnumString( in );
	return out;
}

std::ostream& operator<<(std::ostream &out,cudaVideoChromaFormat& in)
{
	out << Cuda::GetEnumString( in );
	return out;
}

std::ostream& operator<<(std::ostream &out,CUVIDEOFORMAT& in)
{
	out << in.display_area.left << 'x' << in.display_area.top << 'x' << in.display_area.right << 'x' << in.display_area.bottom << ' ';
	out << in.codec << " " << in.chroma_format << " ";
	out << "bitrate: " << in.bitrate;
	return out;
}

std::ostream& operator<<(std::ostream &out,CUVIDPICPARAMS& in)
{
	out << "PicIndex: " << in.CurrPicIdx << " ";
	out << "Slices: " << in.nNumSlices << " ";
	return out;
}

std::ostream& operator<<(std::ostream &out,CUVIDPARSERDISPINFO& in)
{
	out << "PicIndex: " << in.picture_index << " ";
	out << "ProgressiveFrame: " << in.progressive_frame << " ";
	out << "Timestamp: " << Cuda::GetTime( in.timestamp ) << " ";

	return out;
}


Cuda::TVideoLock::TVideoLock(Cuda::TContext& Context) :
	mLock		(nullptr)
{
	auto Error = cuvidCtxLockCreate( &mLock, Context.GetContext() );
	Cuda::IsOkay( Error, "cuvidCtxLockCreate" );
}

Cuda::TVideoLock::~TVideoLock()
{
	if ( mLock )
	{
		auto Error = cuvidCtxLockDestroy( mLock );
		Cuda::IsOkay( Error, "cuvidCtxLockDestroy" );
		mLock = nullptr;
	}
}


bool Cuda::TVideoLock::Lock()
{
	if ( !mLock )
		return false;
	
	auto Error = cuvidCtxLock( mLock, 0 );
	Cuda_IsOkay( Error );
	
	return true;
};


void Cuda::TVideoLock::Unlock()
{
	if ( !mLock )
		return;
	
	auto Error = cuvidCtxUnlock( mLock, 0 );
	Cuda_IsOkay( Error );
}




CudaVideoDecoder::CudaVideoDecoder(const TVideoDecoderParams& Params,std::shared_ptr<Cuda::TContext>& pContext) :
	TVideoDecoder(Params),
	mDecoder(nullptr),
	mParser(nullptr),
	mSource(nullptr),
	mContext( pContext ),
	mTargetMeta( 2048, 2048, SoyPixelsFormat::RGBA )
{
	Soy::Assert( mContext != nullptr, "Context expected" );
	auto& Context = *mContext;

	//	create lock
	mLock.reset(new Cuda::TVideoLock(Context));

	SetWakeMode(SoyWorkerWaitMode::Sleep);

	//	create cuda objects
	if (!Context.Lock())
		throw Soy::AssertException("Failed to lock cuda context");


	try
	{
		CreateParser(Params);
		CreateSource(Params);
		CreateDecoder(Params,Context);
	}
	catch (...)
	{
		Context.Unlock();
		throw;
	}
}


CudaVideoDecoder::~CudaVideoDecoder()
{
	if (mDecoder)
	{
		auto Result = cuvidDestroyDecoder(mDecoder);
		mDecoder = nullptr;
	}

}


void CudaVideoDecoder::StartMovie(Opengl::TContext& Context)
{
	auto Result = cuvidSetVideoSourceState(mSource, cudaVideoState_Started );
	Cuda_IsOkay(Result);

	TVideoDecoder::StartMovie( Context );
}

bool CudaVideoDecoder::PauseMovie(Opengl::TContext& Context)
{
	auto Result = cuvidSetVideoSourceState(mSource, cudaVideoState_Stopped);
	Cuda_IsOkay(Result);

	TVideoDecoder::PauseMovie( Context );
	return true;
}

void CudaVideoDecoder::Shutdown(Opengl::TContext& Context)
{
	TVideoDecoder::Shutdown( Context );
}


bool CudaVideoDecoder::Iteration()
{
	/*
	// Decode a single picture (field or frame)
	CUresult cuvidDecodePicture(CUvideodecoder hDecoder,
		CUVIDPICPARAMS *pPicParams);
	// Post-process and map a video frame for use in cuda
	CUresult cuvidMapVideoFrame(CUvideodecoder hDecoder, int nPicIdx,
		CUdeviceptr *pDevPtr, unsigned int *pPitch,
		CUVIDPROCPARAMS *pVPP);
	// Unmap a previously mapped video frame
	CUresult cuvidUnmapVideoFrame(CUvideodecoder hDecoder, CUdeviceptr DevPtr);
	*/
	return true;
}

SoyTime CudaVideoDecoder::GetDuration()
{
	return SoyTime();
}

cudaVideoCodec CudaVideoDecoder::GetCodec()
{
	return cudaVideoCodec_H264;
}

size_t CudaVideoDecoder::GetMaxDecodeSurfaces()
{
	return 4;
}

size_t CudaVideoDecoder::GetMaxOutputSurfaces()
{
	return 4;
}

void CudaVideoDecoder::OnVideoPacket(CUVIDSOURCEDATAPACKET& Packet)
{
	std::Debug << "Video packet" << std::endl;
	auto Result = cuvidParseVideoData( mParser, &Packet );

	bool EndOfStream = (Packet.flags & CUVID_PKT_ENDOFSTREAM);
	if ( EndOfStream )
		std::Debug << "End of video stream" << std::endl;
}

void CudaVideoDecoder::OnAudioPacket(CUVIDSOURCEDATAPACKET& Packet)
{
	std::Debug << "Video packet" << std::endl;
	//auto Result = cuvidParseAudioData(mVideoParser, &Packet);
	bool EndOfStream = (Packet.flags & CUVID_PKT_ENDOFSTREAM);
	if (EndOfStream)
		std::Debug << "End of audio stream" << std::endl;
}

void CudaVideoDecoder::OnFrameParams(CUVIDPICPARAMS& Frame)
{
	std::Debug << "HandlePictureDecode" << Frame << std::endl;

	//	store this frame's data
	mPicParams[Frame.CurrPicIdx] = Frame;
}


CUVIDPICPARAMS CudaVideoDecoder::GetFrameParams(int Frame)
{
	auto it = mPicParams.find( Frame );
	if ( it == mPicParams.end() )
		throw Soy::AssertException("Frame data not found");

	return it->second;
}


void CudaVideoDecoder::OnDisplayFrame(CUVIDPARSERDISPINFO& DisplayInfo)
{
	std::Debug << "HandlePictureDisplay" << DisplayInfo <<  std::endl;

	if ( !Soy::Assert( mContext!=nullptr, "Context expected" ) )
		return;

	//	gr: find out which thread we block
	static bool Block = false;

	TPixelBufferFrame Frame;
	Frame.mTimestamp = Cuda::GetTime( DisplayInfo.timestamp );
	if ( !Frame.mTimestamp.IsValid() )
	{
		std::Debug << "Invalid timestamp in CUVIDPARSERDISPINFO, frame dropped" << std::endl;
		mOnFramePushFailed.OnTriggered( Frame.mTimestamp );
		return;
	}
	Frame.mPixels.reset( new Cuda::TDisplayFrame(DisplayInfo,*this) );

	PushPixelBuffer( Frame, Block );
}

void CudaVideoDecoder::CreateSource(const TVideoDecoderParams& Params)
{
	auto VideoHandler = [](void *pUserData, CUVIDSOURCEDATAPACKET *pPacket)
	{
		auto& This = *reinterpret_cast<CudaVideoDecoder*>(pUserData);
		int Continue = 1;
		This.OnVideoPacket(*pPacket);
		return Continue;
	};

	auto AudioHandler = [](void *pUserData, CUVIDSOURCEDATAPACKET *pPacket)
	{
		auto& This = *reinterpret_cast<CudaVideoDecoder*>(pUserData);
		int Continue = 1;
		This.OnAudioPacket(*pPacket);
		return Continue;
	};


	CUVIDSOURCEPARAMS VideoSourceParams;
	MemsetZero(VideoSourceParams);

	VideoSourceParams.pUserData = this;
	VideoSourceParams.pfnVideoDataHandler = VideoHandler;   // our local video-handler callback
	VideoSourceParams.pfnAudioDataHandler = AudioHandler;

	//	cuda can't handle bad slashes
	std::string Filename = Params.mFilename;
	
	auto Result = cuvidCreateVideoSource( &mSource, Filename.c_str(), &VideoSourceParams );
	Cuda_IsOkay(Result);
}


bool CudaVideoDecoder::IsStarted()
{
	auto State = cuvidGetVideoSourceState(mSource);
	return State == cudaVideoState_Started;
}

SoyPixelsMeta CudaVideoDecoder::GetDecoderMeta()
{
	auto Format = GetSourceFormat();

	auto Width = Format.display_area.right - Format.display_area.left;
	auto Height = Format.display_area.bottom - Format.display_area.top;

	//	colour 
	SoyPixelsMeta Meta( Width, Height, Cuda::GetPixelFormat(Format.chroma_format) );

	return Meta;
}

std::shared_ptr<Cuda::TStream> CudaVideoDecoder::GetStream()
{
	if ( !mStream )
		mStream.reset( new Cuda::TStream() );

	return mStream;
}


void CudaVideoDecoder::CreateParser(const TVideoDecoderParams& Params)
{
	auto HandleVideoSequence = [](void* UserData, CUVIDEOFORMAT * Format)
	{
		auto& This = *reinterpret_cast<CudaVideoDecoder*>(UserData);
		std::Debug << "HandleVideoSequence " << *Format << std::endl;
		int Result = 1;
		return Result;
	};
	auto HandlePictureDecode = [](void* UserData, CUVIDPICPARAMS * PicParams)
	{
		auto& This = *reinterpret_cast<CudaVideoDecoder*>(UserData);
		This.OnFrameParams(*PicParams);
		int Result = 1;
		return Result;
	};
	auto HandlePictureDisplay = [](void* UserData, CUVIDPARSERDISPINFO *DisplayInfo)
	{
		auto& This = *reinterpret_cast<CudaVideoDecoder*>(UserData);
		This.OnDisplayFrame( *DisplayInfo );
		int Result = 1;
		return Result;
	};

	CUVIDPARSERPARAMS oVideoParserParameters;
	MemsetZero(oVideoParserParameters);
	memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
	oVideoParserParameters.CodecType = GetCodec();
	oVideoParserParameters.ulMaxNumDecodeSurfaces = size_cast<unsigned int>(GetMaxDecodeSurfaces());
	oVideoParserParameters.ulMaxDisplayDelay = 1;  // this flag is needed so the parser will push frames out to the decoder as quickly as it can
	oVideoParserParameters.pUserData = this;
	oVideoParserParameters.pfnSequenceCallback = HandleVideoSequence;    // Called before decoding frames and/or whenever there is a format change
	oVideoParserParameters.pfnDecodePicture = HandlePictureDecode;    // Called when a picture is ready to be decoded (decode order)
	oVideoParserParameters.pfnDisplayPicture = HandlePictureDisplay;   // Called whenever a picture is ready to be displayed (display order)
	auto Result = cuvidCreateVideoParser( &mParser, &oVideoParserParameters);
	Cuda_IsOkay( Result );
}


CUVIDEOFORMAT CudaVideoDecoder::GetSourceFormat()
{
	CUVIDEOFORMAT Format;
	unsigned int GetFormatFlags = 0;
	auto Error = cuvidGetSourceVideoFormat( mSource, &Format, GetFormatFlags );
	Cuda::IsOkay( Error, "cuvidGetSourceVideoFormat" );
	return Format;
}

void CudaVideoDecoder::CreateDecoder(const TVideoDecoderParams& Params,Cuda::TContext& Context)
{
	CUVIDEOFORMAT VideoFormat = GetSourceFormat();
	std::Debug << "Creating decoder with source: " << VideoFormat << std::endl;

	//cudaVideoCreateFlags Flags = cudaVideoCreate_PreferCUDA;
	//cudaVideoCreateFlags Flags = cudaVideoCreate_PreferDXVA;
	cudaVideoCreateFlags Flags = cudaVideoCreate_PreferCUVID;

	CUVIDDECODECREATEINFO CreateInfo;
	MemsetZero( CreateInfo );

	CreateInfo.CodecType = VideoFormat.codec;
    CreateInfo.ulWidth = VideoFormat.coded_width;
    CreateInfo.ulHeight = VideoFormat.coded_height;
    CreateInfo.ulNumDecodeSurfaces = GetMaxDecodeSurfaces();
	CreateInfo.ChromaFormat = VideoFormat.chroma_format;
	CreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
	CreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;

	/*
	//	Limit decode memory to 24MB (16M pixels at 4:2:0 = 24M bytes)
	while (oVideoDecodeCreateInfo_.ulNumDecodeSurfaces * rVideoFormat.coded_width * rVideoFormat.coded_height > 16 * 1024 * 1024)
		CreateInfo.ulNumDecodeSurfaces--;
	*/

	// No scaling
	CreateInfo.ulTargetWidth = mTargetMeta.GetWidth();
	CreateInfo.ulTargetHeight = mTargetMeta.GetHeight();
	CreateInfo.ulNumOutputSurfaces = GetMaxOutputSurfaces();
	CreateInfo.ulCreationFlags = Flags;
	CreateInfo.vidLock = mLock->GetLock();

	// create the decoder
	CUresult Error = cuvidCreateDecoder( &mDecoder, &CreateInfo );
	Cuda::IsOkay(Error,"cuvidCreateDecoder");
}


std::shared_ptr<Cuda::TBuffer> CudaVideoDecoder::GetDisplayFrameBuffer(size_t BufferIndex,size_t DataSize)
{
	if ( BufferIndex >= mDisplayFrameBuffers.GetSize() )
		mDisplayFrameBuffers.SetSize( BufferIndex+1, true );

	//DataSize =  (nDecodedPitch * nHeight * 3 / 2);
	auto& Buffer = mDisplayFrameBuffers[BufferIndex];

	if ( !Buffer )
		Buffer.reset( new Cuda::TBuffer(DataSize) );

	return Buffer;
}


std::shared_ptr<Cuda::TBuffer> CudaVideoDecoder::GetInteropFrameBuffer(size_t BufferIndex,size_t DataSize)
{
	if ( BufferIndex >= mInteropFrameBuffers.GetSize() )
		mInteropFrameBuffers.SetSize( BufferIndex+1, true );

	auto& Buffer = mInteropFrameBuffers[BufferIndex];

	if ( !Buffer )
		Buffer.reset( new Cuda::TBuffer(DataSize) );

	return Buffer;
}




Cuda::TDisplayFrame::TDisplayFrame(CUVIDPARSERDISPINFO& DisplayInfo,CudaVideoDecoder& Decoder) :
	mOpenglSupport	( false ),
	mParent			( Decoder ),
	mDisplayInfo	( DisplayInfo )
{
}

void Cuda::TDisplayFrame::Lock(ArrayBridge<Opengl::TTexture>&& Textures, Opengl::TContext& Context)
{
}

void Cuda::TDisplayFrame::Lock(ArrayBridge<SoyPixelsImpl*>&& Textures)
{
	auto pContext = mParent.GetContext();
	if (!Soy::Assert(pContext != nullptr, "expected context"))
		return;
	auto& Context = *pContext;

	//	lock context
	//	gr: will this unlock with an exception?
	Context.Lock();
 


	// Push the current CUDA context (only if we are using CUDA decoding path)
	//	gr: attach context to current thread. I think
	auto Error = cuCtxPushCurrent( Context.GetContext() );
	Cuda::IsOkay( Error, "cuCtxPushCurrent" );
	uint8* Pixels = nullptr;

	try
	{
		UpdateFramePixels( Textures, Context );
	}
	catch ( std::exception& e )
	{
		std::Debug << "Failed to update pixels " << e.what() << std::endl;
		Unlock();
		throw;
	}
}

void Cuda::TDisplayFrame::UpdateFramePixels(ArrayBridge<SoyPixelsImpl*>& Textures,Cuda::TContext& Context)
{
	int num_fields = (mDisplayInfo.progressive_frame ? (1) : (2+mDisplayInfo.repeat_first_field));


	for (int active_field=0; active_field<num_fields; active_field++)
    {
        auto nRepeats = mDisplayInfo.repeat_first_field;
        CUVIDPROCPARAMS oVideoProcessingParameters;
		MemsetZero( oVideoProcessingParameters );

        oVideoProcessingParameters.progressive_frame = mDisplayInfo.progressive_frame;
        oVideoProcessingParameters.second_field      = active_field;
        oVideoProcessingParameters.top_field_first   = mDisplayInfo.top_field_first;
        oVideoProcessingParameters.unpaired_field    = (num_fields == 1);

        unsigned int nDecodedPitch = 0;
		CUdeviceptr DecodedFramePtr = CUDA_INVALID_VALUE;

		//	from cuviddec.h
		
// Overall data flow:
//  - cuvidCreateDecoder(...)
//  For each picture:
//  - cuvidDecodePicture(N)
//  - cuvidMapVideoFrame(N-4)
//  - do some processing in cuda
//  - cuvidUnmapVideoFrame(N-4)
//  - cuvidDecodePicture(N+1)
//  - cuvidMapVideoFrame(N-3)
//    ...
//  - cuvidDestroyDecoder(...)
//
// NOTE:
// - In the current version, the cuda context MUST be created from a D3D device, using cuD3D9CtxCreate function.
//   For multi-threaded operation, the D3D device must also be created with the D3DCREATE_MULTITHREADED flag.
// - There is a limit to how many pictures can be mapped simultaneously (ulNumOutputSurfaces)
// - cuVidDecodePicture may block the calling thread if there are too many pictures pending 
//   in the decode queue
//
		CUVIDPICPARAMS Params = mParent.GetFrameParams( mDisplayInfo.picture_index );
		auto Result = cuvidDecodePicture( mParent.GetDecoder(), &Params );
		Cuda::IsOkay(Result,"cuvidDecodePicture");

        // map decoded video frame to CUDA surface
		Result = cuvidMapVideoFrame( mParent.GetDecoder(), mDisplayInfo.picture_index, &DecodedFramePtr, &nDecodedPitch, &oVideoProcessingParameters );
		Cuda::IsOkay(Result,"cuvidMapVideoFrame");
		Soy::Assert( DecodedFramePtr != CUDA_INVALID_VALUE, "cuvidMapVideoFrame null device ptr" );
		Soy::Assert( nDecodedPitch != 0, "cuvidMapVideoFrame pitch is zero" );

		auto TargetMeta = mParent.GetTargetMeta();
		auto DecoderMeta = mParent.GetDecoderMeta();
		unsigned int nWidth = DecoderMeta.GetWidth();
        unsigned int nHeight = DecoderMeta.GetHeight();

		//	gr: im assuming cuvid needs aligned buffers tow ork....
		//nWidth  = PAD_ALIGN(g_pVideoDecoder->targetWidth() , 0x3F);
        //nHeight = PAD_ALIGN(g_pVideoDecoder->targetHeight(), 0x0F);
        // map OpenGL PBO or CUDA memory
        size_t pFramePitch = 0;

        // If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
		auto DataSize = (nDecodedPitch * nHeight * 3 / 2);
		auto Buffer = mParent.GetDisplayFrameBuffer(active_field,DataSize);

		auto Stream = mParent.GetStream();
		Soy::Assert( Stream!=nullptr, "Failed to get stream from CudaVideoDecoder");

		// If streams are enabled, we can perform the readback to the host while the kernel is executing
		Buffer->Read( DecodedFramePtr, *Stream, true );
		Result = cuStreamSynchronize( Stream->mStream );
		Cuda::IsOkay( Result, "cuStreamSynchronize" );
		
		mLockedPixelBuffers.PushBack(Buffer);

		//	need to split pixels here
		Array<std::shared_ptr<SoyPixelsImpl>> PixelPlanes;
		Buffer->SplitPlanes( GetArrayBridge(PixelPlanes) );
		for ( int p=0;	p<PixelPlanes.GetSize();	p++ )
		{
			mLockedPixels.PushBackArray( PixelPlanes[p] );
			Textures.PushBack( PixelPlanes[p].get() );
		}
		
		/*
  		CUdeviceptr InteropFramePtr = CUDA_INVALID_VALUE;
      if (mOpenglSupprot)
        {
            // map the texture surface
            g_pImageGL->map(&pInteropFrame[active_field], &pFramePitch, active_field);
            pFramePitch = g_nWindowWidth * 4;
        }
        else
		{
			size_t InteropFrameSize = TargetMeta.GetWidth() * TargetMeta.GetHeight() * 2;
			pInteropFrame[active_field] = mParent.GetInteropFrameBuffer( active_field, InteropFrameSize );
			pFramePitch = g_pVideoDecoder->targetWidth() * 2;
		}

        // perform post processing on the CUDA surface (performs colors space conversion and post processing)
        // comment this out if we inclue the line of code seen above
		
       // cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field], pFramePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);

		

        if (mOpenglSupport)
        {
            // unmap the texture surface
            g_pImageGL->unmap(active_field);
        }
		*/

		// unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
		Result = cuvidUnmapVideoFrame(  mParent.GetDecoder(), DecodedFramePtr );
		Cuda::IsOkay(Result,"cuvidUnmapVideoFrame");
    }

}


void Cuda::TDisplayFrame::Unlock()
{
	//	release buffer [references]
	mLockedPixelBuffers.Clear();
	mLockedPixels.Clear();

	//	detach from the Current thread
	auto Result = cuCtxPopCurrent(nullptr);
	Cuda::IsOkay(Result,"cuCtxPopCurrent");

	auto pContext = mParent.GetContext();
	Soy::Assert(pContext!=nullptr, "expected context");
	auto& Context = *pContext;
	Context.Unlock();

}
