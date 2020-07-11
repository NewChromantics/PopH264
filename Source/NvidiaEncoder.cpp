#include "SoyPixels.h"


//	fake some linux stuff so we can compile on other platforms
#if !defined(TARGET_LINUX)
#include <stdint.h>
typedef int8_t __s8;
typedef uint8_t __u8;
typedef int16_t __s16;
typedef uint16_t __u16;
typedef int32_t __s32;
typedef uint32_t __u32;
typedef int64_t __s64;
typedef uint64_t __u64;

typedef int8_t s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;
typedef int64_t s64;
typedef uint64_t u64;
#define __user
typedef uint32_t __le32;

#include <linux/videodev2.h>
#include "Linux/include/linux/videodev2.h"
#include <linux/v4l2-controls.h>
#endif

//#include "nvidia/samples/01_video_encode/video_encode.h"
#include "NvidiaEncoder.h"
#include "nvidia/include/NvVideoEncoder.h"

namespace Nvidia
{
	void	IsOkay(int Result, const char* Context);
	void	Log(void *data, int i_level, const char *psz, va_list args);

	int		GetColourSpace(SoyPixelsFormat::Type Format);

	class TPacket;
}

void Nvidia::IsOkay(int Result, const char* Context)
{
	if(Result == 0)
		return;

	std::stringstream Error;
	Error << "nvidia error " << Result << " (" << Context << ")";
	throw Soy::AssertException(Error);
}


namespace V4lPixelFormat
{
	enum Type : uint32_t
	{
		YUV420M = V4L2_PIX_FMT_YUV420M,
	};
}


//	to make compiling easier, we have a more native class
//	which references any linux/nvidia types
class Nvidia::TNative
{
public:
	void	OnFrameEncoded(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,NvBuffer * shared_buffer);
	
public:
	v4l2_memory	mMemoryMode = V4L2_MEMORY_USERPTR;
};


Nvidia::TEncoderParams::TEncoderParams(json11::Json& Options)
{
	/*
	auto SetInt = [&](const char* Name,size_t& ValueUnsigned)
	{
		auto& Handle = Options[Name];
		if ( !Handle.is_number() )
			return false;
		auto Value = Handle.int_value();
		if ( Value < 0 )
		{
			std::stringstream Error;
			Error << "Value for " << Name << " is " << Value << ", not expecting negative";
			throw Soy::AssertException(Error);
		}
		ValueUnsigned = Value;
		return true;
	};
	auto SetBool = [&](const char* Name, bool& Value)
	{
		auto& Handle = Options[Name];
		if (!Handle.is_bool())
			return false;
		Value = Handle.bool_value();
		return true;
	};
	SetInt(POPH264_ENCODER_KEY_QUALITY, mPreset);
	SetInt(POPH264_ENCODER_KEY_PROFILELEVEL, mProfileLevel);
	SetInt(POPH264_ENCODER_KEY_ENCODERTHREADS, mEncoderThreads);
	SetInt(POPH264_ENCODER_KEY_LOOKAHEADTHREADS, mLookaheadThreads);
	SetBool(POPH264_ENCODER_KEY_BSLICEDTHREADS, mBSlicedThreads);
	SetBool(POPH264_ENCODER_KEY_VERBOSEDEBUG, mEnableLog);
	SetBool(POPH264_ENCODER_KEY_DETERMINISTIC, mDeterministic);
	SetBool(POPH264_ENCODER_KEY_CPUOPTIMISATIONS, mCpuOptimisations);
	
	//	0 is auto on AVF, so handle that
	if (mProfileLevel == 0)
		mProfileLevel = 30;
	*/
}

V4lPixelFormat::Type GetPixelFormat(SoyPixelsFormat::Type Format)
{
	switch(Format)
	{
		case SoyPixelsFormat::Yuv_844:
			return V4lPixelFormat::YUV420M;
			
		default:break;
	}
	
	std::stringstream Error;
	Error << "No conversion from " << Format << " to v4l";
	throw Soy::AssertException(Error);
}



Nvidia::TEncoder::TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutPacket) :
	PopH264::TEncoder( OnOutPacket )
{
	mpNative.reset(new TNative);
	
	//	the nvvideoencoder is a wrapper for a V4L2 device
	//	which is a standard linux file i/o stream
	//	alloc an encoder in blocking mode
	//	so the non-blocking mode is an IO mode
	//	gr: what is enc0 ? device name?
	//ctx.enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);
	std::Debug << "Creating video encoder" << std::endl;
	mEncoder = NvVideoEncoder::createVideoEncoder("enc0");
	if ( !mEncoder )
		throw Soy::AssertException("Failed to allocate nvidia encoder");
	
	/*
	auto& Encoder = *mEncoder;
	auto Result = Encoder.subscribeEvent(V4L2_EVENT_EOS,0,0);
	IsOkay(Result,"Failed to subscribe to EOS event");
	*/
}

void Nvidia::TEncoder::InitEncoder(SoyPixelsMeta PixelMeta)
{
	if ( mInitialised )
		return;
	
	//	"capture" plane needs to be set before "output"
	InitH264Format(PixelMeta);
	InitYuvFormat(PixelMeta);
	InitH264Callback();
	InitYuvCallback();
	
	mInitialised = true;
}


void Nvidia::TEncoder::InitDmaBuffers(size_t BufferCount)
{
	Soy_AssertTodo();
	/*
	int ret=0;
	NvBufferCreateParams cParams;
	int fd;
	ret = ctx->enc->output_plane.reqbufs(V4L2_MEMORY_DMABUF,num_buffers);
	if(ret)
	{
		cerr << "reqbufs failed for output plane V4L2_MEMORY_DMABUF" << endl;
		return ret;
	}
	for (uint32_t i = 0; i < ctx->enc->output_plane.getNumBuffers(); i++)
	{
		cParams.width = ctx->width;
		cParams.height = ctx->height;
		cParams.layout = NvBufferLayout_Pitch;
		if (ctx->enableLossless && ctx->encoder_pixfmt == V4L2_PIX_FMT_H264)
		{
			cParams.colorFormat = NvBufferColorFormat_YUV444;
		}
		else if (ctx->profile == V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10)
		{
			cParams.colorFormat = NvBufferColorFormat_NV12_10LE;
		}
		else
		{
			cParams.colorFormat = ctx->enable_extended_colorformat ?
			NvBufferColorFormat_YUV420_ER : NvBufferColorFormat_YUV420;
		}
		cParams.nvbuf_tag = NvBufferTag_VIDEO_ENC;
		cParams.payloadType = NvBufferPayload_SurfArray;
		// Create output plane fd for DMABUF io-mode
		ret = NvBufferCreateEx(&fd, &cParams);
		if(ret < 0)
		{
			cerr << "Failed to create NvBuffer" << endl;
			return ret;
		}
		ctx->output_plane_fd[i]=fd;
	}
	return ret;
	 */
}

void Nvidia::TEncoder::InitYuvFormat(SoyPixelsMeta PixelMeta)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	//	output plane = yuv input plane
	auto& YuvPlane = GetYuvPlane();
	auto& Encoder = *mEncoder;

	//	the input is a "capture" plane
	//SoyPixelsMeta PixelMeta(100,100,SoyPixelsFormat::Yuv_844);
	//auto PixelFormat = GetPixelFormat( PixelMeta.GetFormat() );
	auto Format = V4L2_PIX_FMT_YUV420M;
	auto Width = PixelMeta.GetWidth();
	auto Height = PixelMeta.GetHeight();
	//auto MaxSize = InputFormat.GetDataSize();
	auto Result = Encoder.setOutputPlaneFormat( Format, Width, Height );
	IsOkay(Result,"InitInputFormat failed setOutputPlaneFormat");

	
	InitEncodingParams();

	//	setup memory read mode
	auto& mMemoryMode = mNative.mMemoryMode;
	switch( mMemoryMode )
	{
		case V4L2_MEMORY_MMAP:
			std::Debug << "Init memory mode V4L2_MEMORY_MMAP" << std::endl;
			Result = YuvPlane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
			break;
			
		case V4L2_MEMORY_USERPTR:
			std::Debug << "Init memory mode V4L2_MEMORY_USERPTR" << std::endl;
			Result = YuvPlane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
			break;
			
		case V4L2_MEMORY_DMABUF:
			std::Debug << "Init memory mode V4L2_MEMORY_DMABUF" << std::endl;
			InitDmaBuffers(10);
			Result = 0;
			break;
			
		default:
		{
			std::stringstream Debug;
			Debug << "Unhandled memory mode " << magic_enum::enum_name(mMemoryMode);
			throw Soy::AssertException(Debug);
		}
	}
	IsOkay(Result,"Setting up memory mode");
}


NvV4l2ElementPlane& Nvidia::TEncoder::GetYuvPlane()
{
	auto& Encoder = *mEncoder;
	auto& YuvPlane = Encoder.output_plane;
	return YuvPlane;
}

NvV4l2ElementPlane& Nvidia::TEncoder::GetH264Plane()
{
	auto& Encoder = *mEncoder;
	auto& H264Plane = Encoder.capture_plane;
	return H264Plane;
}


void Nvidia::TEncoder::InitEncodingParams()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	auto& Encoder = *mEncoder;

	auto Profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
	auto BitRate = 8 * 1000 * 500;
	auto Level = V4L2_MPEG_VIDEO_H264_LEVEL_3_0;
	
	//	set other params
	auto Result = Encoder.setProfile(Profile);
	IsOkay(Result,"Failed to set level");

	Result = Encoder.setBitrate(BitRate);
	IsOkay(Result,"Failed to set bitrate");
	
	Result = Encoder.setLevel(Level);
	IsOkay(Result,"Failed to set level");
}


//	nvidia needs to know w/h
void Nvidia::TEncoder::InitH264Format(SoyPixelsMeta InputMeta)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;

	//	capture plane = h264 output plane
	auto& H264Plane = GetH264Plane();
	auto& Encoder = *mEncoder;
	
	//	needs to be specified
	auto Width = InputMeta.GetWidth();
	auto Height = InputMeta.GetHeight();
	auto BufferSize = 1024 * 1024 * 2;
	auto Format = V4L2_PIX_FMT_H264;

	auto Result = Encoder.setCapturePlaneFormat( Format, Width, Height, BufferSize );
	IsOkay(Result,"InitOutputFormat failed setCapturePlaneFormat");
}


void Nvidia::TEncoder::InitH264Callback()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;

	//	CFunc callback
	//	encoder_capture_plane_dq_callback
	auto EncoderCallback = [](struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
							  NvBuffer * shared_buffer, void * This)
	{
		auto* ThisEncoder = reinterpret_cast<Nvidia::TEncoder*>(This);
		//ThisEncoder->mNative.OnFrameEncoded( v4l2_buf, buffer, shared_buffer );
		std::Debug << "H264 plane dequeue (frame encoded)" << std::endl;
		
		//	gr: what are we returning!
		return true;
	};
	
	auto& H264Plane = GetH264Plane();
	
	auto* This = this;
	H264Plane.startDQThread(This);
	H264Plane.setDQThreadCallback(EncoderCallback);

	//	queue empty buffers for encoder to use
	for ( auto i=0;	i<H264Plane.getNumBuffers();	i++ )
	{
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		
		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
		
		v4l2_buf.index = i;
		v4l2_buf.m.planes = planes;
		
		auto Result = H264Plane.qBuffer(v4l2_buf, nullptr);
		IsOkay(Result,"Failed to queue H264/capture_plane buffer");
	}
}



void Nvidia::TEncoder::InitYuvCallback()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	auto& YuvPlane = GetYuvPlane();
	
	//	CFunc callback
	//	encoder_capture_plane_dq_callback
	auto Callback = [](struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
							  NvBuffer * shared_buffer, void * This)
	{
		auto* ThisEncoder = reinterpret_cast<Nvidia::TEncoder*>(This);
		//ThisEncoder->mNative.OnFrameEncoded( v4l2_buf, buffer, shared_buffer );
		std::Debug << "YUV plane dequeue (index = " << v4l2_buf->index << ")" << std::endl;
	
		//	what is return!
		return true;
	};

	//	setup callback
	auto* This = this;
	YuvPlane.startDQThread(This);
	YuvPlane.setDQThreadCallback(Callback);

	
	/*
	//	Enqueue all the empty capture plane buffers.
	//	gr: which are the H264 output buffers
	for (uint32_t i = 0; i <Encoder.capture_plane.getNumBuffers(); i++)
	{
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		
		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
		
		v4l2_buf.index = i;
		v4l2_buf.m.planes = planes;
		
		auto Result = Encoder.capture_plane.qBuffer(v4l2_buf, nullptr);
		IsOkay(Result,"Failed to queue H264/capture_plane buffer");
	}
	*/
}

/*
void Nvidia::TEncoder::OnFrameEncoded(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,NvBuffer * shared_buffer)
{
	auto& Encoder = *mEncoder;

	//	gr: number of frames ready?
	uint32_t frame_num = Encoder.capture_plane.getTotalDequeuedBuffers() - 1;
	uint32_t ReconRef_Y_CRC = 0;
	uint32_t ReconRef_U_CRC = 0;
	uint32_t ReconRef_V_CRC = 0;
	static uint32_t num_encoded_frames = 1;
	struct v4l2_event ev;
	int ret = 0;
	
	if (!v4l2_buf)
	{
		std::Debug << __PRETTY_FUNCTION__ << " error while dequeing output buffer (v4l2_buf=null)" << std::endl;
		return;
	}
	
	if ( mUsingCommands )
	{
		if(v4l2_buf->flags & V4L2_BUF_FLAG_LAST)
		{
			std::Debug << __PRETTY_FUNCTION__ << "V4L2_BUF_FLAG_LAST" << std::endl;
			/*
			memset(&ev,0,sizeof(struct v4l2_event));
			ret = ctx->enc->dqEvent(ev,1000);
			if (ret < 0)
				cout << "Error in dqEvent" << endl;
			if(ev.type == V4L2_EVENT_EOS)
				return false;
			*//*
		}
	}
	
	//	end of string
	if (buffer->planes[0].bytesused == 0)
	{
		std::Debug << "Got 0 size buffer in capture" << std::endl;
		return;
	}
	
	//	Computing CRC with each frame
	//if(ctx->pBitStreamCrc)
	//	CalculateCrc (ctx->pBitStreamCrc, buffer->planes[0].data, buffer->planes[0].bytesused);
	

	//	push/read the encoded buffer
	OnFrame(buffer);

	//	get frame meta
	/*
	{
		v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
		auto Result = Encoder.getMetadata(v4l2_buf->index, enc_metadata);
		if ( Result == 0 )
		{
				cout << "Frame " << frame_num <<
				": isKeyFrame=" << (int) enc_metadata.KeyFrame <<
				" AvgQP=" << enc_metadata.AvgQP <<
				" MinQP=" << enc_metadata.FrameMinQP <<
				" MaxQP=" << enc_metadata.FrameMaxQP <<
				" EncodedBits=" << enc_metadata.EncodedFrameBits <<
				endl;
			}
		}
	}
	*/

	//	Get motion vector parameters of the frames from encoder
	/*
	{
		v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata;
		if (ctx->enc->getMotionVectors(v4l2_buf->index, enc_mv_metadata) == 0)
		{
			uint32_t numMVs = enc_mv_metadata.bufSize / sizeof(MVInfo);
			MVInfo *pInfo = enc_mv_metadata.pMVInfo;
			
			cout << "Frame " << frame_num << ": Num MVs=" << numMVs << endl;
			
			for (uint32_t i = 0; i < numMVs; i++, pInfo++)
			{
				cout << i << ": mv_x=" << pInfo->mv_x <<
				" mv_y=" << pInfo->mv_y <<
				" weight=" << pInfo->weight <<
				endl;
			}
		}
	}
	*//*
	
	//	put the output/h264 buffer back so it can be used
	auto Result = Encoder.capture_plane.qBuffer(*v4l2_buf, nullptr);
	IsOkay(Result,"Post-new-buffer re-enqueuing buffer for encoder");
}
*/



void Nvidia::TEncoder::Start()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	auto& Encoder = *mEncoder;
	auto UseCommand = true;
	
	if ( UseCommand )
	{
		//	Send v4l2 command for encoder start
		auto Result = Encoder.setEncoderCommand(V4L2_ENC_CMD_START, 0);
		IsOkay(Result,"Failed to send V4L2_ENC_CMD_START");
	}
	else
	{
		//	start streaming
		auto Result = Encoder.output_plane.setStreamStatus(true);
		IsOkay(Result,"output_plane enable streaming");
		
		Result = Encoder.capture_plane.setStreamStatus(true);
		IsOkay(Result,"capture_plane enable streaming");
	}

}


void Nvidia::TEncoder::Sync()
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	auto& mMemoryMode = mNative.mMemoryMode;
	/*
	if(mMemoryMode == V4L2_MEMORY_DMABUF || mMemoryMode == V4L2_MEMORY_MMAP)
	{
		for (uint32_t j = 0 ; j < buffer->n_planes; j++)
		{
			ret = NvBufferMemSyncForDevice (buffer->planes[j].fd, j, (void **)&buffer->planes[j].data);
			if (ret < 0)
			{
				cerr << "Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
				abort(&ctx);
				goto cleanup;
			}
		}
	}
	
	if(mMemoryMode == V4L2_MEMORY_DMABUF)
	{
		for (uint32_t j = 0 ; j < buffer->n_planes ; j++)
		{
			v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
		}
	}
	 */
}

void Nvidia::TEncoder::ReadNextFrame()
{
	auto& Encoder = *mEncoder;
	
}


void Nvidia::TEncoder::WaitForEnd()
{
	auto& Encoder = *mEncoder;
	/*
	//	if blocking mode
	//	Wait till capture plane DQ Thread finishes
	//	i.e. all the capture plane buffers are dequeued.
	bool EndOfStream = false;
	auto Result = encoder_proc_blocking(ctx, EndOfStream);
	IsOkay(Result,"encoder_proc_blocking");
	Encoder.capture_plane.waitForDQThread(-1);
	 */
}

/*
int
read_video_frame(std::ifstream * stream, NvBuffer & buffer)
{
	uint32_t i, j;
	char *data;
	
	for (i = 0; i < buffer.n_planes; i++)
	{
		NvBuffer::NvBufferPlane &plane = buffer.planes[i];
		std::streamsize bytes_to_read =
		plane.fmt.bytesperpixel * plane.fmt.width;
		data = (char *) plane.data;
		plane.bytesused = 0;
		for (j = 0; j < plane.fmt.height; j++)
		{
			stream->read(data, bytes_to_read);
			if (stream->gcount() < bytes_to_read)
				return -1;
			data += plane.fmt.stride;
		}
		plane.bytesused = plane.fmt.stride * plane.fmt.height;
	}
	return 0;
}

int
write_video_frame(std::ofstream * stream, NvBuffer &buffer)
{
	uint32_t i, j;
	char *data;
	
	for (i = 0; i < buffer.n_planes; i++)
	{
		NvBuffer::NvBufferPlane &plane = buffer.planes[i];
		size_t bytes_to_write =
		plane.fmt.bytesperpixel * plane.fmt.width;
		
		data = (char *) plane.data;
		for (j = 0; j < plane.fmt.height; j++)
		{
			stream->write(data, bytes_to_write);
			if (!stream->good())
				return -1;
			data += plane.fmt.stride;
		}
	}
	return 0;
}
*/
void Nvidia::TEncoder::Shutdown()
{
/*
	if(ctx.b_use_enc_cmd)
	{
		//	Send v4l2 command for encoder stop
		ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
		eos = true;
		break;
	}
	else
	{
		eos = true;
		v4l2_buf.m.planes[0].m.userptr = 0;
		v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
	}
 */
}


void Nvidia::TEncoder::FinishEncoding()
{
	
}

void Nvidia::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, const std::string& Meta, bool Keyframe)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	Soy_AssertTodo();
}

void Nvidia::TEncoder::Encode(const SoyPixelsImpl& Pixels,const std::string& Meta,bool Keyframe)
{
	std::Debug << __PRETTY_FUNCTION__ << std::endl;
	InitEncoder( Pixels.GetMeta() );
	
	auto& Encoder = *mEncoder;
	auto& YuvPlane = GetYuvPlane();

	//	gr: get a valid DEQUEUED buffer, so we should have a list from the encoder
	//		of dequeued input planes
	int BufferIndex = 0;
	
	//	pick a buffer
	//for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++)
	struct v4l2_buffer v4l2_buf;
	struct v4l2_plane planes[MAX_PLANES];
	NvBuffer* pBuffer = YuvPlane.getNthBuffer(BufferIndex);
	if ( !pBuffer )
		throw Soy::AssertException("GetNthBuffer null");
	
	memset(&v4l2_buf, 0, sizeof(v4l2_buf));
	memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
	
	v4l2_buf.index = BufferIndex;
	v4l2_buf.m.planes = planes;
	
	auto& mMemoryMode = mNative.mMemoryMode;
	
	if ( mMemoryMode == V4L2_MEMORY_DMABUF)
	{
		throw Soy::AssertException("todo: DMA map file mode");
		/*
		v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
		v4l2_buf.memory = V4L2_MEMORY_DMABUF;
		//	Map output plane buffer for memory type DMABUF.
		auto Result = YuvPlane.mapOutputBuffers(v4l2_buf, ctx.output_plane_fd[i]);
		IsOkay(Result,"Error while mapping buffer at output plane");
		*/
	}
	
	//	fill buffer with yuv
	auto& Buffer = *pBuffer;
	for ( auto p=0;	p<Buffer.n_planes; p++)
	{
		NvBuffer::NvBufferPlane& Plane = Buffer.planes[p];
		std::streamsize bytes_to_read = Plane.fmt.bytesperpixel * Plane.fmt.width;
		throw Soy::AssertException("Todo: fill yuv plane buffer");
		/*
		data = (char *) plane.data;
		plane.bytesused = 0;
		for (j = 0; j < plane.fmt.height; j++)
		{
			stream->read(data, bytes_to_read);
			if (stream->gcount() < bytes_to_read)
				return -1;
			data += plane.fmt.stride;
		}
		plane.bytesused = plane.fmt.stride * plane.fmt.height;
		*/
	}
	
	Sync();
	
	//	queue buffer for "output plane" (input)
	auto Result = YuvPlane.qBuffer(v4l2_buf, nullptr);
	IsOkay(Result,"Error while queueing buffer at output plane");
}


