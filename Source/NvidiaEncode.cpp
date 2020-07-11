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
#endif

//#include "nvidia/samples/01_video_encode/video_encode.h"
#include "SoyPixels.h"
//#include <linux/videodev2.h>
#include "Linux/include/linux/videodev2.h"
#include "NvidiaEncode.h"


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


auto x = V4L2_PIX_FMT_YUV420M;

namespace V4lPixelFormat
{
	enum Type : uint32_t
	{
		YUV420M = x,
	};
}

v4l2_memory x;
namespace V4lMemoryMode
{
	enum Type : uint32_t
	{
		DMABUF = V4L2_MEMORY_DMABUF,
	};
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



Nvidia::TEncoder::TEncoder(TEncoderParams& Params,std::function<void(PopH264::TPacket&)> OnOutPacket)
{
	//	the nvvideoencoder is a wrapper for a V4L2 device
	//	which is a standard linux file i/o stream
	//	alloc an encoder in blocking mode
	//	so the non-blocking mode is an IO mode
	//	gr: what is enc0 ? device name?
	//ctx.enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);
	mEncoder = NvVideoEncoder::createVideoEncoder("enc0")
	if ( !mEncoder )
		throw Soy::AssertException("Failed to allocate nvidia encoder");
	
	
	auto& Encoder = *mEncoder;
	auto Result = Encoder.subscribeEvent(V4L2_EVENT_EOS,0,0);
	IsOkay(Result,"Failed to subscribe to EOS event");
	

}


void Nvidia::TEncoder::InitInputFormat()
{
	//	gr: nvidia demo calls this the output plane
	
	auto& Encoder = *mEncoder;
	//	the input is a "capture" plane
	SoyPixelsMeta PixelMeta(100,100,SoyPixelsFormat::Yuv_844);
	auto PixelFormat = GetPixelFormat( PixelMeta.GetFormat() );
	auto Format = V4L2_PIX_FMT_YUV444M;
	auto MaxSize = InputFormat.GetDataSize();
	auto Result = Encoder.setOutputPlaneFormat(Format, Width, Height, MaxSize );
	IsOkay(Result,"InitInpuFormat failed setOutputPlaneFormat");

	//	setup memory read mode
	ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
	ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
	ret = setup_output_dmabuf(&ctx,10);
	
}

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
			*/
		}
	}
	
	//	end of string
	if (buffer->planes[0].bytesused == 0)
	{
		std::Debug << "Got 0 size buffer in capture" << std::endl;
		return;
	}
	
	//	Computing CRC with each frame
	/*
	if(ctx->pBitStreamCrc)
		CalculateCrc (ctx->pBitStreamCrc, buffer->planes[0].data, buffer->planes[0].bytesused);
	*/

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
	*/
	
	//	put the output/h264 buffer back so it can be used
	auto Result = Encoder.capture_plane.qBuffer(*v4l2_buf, nullptr);
	IsOkay(Result,"Post-new-buffer re-enqueuing buffer for encoder");
}



void Nvidia::TEncoder::InitOutputFormat()
{
	//	nvidia demo calls this the capture plane
	auto& Encoder = *mEncoder;
	auto Profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE;
	auto Format = V4L2_PIX_FMT_H264;
	auto Width = 100;
	auto Height = 100;
	auto BufferSize = 1024 * 1024 * 2;
	auto Result = Encoder.setCapturePlaneFormat( OutputFormat, Width, Height, BufferSize );
	IsOkay(Result,"InitOutputFormat failed setCapturePlaneFormat");

	//	set other params
	Result = Encoder.setBitrate(ctx.bitrate);
	IsOkay(Result,"Failed to set bitrate");

	setProfile
	auto Level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
	Result = Encoder.setLevel(Level);
	IsOkay(Result,"Failed to set level");

	
	//	setup memory mode
	switch(ctx.output_memory_type)
	{
		case V4L2_MEMORY_MMAP:
			ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
			TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
			break;
			
		case V4L2_MEMORY_USERPTR:
			ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
			TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
			break;
			
		case V4L2_MEMORY_DMABUF:
			ret = setup_output_dmabuf(&ctx,10);
			TEST_ERROR(ret < 0, "Could not setup plane", cleanup);
			break;
		default :
			TEST_ERROR(true, "Not a valid plane", cleanup);
	}
	
	
	//	CFunc callback
	//	encoder_capture_plane_dq_callback
	auto EncoderCallback = [](struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
									  NvBuffer * shared_buffer, void * This)
	{
		This->OnFrameEncoded( v4l2_buf, buffer, shared_buffer );
	};
	
	//	setup capture_plane/h264 callback
	Encoder.capture_plane.setDQThreadCallback(EncoderCallback);
	
	/* startDQThread starts a thread internally which calls the
	 encoder_capture_plane_dq_callback whenever a buffer is dequeued
	 on the plane */
	auto* This = this;
	ctx.enc->capture_plane.startDQThread(This);
	
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

}

void Nvidia::TEncoder::Start()
{
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

void Nvidia::TEncoder::EncodeFrame()
{
	auto& Encoder = *mEncoder;

	int BufferIndex = 0;
	
	//	pick a buffer
	//for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++)
	struct v4l2_buffer v4l2_buf;
	struct v4l2_plane planes[MAX_PLANES];
	NvBuffer *buffer = Encoder.output_plane.getNthBuffer(BufferIndex);
	
	memset(&v4l2_buf, 0, sizeof(v4l2_buf));
	memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
	
	v4l2_buf.index = BufferIndex;
	v4l2_buf.m.planes = planes;

	if ( mMemoryMode == V4L2_MEMORY_DMABUF)
	{
		v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
		v4l2_buf.memory = V4L2_MEMORY_DMABUF;
		//	Map output plane buffer for memory type DMABUF.
		auto Result = Encoder.output_plane.mapOutputBuffers(v4l2_buf, ctx.output_plane_fd[i]);
		IsOkay(Result,"Error while mapping buffer at output plane");
	}
	
	//	fill buffer with yuv
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
	
	Sync();
	
	//	queue buffer for "output plane" (input)
	Result = Encoder.output_plane.qBuffer(v4l2_buf, nullptr);
	IsOkay(Result,"Error while queueing buffer at output plane");
}

void Nvidia::TEncoder::Sync()
{
	if(ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP)
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
	
	if(ctx.output_memory_type == V4L2_MEMORY_DMABUF)
	{
		for (uint32_t j = 0 ; j < buffer->n_planes ; j++)
		{
			v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
		}
	}
}

void NVidia::TEncoder::ReadNextFrame()
{
	auto& Encoder = *mEncoder;
	
}


void NVidia::TEncoder::WaitForEnd()
{
	auto& Encoder = *mEncoder;
	
	//	if blocking mode
	//	Wait till capture plane DQ Thread finishes
	//	i.e. all the capture plane buffers are dequeued.
	bool EndOfStream = false;
	auto Result = encoder_proc_blocking(ctx, EndOfStream);
	IsOkay(Result,"encoder_proc_blocking");
	Encoder.capture_plane.waitForDQThread(-1);
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
}

	
/**
  * Set encoder context defaults values.
  *
  * @param ctx : Encoder context
  */
static void
set_defaults(context_t * ctx)
{
	// tsdk: These are carried over from the nvidia example
	memset(ctx, 0, sizeof(context_t));

	ctx->raw_pixfmt = V4L2_PIX_FMT_YUV420M;
	ctx->bitrate = 4 * 1024 * 1024;
	ctx->peak_bitrate = 0;
	ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
	ctx->ratecontrol = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
	ctx->iframe_interval = 30;
	ctx->externalRPS = false;
	ctx->enableGDR = false;
	ctx->enableROI = false;
	ctx->bnoIframe = false;
	ctx->bGapsInFrameNumAllowed = false;
	ctx->bReconCrc = false;
	ctx->enableLossless = false;
	ctx->nH264FrameNumBits = 0;
	ctx->nH265PocLsbBits = 0;
	ctx->idr_interval = 256;
	ctx->level = -1;
	ctx->fps_n = 30;
	ctx->fps_d = 1;
	ctx->gdr_start_frame_number = 0xffffffff;
	ctx->gdr_num_frames = 0xffffffff;
	ctx->gdr_out_frame_number = 0xffffffff;
	ctx->num_b_frames = (uint32_t) -1;
	ctx->nMinQpI = (uint32_t)QP_RETAIN_VAL;
	ctx->nMaxQpI = (uint32_t)QP_RETAIN_VAL;
	ctx->nMinQpP = (uint32_t)QP_RETAIN_VAL;
	ctx->nMaxQpP = (uint32_t)QP_RETAIN_VAL;
	ctx->nMinQpB = (uint32_t)QP_RETAIN_VAL;
	ctx->nMaxQpB = (uint32_t)QP_RETAIN_VAL;
	ctx->use_gold_crc = false;
	ctx->pBitStreamCrc = NULL;
	ctx->externalRCHints = false;
	ctx->input_metadata = false;
	ctx->sMaxQp = 51;
	ctx->stats = false;
	ctx->stress_test = 1;
	ctx->output_memory_type = V4L2_MEMORY_DMABUF;
	ctx->copy_timestamp = false;
	ctx->start_ts = 0;
	ctx->max_perf = 0;
	ctx->blocking_mode = 1;
	ctx->startf = 0;
	ctx->endf = 0;
	ctx->num_output_buffers = 6;
	ctx->num_frames_to_encode = -1;
}

// Initialisation
// Create Video Encoder
void Nvidia::TEncoder::AllocEncoder(const SoyPixelsMeta& Meta)
{
	// Create encoder context.
	context_t ctx;

}

// Reference encode_proc in the nvidia_encode_main example
void Nvidia::TEncoder::Encode(const SoyPixelsImpl& Luma, const SoyPixelsImpl& ChromaU, const SoyPixelsImpl& ChromaV, context_t& ctx)
{
	// Create a var that holds a return value that is written over
	int ret = 0;
	// Any Error codes
	int error = 0;
	// End Of Stream, this runs on a loop until this becomes true
	bool eos = false;

	/* Set default values for encoder context members. */
	set_defaults(&ctx);

	/* Set thread name for encoder Output Plane thread. */
	pthread_setname_np(pthread_self(),"EncOutPlane");

	if (ctx.endf) {
		IsOkay(ctx.startf > ctx.endf, "End frame should be greater than start frame");
		ctx.num_frames_to_encode = ctx.endf - ctx.startf + 1;
	}

	/* Open input file for raw yuv, where does this file come from?*/
	ctx.in_file = new std::ifstream(ctx.in_file_path);
	IsOkay(!ctx.in_file->is_open(), "Could not open input file");

		if (!ctx.stats)
	{
		/* Open output file for encoded bitstream */
		ctx.out_file = new std::ofstream(ctx.out_file_path);
		IsOkay(!ctx.out_file->is_open(), "Could not open output file");
	}

	// Create NvVideoEncoder object in blocking mode
	ctx.enc = NvVideoEncoder::createVideoEncoder("enc0");

	IsOkay(!ctx.enc, "Could not create encoder");

/* Set encoder capture plane format.
NOTE: It is necessary that Capture Plane format be set before Output Plane
format. It is necessary to set width and height on the capture plane as well */
	ret = ctx.enc->setCapturePlaneFormat(V4L2_PIX_FMT_H264, ctx.width,
									  ctx.height, 2 * 1024 * 1024);
	IsOkay(ret < 0, "Could not set capture plane format");
}

// tsdk: V4L2 is coming in here 
int NvVideoEncoder::setCapturePlaneFormat(uint32_t pixfmt, uint32_t width, uint32_t height, uint32_t sizeimage)
{
	struct v4l2_format format;

	memset(&format, 0, sizeof(struct v4l2_format));
	format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	switch (pixfmt)
	{
		case V4L2_PIX_FMT_H264:
		case V4L2_PIX_FMT_H265:
		case V4L2_PIX_FMT_VP8:
		case V4L2_PIX_FMT_VP9:
			capture_plane_pixfmt = pixfmt;
			break;
		default:
			ERROR_MSG("Unknown supported pixel format for encoder " << pixfmt);
			return -1;
	}

	format.fmt.pix_mp.pixelformat = pixfmt;
	format.fmt.pix_mp.width = width;
	format.fmt.pix_mp.height = height;
	format.fmt.pix_mp.num_planes = 1;
	format.fmt.pix_mp.plane_fmt[0].sizeimage = sizeimage;

	return capture_plane.setFormat(format);
}
