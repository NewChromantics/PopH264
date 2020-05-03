#include "SoyAvf.h"
#include "SoyH264.h"
#include "SoyFilesystem.h"
#include "SoyFourcc.h"

#include <CoreMedia/CMBase.h>
#include <VideoToolbox/VTBase.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>
#include <CoreMedia/CMSampleBuffer.h>
#include <CoreMedia/CMFormatDescription.h>
#include <CoreMedia/CMTime.h>
#include <VideoToolbox/VTSession.h>
#include <VideoToolbox/VTCompressionProperties.h>
#include <VideoToolbox/VTCompressionSession.h>
#include <VideoToolbox/VTDecompressionSession.h>
#include <VideoToolbox/VTErrors.h>




#define CV_VIDEO_TYPE_META(Enum,SoyFormat)	TCvVideoTypeMeta( Enum, #Enum, SoyFormat )
#define CV_VIDEO_INVALID_ENUM		0
class TCvVideoTypeMeta
{
public:
	TCvVideoTypeMeta(OSType Enum,const char* EnumName,SoyPixelsFormat::Type SoyFormat) :
	mPlatformFormat		( Enum ),
	mName				( EnumName ),
	mSoyFormat			( SoyFormat )
	{
		if ( !IsValid() )
			throw Soy::AssertException("Expected valid enum - or invalid enum is bad" );
	}
	TCvVideoTypeMeta() :
	mPlatformFormat		( CV_VIDEO_INVALID_ENUM ),
	mName				( "Invalid enum" ),
	mSoyFormat			( SoyPixelsFormat::Invalid )
	{
	}
	
	bool		IsValid() const		{	return mPlatformFormat != CV_VIDEO_INVALID_ENUM;	}
	
	bool		operator==(const OSType& Enum) const					{	return mPlatformFormat == Enum;	}
	bool		operator==(const SoyPixelsFormat::Type& Format) const	{	return mSoyFormat == Format;	}
	
public:
	OSType					mPlatformFormat;
	SoyPixelsFormat::Type	mSoyFormat;
	std::string				mName;
};


static TCvVideoTypeMeta Cv_PixelFormatMap[] =
{
	/*
	 //	from avfDecoder ResolveFormat(id)
	 //	gr: RGBA never seems to decode, but with no error[on osx]
	 case SoyPixelsFormat::RGBA:
	 //	BGRA is internal format on IOS so return that as default
	 case SoyPixelsFormat::BGRA:
	 default:
	 */
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_OneComponent8,	SoyPixelsFormat::Luma_Full ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_OneComponent8,	SoyPixelsFormat::Luma_Ntsc ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_OneComponent8,	SoyPixelsFormat::Luma_Smptec ),
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_24RGB,	SoyPixelsFormat::RGB ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_24BGR,	SoyPixelsFormat::BGR ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_32BGRA,	SoyPixelsFormat::BGRA ),
	
	//	gr: PopFace creating a pixel buffer from a unity "argb" texture, failed as RGBA is unsupported...
	//	gr: ARGB is accepted, but channels are wrong
	CV_VIDEO_TYPE_META( kCVPixelFormatType_32RGBA,	SoyPixelsFormat::RGBA ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_32ARGB,	SoyPixelsFormat::ARGB ),
	
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,	SoyPixelsFormat::Yuv_8_88_Full ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,	SoyPixelsFormat::Yuv_8_88_Ntsc ),
	
	//	gr: this is CHROMA|YUV! not YUV, this is why the fourcc is 2vuy
	//		kCVPixelFormatType_422YpCbCr8     = '2vuy',     /* Component Y'CbCr 8-bit 4:2:2, ordered Cb Y'0 Cr Y'1 */
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr8,	SoyPixelsFormat::Uvy_844_Full ),
	
	//	todo: check these
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr8FullRange,	SoyPixelsFormat::Yuv_844_Full ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr8_yuvs,	SoyPixelsFormat::Yuv_844_Ntsc ),
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_420YpCbCr8Planar,	SoyPixelsFormat::YYuv_8888_Ntsc ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_420YpCbCr8PlanarFullRange,	SoyPixelsFormat::YYuv_8888_Full ),
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_1Monochrome,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_2Indexed,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_4Indexed,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_8Indexed,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_1IndexedGray_WhiteIsZero,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_2IndexedGray_WhiteIsZero,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_4IndexedGray_WhiteIsZero,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_8IndexedGray_WhiteIsZero,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16BE555,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16LE555,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16LE5551,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16BE565,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16LE565,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_32ABGR,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_64ARGB,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_48RGB,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_32AlphaGray,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_16Gray,	SoyPixelsFormat::Invalid ),
	
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_4444YpCbCrA8,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_4444YpCbCrA8R,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_444YpCbCr8,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr16,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr10,	SoyPixelsFormat::Invalid ),
	CV_VIDEO_TYPE_META( kCVPixelFormatType_444YpCbCr10,	SoyPixelsFormat::Invalid ),
	
	CV_VIDEO_TYPE_META( kCVPixelFormatType_422YpCbCr_4A_8BiPlanar,	SoyPixelsFormat::Invalid ),
	
	//	the logitech C22 has this format, which apparently might be a kind of motion jpeg
	CV_VIDEO_TYPE_META( 'dmb1',	SoyPixelsFormat::Invalid ),
};


std::ostream& operator<<(std::ostream& out,const AVAssetExportSessionStatus& in)
{
	switch ( in )
	{
		case AVAssetExportSessionStatusUnknown:		out << "AVAssetExportSessionStatusUnknown";	 return out;
		case AVAssetExportSessionStatusWaiting:		out << "AVAssetExportSessionStatusWaiting";	 return out;
		case AVAssetExportSessionStatusExporting:	out << "AVAssetExportSessionStatusExporting";	 return out;
		case AVAssetExportSessionStatusCompleted:	out << "AVAssetExportSessionStatusCompleted";	 return out;
		case AVAssetExportSessionStatusFailed:		out << "AVAssetExportSessionStatusFailed";	 return out;
		case AVAssetExportSessionStatusCancelled:	out << "AVAssetExportSessionStatusCancelled";	 return out;
		default:
		{
			out << "AVAssetExportSessionStatus<" << static_cast<int>( in ) << ">";
			return out;
		}
	}
	
}



OSType Avf::GetPlatformPixelFormat(SoyPixelsFormat::Type Format)
{
	auto Table = GetRemoteArray( Cv_PixelFormatMap );
	auto* Meta = GetArrayBridge(Table).Find( Format );
	
	if ( !Meta )
		return CV_VIDEO_INVALID_ENUM;
	
	return Meta->mPlatformFormat;
}

void Avf::GetFormatDescriptionData(ArrayBridge<uint8>&& Data,CMFormatDescriptionRef FormatDesc,size_t ParamIndex)
{
	size_t ParamCount = 0;
	auto Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( FormatDesc, 0, nullptr, nullptr, &ParamCount, nullptr );
	Avf::IsOkay( Result, "Get H264 param 0");
	
	/*
	 //	known bug on ios?
	 if (status ==
		CoreMediaGlue::kCMFormatDescriptionBridgeError_InvalidParameter) {
		DLOG(WARNING) << " assuming 2 parameter sets and 4 bytes NAL length header";
		pset_count = 2;
		nal_size_field_bytes = 4;
	 */
	
	if ( ParamIndex >= ParamCount )
		throw Soy::AssertException("SPS missing");
	
	const uint8_t* ParamsData = nullptr;;
	size_t ParamsSize = 0;
	Result = CMVideoFormatDescriptionGetH264ParameterSetAtIndex( FormatDesc, ParamIndex, &ParamsData, &ParamsSize, nullptr, nullptr);
	
	Avf::IsOkay( Result, "Failed to get H264 param X" );
	
	Data.PushBackArray( GetRemoteArray( ParamsData, ParamsSize ) );
}



//	lots of errors in macerrors.h with no string conversion :/
#define TESTENUMERROR(e,Enum)	if ( (e) == (Enum) )	return #Enum ;


//	http://stackoverflow.com/questions/2196869/how-do-you-convert-an-iphone-osstatus-code-to-something-useful
std::string Avf::GetString(OSStatus Status)
{
	TESTENUMERROR(Status,kVTPropertyNotSupportedErr);
	TESTENUMERROR(Status,kVTPropertyReadOnlyErr);
	TESTENUMERROR(Status,kVTParameterErr);
	TESTENUMERROR(Status,kVTInvalidSessionErr);
	TESTENUMERROR(Status,kVTAllocationFailedErr);
	TESTENUMERROR(Status,kVTPixelTransferNotSupportedErr);
	TESTENUMERROR(Status,kVTCouldNotFindVideoDecoderErr);
	TESTENUMERROR(Status,kVTCouldNotCreateInstanceErr);
	TESTENUMERROR(Status,kVTCouldNotFindVideoEncoderErr);
	TESTENUMERROR(Status,kVTVideoDecoderBadDataErr);
	TESTENUMERROR(Status,kVTVideoDecoderUnsupportedDataFormatErr);
	TESTENUMERROR(Status,kVTVideoDecoderMalfunctionErr);
	TESTENUMERROR(Status,kVTVideoEncoderMalfunctionErr);
	TESTENUMERROR(Status,kVTVideoDecoderNotAvailableNowErr);
	TESTENUMERROR(Status,kVTImageRotationNotSupportedErr);
	TESTENUMERROR(Status,kVTVideoEncoderNotAvailableNowErr);
	TESTENUMERROR(Status,kVTFormatDescriptionChangeNotSupportedErr);
	TESTENUMERROR(Status,kVTInsufficientSourceColorDataErr);
	TESTENUMERROR(Status,kVTCouldNotCreateColorCorrectionDataErr);
	TESTENUMERROR(Status,kVTColorSyncTransformConvertFailedErr);
	TESTENUMERROR(Status,kVTVideoDecoderAuthorizationErr);
	TESTENUMERROR(Status,kVTVideoEncoderAuthorizationErr);
	TESTENUMERROR(Status,kVTColorCorrectionPixelTransferFailedErr);
	TESTENUMERROR(Status,kVTMultiPassStorageIdentifierMismatchErr);
	TESTENUMERROR(Status,kVTMultiPassStorageInvalidErr);
	TESTENUMERROR(Status,kVTFrameSiloInvalidTimeStampErr);
	TESTENUMERROR(Status,kVTFrameSiloInvalidTimeRangeErr);
	TESTENUMERROR(Status,kVTCouldNotFindTemporalFilterErr);
	TESTENUMERROR(Status,kVTPixelTransferNotPermittedErr);
	
	TESTENUMERROR(Status,kCVReturnInvalidArgument);
	
	TESTENUMERROR(Status,kCMBlockBufferStructureAllocationFailedErr);
	TESTENUMERROR(Status,kCMBlockBufferBlockAllocationFailedErr);
	TESTENUMERROR(Status,kCMBlockBufferBadCustomBlockSourceErr);
	TESTENUMERROR(Status,kCMBlockBufferBadOffsetParameterErr);
	TESTENUMERROR(Status,kCMBlockBufferBadLengthParameterErr);
	TESTENUMERROR(Status,kCMBlockBufferBadPointerParameterErr);
	TESTENUMERROR(Status,kCMBlockBufferEmptyBBufErr);
	TESTENUMERROR(Status,kCMBlockBufferUnallocatedBlockErr);
	TESTENUMERROR(Status,kCMBlockBufferInsufficientSpaceErr);
	
	TESTENUMERROR(Status,kCVReturnInvalidArgument);
	TESTENUMERROR(Status,kCVReturnAllocationFailed);
#if defined(AVAILABLE_MAC_OS_X_VERSION_10_11_AND_LATER)
	TESTENUMERROR(Status,kCVReturnUnsupported);
#endif
	TESTENUMERROR(Status,kCVReturnInvalidDisplay);
	TESTENUMERROR(Status,kCVReturnDisplayLinkAlreadyRunning);
	TESTENUMERROR(Status,kCVReturnDisplayLinkNotRunning);
	TESTENUMERROR(Status,kCVReturnDisplayLinkCallbacksNotSet);
	TESTENUMERROR(Status,kCVReturnInvalidPixelFormat);
	TESTENUMERROR(Status,kCVReturnInvalidSize);
	TESTENUMERROR(Status,kCVReturnInvalidPixelBufferAttributes);
	TESTENUMERROR(Status,kCVReturnPixelBufferNotOpenGLCompatible);
	TESTENUMERROR(Status,kCVReturnPixelBufferNotMetalCompatible);
	TESTENUMERROR(Status,kCVReturnWouldExceedAllocationThreshold);
	TESTENUMERROR(Status,kCVReturnPoolAllocationFailed);
	TESTENUMERROR(Status,kCVReturnInvalidPoolAttributes);
	
	//	decompression gives us this
	TESTENUMERROR(Status,MACH_RCV_TIMED_OUT);
	
	switch ( static_cast<sint32>(Status) )
	{
		case -8961:	return "kVTPixelTransferNotSupportedErr -8961";
		case -8969:	return "kVTVideoDecoderBadDataErr -8969";
		case -8970:	return "kVTVideoDecoderUnsupportedDataFormatErr -8970";
		case -8960:	return "kVTVideoDecoderMalfunctionErr -8960";
		default:
			break;
	}
	
	//	gr: can't find any documentation on this value.
	if ( static_cast<sint32>(Status) == -12349 )
		return "Unknown VTDecodeFrame error -12349";
	
	//	as integer..
	std::stringstream Error;
	Error << "OSStatus = " << static_cast<sint32>(Status);
	return Error.str();
	
	//	could be fourcc?
	Soy::TFourcc Fourcc( CFSwapInt32HostToBig(Status) );
	return Fourcc.GetString();
	/*
	 //	example with specific bundles...
	 NSBundle *bundle = [NSBundle bundleWithIdentifier:@"com.apple.security"];
	 NSString *key = [NSString stringWithFormat:@"%d", (int)Status];
	 auto* StringNs = [bundle localizedStringForKey:key value:key table:@"SecErrorMessages"];
	 return Soy::NSStringToString( StringNs );
	 */
}
/*
 std::shared_ptr<AvfCompressor::TInstance> AvfCompressor::Allocate(const TCasterParams& Params)
 {
	return std::shared_ptr<AvfCompressor::TInstance>( new AvfCompressor::TInstance(Params) );
 }
 */

void Avf::IsOkay(OSStatus Error,const std::string& Context)
{
	IsOkay( Error, Context.c_str() );
}

void Avf::IsOkay(OSStatus Error,const char* Context)
{
	//	kCVReturnSuccess
	if ( Error == noErr )
		return;
	
	std::stringstream ErrorString;
	ErrorString << "OSStatus/CVReturn error in " << Context << ": " << GetString(Error);
	
	throw Soy::AssertException( ErrorString.str() );
}



SoyPixelsFormat::Type Avf::GetPixelFormat(OSType Format)
{
	auto Table = GetRemoteArray( Cv_PixelFormatMap );
	auto* Meta = GetArrayBridge(Table).Find( Format );
	
	if ( !Meta )
	{
		Soy::TFourcc Fourcc(Format);
		std::Debug << "Unknown Avf CV pixel format (" << Fourcc << " 0x" << std::hex << Format << ")" << std::dec << std::endl;
		
		return SoyPixelsFormat::Invalid;
	}
	return Meta->mSoyFormat;
}


SoyPixelsFormat::Type Avf::GetPixelFormat(NSNumber* Format)
{
	auto FormatInt = [Format integerValue];
	return GetPixelFormat( static_cast<OSType>(FormatInt) );
}


void PixelReleaseCallback(void *releaseRefCon, const void *baseAddress)
{
	//std::Debug << __func__ << std::endl;
	
	//	this page says we need to release
	//	http://codefromabove.com/2015/01/av-foundation-saving-a-sequence-of-raw-rgb-frames-to-a-movie/
	if ( releaseRefCon != nullptr )
	{
		CFDataRef bufferData = (CFDataRef)releaseRefCon;
		CFRelease(bufferData);
	}
}

CVPixelBufferRef Avf::PixelsToPixelBuffer(const SoyPixelsImpl& Image)
{
	CFAllocatorRef PixelBufferAllocator = nullptr;
	OSType PixelFormatType = GetPlatformPixelFormat( Image.GetFormat() );
	auto& PixelsArray = Image.GetPixelsArray();
	auto* Pixels = const_cast<uint8*>( PixelsArray.GetArray() );
	auto BytesPerRow = Image.GetMeta().GetRowDataSize();
	void* ReleaseContext = nullptr;
	CFDictionaryRef PixelBufferAttributes = nullptr;
	
#if defined(TARGET_OSX)
	//	gr: hack, cannot create RGBA pixel buffer on OSX. do a last-min conversion here, but ideally it's done beforehand
	//		REALLY ideally we can go from texture to CVPixelBuffer
	if ( Image.GetFormat() == SoyPixelsFormat::RGBA && PixelFormatType == kCVPixelFormatType_32RGBA )
	{
		//std::Debug << "CVPixelBufferCreateWithBytes will fail with RGBA, forcing BGRA" << std::endl;
		PixelFormatType = kCVPixelFormatType_32BGRA;
	}
#endif
	
	CVPixelBufferRef PixelBuffer = nullptr;
	auto Result = CVPixelBufferCreateWithBytes( PixelBufferAllocator, Image.GetWidth(), Image.GetHeight(), PixelFormatType, Pixels, BytesPerRow, PixelReleaseCallback, ReleaseContext, PixelBufferAttributes, &PixelBuffer );
	
	Soy::TFourcc Fourcc(PixelFormatType);
	std::stringstream Error;
	Error << "CVPixelBufferCreateWithBytes " << Image.GetMeta() << "(" << Fourcc << ")";
	Avf::IsOkay( Result, Error.str() );
	
	return PixelBuffer;
}

CFPtr<CMFormatDescriptionRef> Avf::GetFormatDescriptionH264(const ArrayBridge<uint8_t>& Sps,const ArrayBridge<uint8_t>& Pps,H264::NaluPrefix::Type NaluPrefixType)
{
	CFAllocatorRef Allocator = nil;
	
	//	need to strip nalu prefix from these
	auto SpsPrefixLength = H264::GetNaluLength(Sps);
	auto PpsPrefixLength = H264::GetNaluLength(Pps);

	auto* SpsStart = &Sps[SpsPrefixLength];
	auto* PpsStart = &Pps[PpsPrefixLength];

	BufferArray<const uint8_t*,2> Params;
	BufferArray<size_t,2> ParamSizes;
	Params.PushBack( SpsStart );
	ParamSizes.PushBack( Sps.GetDataSize() );
	Params.PushBack( PpsStart );
	ParamSizes.PushBack( Pps.GetDataSize() );
	
	//	ios doesnt support annexb, so we will have to convert inputs
	//	lets use 32 bit nalu size prefix
	if ( NaluPrefixType == H264::NaluPrefix::AnnexB )
		NaluPrefixType = H264::NaluPrefix::ThirtyTwo;
	auto NaluLength = static_cast<int>(NaluPrefixType);

	CFPtr<CMFormatDescriptionRef> FormatDesc;
	//	-12712 http://stackoverflow.com/questions/25078364/cmvideoformatdescriptioncreatefromh264parametersets-issues
	auto Result = CMVideoFormatDescriptionCreateFromH264ParameterSets( Allocator, Params.GetSize(), Params.GetArray(), ParamSizes.GetArray(), NaluLength, &FormatDesc.mObject );
	Avf::IsOkay( Result, "CMVideoFormatDescriptionCreateFromH264ParameterSets" );

	return FormatDesc;
}
