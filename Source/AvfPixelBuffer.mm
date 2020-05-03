#import "AvfPixelBuffer.h"
#include <SoyOpengl.h>
#include <SoyAvf.h>

#if defined(ENABLE_METAL)
#include <SoyMetal.h>
#endif


std::string Platform::GetExtensions(CMFormatDescriptionRef FormatDescription)
{
	auto Extensions = (NSDictionary*)CMFormatDescriptionGetExtensions( FormatDescription );
	
	std::stringstream Output;
	
	for ( NSString* Key in Extensions )
	{
		Output << Soy::NSStringToString( Key ) << "=";
		
		@try
		{
			NSString* Value = [[Extensions objectForKey:Key] description];
			Output << Soy::NSStringToString( Value );
		}
		@catch (NSException* e)
		{
			Output << "<unkown value " << Soy::NSErrorToString( e ) << ">";
		}
		Output << ", ";
	}
	return Output.str();
}


bool GetExtension(CMFormatDescriptionRef FormatDescription,const std::string& Key,bool& Value)
{
	auto Extensions = (NSDictionary*)CMFormatDescriptionGetExtensions( FormatDescription );
	
	NSString* KeyNs = Soy::StringToNSString( Key );
	
	@try
	{
		NSString* DictValue = [[Extensions objectForKey:KeyNs] description];
		std::stringstream ValueStr;
		ValueStr << Soy::NSStringToString( DictValue );
		return Soy::StringToType( Value, ValueStr.str() );
	}
	@catch (NSException* e)
	{
		std::Debug << "<unkown value " << Soy::NSErrorToString( e ) << ">" << std::endl;
		return false;
	}
}


bool GetExtension(CMFormatDescriptionRef FormatDescription,const std::string& Key,std::string& Value)
{
	auto Extensions = (NSDictionary*)CMFormatDescriptionGetExtensions( FormatDescription );
	
	NSString* KeyNs = Soy::StringToNSString( Key );
	
	@try
	{
		NSString* DictValue = [[Extensions objectForKey:KeyNs] description];
		Value = Soy::NSStringToString( DictValue );
		return true;
	}
	@catch (NSException* e)
	{
		std::Debug << "<unkown value " << Soy::NSErrorToString( e ) << ">" << std::endl;
		return false;
	}
}


std::string GetH264ProfileName(uint8 AtomProfileIdc,uint8 AtomProfileIop,uint8 AtomLevel)
{
	//	http://stackoverflow.com/questions/21120717/h-264-video-wont-play-on-ios
	std::stringstream Name;
	if ( AtomProfileIdc == 0x42 )
		Name << "Baseline Profile";
	else if ( AtomProfileIdc == 0x4D )
		Name << "Main Profile";
	else if ( AtomProfileIdc == 0x58 )
		Name << "Extended Profile";
	else if ( AtomProfileIdc == 0x58 )
		Name << "High Profile";
	else
		Name << std::hex << AtomProfileIdc << "? Profile";
	
	//	show level
	//	https://github.com/ford-prefect/gst-plugins-bad/blob/master/sys/applemedia/vtdec.c
	//	http://stackoverflow.com/questions/21120717/h-264-video-wont-play-on-ios
	//	value is decimal * 10. Even if the data is bad, this should still look okay
	int Minor = AtomLevel % 10;
	int Major = (AtomLevel-Minor) / 10;
	
	Name << " " << Major << "." << Minor;
	
	//	get Iop options
	/*
		constraint_set0_flag 0 u(1)
	 constraint_set1_flag 0 u(1)
	 constraint_set2_flag 0 u(1)
	 constraint_set3_flag 0 u(1)
	 constraint_set4_flag 0 u(1)
	 constraint_set5_flag 0 u(1)
	 reserved_zero_2bits
	 */
	if ( AtomProfileIop != 0 )
		Name << " constraint ";
	
	for ( int i=0;	i<6;	i++ )
	{
		//	reversed endian
		int Bit = (8-i);
		if ( AtomProfileIop & (1<<Bit) )
			Name << i;
	}
	
	return Name.str();
}




void ExtractAppleEmbeddedData(std::string Atoms,std::map<std::string,std::string>& Dictionary)
{
	//	some kinda wierd {} wrapped key system
	BufferArray<char,10> Whitespace;
	Whitespace.PushBack(' ');
	Whitespace.PushBack('\n');
	Whitespace.PushBack('\t');
	Soy::StringTrimLeft( Atoms, '{' );
	
	while ( !Atoms.empty() )
	{
		Soy::StringTrimLeft( Atoms, GetArrayBridge(Whitespace) );
		
		//	read string until whitespace
		std::string Key = Soy::StringPopUntil( Atoms, ' ' );
		Soy::StringTrimLeft( Atoms, GetArrayBridge(Whitespace) );
		Soy::StringTrimLeft( Atoms, '=' );
		Soy::StringTrimLeft( Atoms, GetArrayBridge(Whitespace) );
		
		std::string Value;
		//	array of bytes
		if ( Atoms[0] == '<' )
		{
			Soy::StringTrimLeft( Atoms, '<' );
			Value = Soy::StringPopUntil( Atoms, '>' );
			Soy::StringTrimLeft( Atoms, '>' );
		}
		else
		{
			//	some other kind that needs special handling
		}
		Value += Soy::StringPopUntil( Atoms, ';' );
		Soy::StringTrimLeft( Atoms, ';' );
		
		Dictionary[Key] = Value;
	}
}

void ExtractAppleEmbeddedData(std::string Atoms,const std::string& Key,ArrayBridge<uint8>&& Data)
{
	std::map<std::string,std::string> Dictionary;
	ExtractAppleEmbeddedData( Atoms, Dictionary );
	
	auto ValueIt = Dictionary.find( Key );
	if ( ValueIt == Dictionary.end() )
	{
		std::stringstream Error;
		Error << "Couldn't find " << Key << " in apple embedded data";
		throw Soy::AssertException( Error.str() );
	}
	
	//	parse hex to bytes
	std::string DataStr = ValueIt->second;
	for ( int i=0;	i<DataStr.length();	i+=2 )
	{
		//	skip spacers
		if ( DataStr[i] == ' ' )
		{
			i++;
			continue;
		}
		Data.PushBack( Soy::HexToByte( DataStr[i+0], DataStr[i+1] ) );
	}
}





std::string Platform::GetCodec(CMFormatDescriptionRef FormatDescription)
{
	//	gr: use Soy::FourCCToString
	// Get the codec and correct endianness
	auto FourCC = CMFormatDescriptionGetMediaSubType( FormatDescription );
	CMVideoCodecType FormatCodec = CFSwapInt32BigToHost( FourCC );
	std::string CodecStr = Soy::FourCCToString( FormatCodec );
	
	//	for H264 (AVC1) get extended codec info
	if ( CodecStr == "avc1" )
	{
		//	get atom info
		std::string Atoms;
		if ( GetExtension( FormatDescription, "SampleDescriptionExtensionAtoms", Atoms ) )
		{
			try
			{
				//	bjork- plays on 5s
				//	{\n    avcC = <014d4032 ffe1002b 674d4032 96520040 0080dff8 0008000a 84000003 00040000 0300f392 00009896 00016e36 fc6383b4 2c5a2401 000468eb 7352>;\n}
				//	mancity01 starts, but never shows anything on 5s
				//	{\n    avcC = <01640033 ffe1001f 67640033 ac2ca400 f0010fb0 15202020 28000003 00080000 030184ed 0b168901 000568eb 735250>;\n}
				
				//	https://developer.apple.com/library/ios/documentation/NetworkingInternet/Conceptual/StreamingMediaGuide/FrequentlyAskedQuestions/FrequentlyAskedQuestions.html
				
				//	H.264 Baseline Level 3.0, Baseline Level 3.1, Main Level 3.1, and High Profile Level 4.1.
				
				//	extract the avcC data
				Array<uint8> AtomData;
				ExtractAppleEmbeddedData(Atoms,"avcC", GetArrayBridge(AtomData) );
				
				//	first data is profile data
				//	01 required
				//	PP PP profile
				//	LL level
				//	http://stackoverflow.com/questions/21120717/h-264-video-wont-play-on-ios
				Soy::Assert( AtomData.GetSize() >= 4, "Not enough atom data to get h264 profile&level" );
				Soy::Assert( AtomData[0] == 1, "Expected 1 as first atom byte");
				
				CodecStr = "H264";
				CodecStr += " ";
				CodecStr += GetH264ProfileName( AtomData[1], AtomData[2], AtomData[3] );
			}
			catch(std::exception& e)
			{
				std::Debug << "Failed to get H264 atom info: " << e.what() << std::endl;
			}
		}
	}
	
	return CodecStr;
}


std::string Platform::GetCVReturnString(CVReturn Error)
{
#define CASE(e)	case (e): return #e
	switch ( Error )
	{
			CASE( kCVReturnSuccess );
			CASE( kCVReturnFirst );
			CASE( kCVReturnInvalidArgument );
			CASE( kCVReturnAllocationFailed );
			CASE( kCVReturnInvalidDisplay );
			CASE( kCVReturnDisplayLinkAlreadyRunning );
			CASE( kCVReturnDisplayLinkNotRunning );
			CASE( kCVReturnDisplayLinkCallbacksNotSet );
			CASE( kCVReturnInvalidPixelFormat );
			CASE( kCVReturnInvalidSize );
			CASE( kCVReturnInvalidPixelBufferAttributes );
			CASE( kCVReturnPixelBufferNotOpenGLCompatible );
			CASE( kCVReturnPixelBufferNotMetalCompatible );
			CASE( kCVReturnWouldExceedAllocationThreshold );
			CASE( kCVReturnPoolAllocationFailed );
			CASE( kCVReturnInvalidPoolAttributes );
		default:
		{
			std::stringstream Err;
			Err << "Unknown CVReturn error: " << Error;
			return Err.str();
		}
	}
#undef CASE
}



#if defined(TARGET_IOS)
Opengl::TTexture ExtractPlaneTexture(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVOpenGLESTextureRef>& TextureRef,CFAllocatorRef& Allocator,size_t PlaneIndex,SoyPixelsFormat::Type PlaneFormat)
{
	GLenum TextureType = GL_TEXTURE_2D;
	GLenum PixelComponentType = GL_UNSIGNED_BYTE;
	
	//	gr: switch these if's to the right pixelformat mapping
	GLenum InternalFormat = GL_INVALID_VALUE;		//	gr: spent a WEEK trying to get this to work. RGBA is the only one that works. not BGRA! ignore the docs!
	GLenum PixelFormat = GL_INVALID_VALUE;			//	todo: fetch this from image buffer
	size_t Width = 0;
	size_t Height = 0;

	//	gr: as far as I can tell, we can't get the pixel format for a plane, so we've pre-empted it
	auto Format = PlaneFormat;
	
	if ( Format == SoyPixelsFormat::Luma_Full || Format == SoyPixelsFormat::Luma_Ntsc || Format == SoyPixelsFormat::Luma_Smptec )
	{
		TextureType = GL_LUMINANCE;
		PixelFormat = GL_LUMINANCE;
		Width = CVPixelBufferGetWidthOfPlane( ImageBuffer, PlaneIndex );
		Height = CVPixelBufferGetHeightOfPlane( ImageBuffer, PlaneIndex );
	}
	else if ( Format == SoyPixelsFormat::GreyscaleAlpha )
	{
		TextureType = GL_LUMINANCE_ALPHA;
		PixelFormat = GL_LUMINANCE_ALPHA;
		Width = CVPixelBufferGetWidthOfPlane( ImageBuffer, PlaneIndex );
		Height = CVPixelBufferGetHeightOfPlane( ImageBuffer, PlaneIndex );
	}
	else
	{
		//	gr: spent a WEEK trying to get this to work. RGBA is the only one that works. not BGRA! ignore the docs!
		//	todo: fetch this from image buffer
		InternalFormat = GL_RGBA;
		PixelFormat = GL_BGRA;
		Width = CVPixelBufferGetWidth( ImageBuffer );
		Height = CVPixelBufferGetHeight( ImageBuffer );
	}
	
	auto Result = CVOpenGLESTextureCacheCreateTextureFromImage(	Allocator,
															   TextureCache.mOpenglTextureCache.mObject,
															   ImageBuffer,
															   nullptr,
															   TextureType,
															   InternalFormat,
															   size_cast<GLsizei>(Width),
															   size_cast<GLsizei>(Height),
															   PixelFormat,
															   PixelComponentType,
															   PlaneIndex,
															   &TextureRef.mObject
															   );
	
	if ( Result != kCVReturnSuccess || !TextureRef.mObject )
	{
		auto BytesPerRow = CVPixelBufferGetBytesPerRowOfPlane( ImageBuffer, PlaneIndex );
		Opengl::IsOkay("Failed to CVOpenGLTextureCacheCreateTextureFromImage",false);
		std::Debug << "Failed to create texture from image " << Platform::GetCVReturnString(Result) << " bytes per row: " << BytesPerRow << "plane #" << PlaneIndex << " as " << Format << std::endl;
		return Opengl::TTexture();
	}
	
	auto RealTextureType = CVOpenGLESTextureGetTarget( TextureRef.mObject );
	auto RealTextureName = CVOpenGLESTextureGetName( TextureRef.mObject );
	SoyPixelsMeta Meta( Width, Height, SoyPixelsFormat::RGBA );
	
	Opengl::TTexture Texture( RealTextureName, Meta, RealTextureType );
	return Texture;
}
#endif

#if defined(TARGET_IOS) && defined(ENABLE_METAL)
Metal::TTexture ExtractPlaneTexture_Metal(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVMetalTextureRef>& TextureRef,CFAllocatorRef& Allocator,size_t PlaneIndex,SoyPixelsFormat::Type PlaneFormat)
{
	MTLPixelFormat PixelFormat = MTLPixelFormatInvalid;
	size_t Width = 0;
	size_t Height = 0;
	
	//	gr: as far as I can tell, we can't get the pixel format for a plane, so we've pre-empted it
	auto Format = PlaneFormat;
	
	if ( Format == SoyPixelsFormat::Luma_Full || Format == SoyPixelsFormat::Luma_Ntsc || Format == SoyPixelsFormat::Luma_Smptec )
	{
		PixelFormat = MTLPixelFormatR8Uint;
		Width = CVPixelBufferGetWidthOfPlane( ImageBuffer, PlaneIndex );
		Height = CVPixelBufferGetHeightOfPlane( ImageBuffer, PlaneIndex );
	}
	else if ( Format == SoyPixelsFormat::GreyscaleAlpha )
	{
		PixelFormat = MTLPixelFormatRG8Uint;
		Width = CVPixelBufferGetWidthOfPlane( ImageBuffer, PlaneIndex );
		Height = CVPixelBufferGetHeightOfPlane( ImageBuffer, PlaneIndex );
	}
	else
	{
		PixelFormat = MTLPixelFormatBGRA8Unorm;
		Width = CVPixelBufferGetWidth( ImageBuffer );
		Height = CVPixelBufferGetHeight( ImageBuffer );
	}
	
	
	auto Result = CVMetalTextureCacheCreateTextureFromImage(	Allocator,
															   TextureCache.mMetalTextureCache.mObject,
															   ImageBuffer,
															   nullptr,
															   PixelFormat,
															   size_cast<GLsizei>(Width),
															   size_cast<GLsizei>(Height),
															   PlaneIndex,
															   &TextureRef.mObject
															   );
	
	if ( Result != kCVReturnSuccess || !TextureRef.mObject )
	{
		auto BytesPerRow = CVPixelBufferGetBytesPerRowOfPlane( ImageBuffer, PlaneIndex );
		Opengl::IsOkay("Failed to CVMetalTextureCacheCreateTextureFromImage",false);
		std::Debug << "Failed to create texture from image " << Platform::GetCVReturnString(Result) << " bytes per row: " << BytesPerRow << "plane #" << PlaneIndex << " as " << Format << std::endl;
		return Metal::TTexture();
	}
	
	id<MTLTexture> MetalTexture = CVMetalTextureGetTexture( TextureRef.mObject );
	Metal::TTexture Texture( MetalTexture );
	return Texture;
}
#endif

#if defined(TARGET_IOS)
Opengl::TTexture ExtractNonPlanarTexture(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVOpenGLESTextureRef>& TextureRef,CFAllocatorRef& Allocator)
#elif defined(TARGET_OSX)
Opengl::TTexture ExtractNonPlanarTexture(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVOpenGLTextureRef>& TextureRef,CFAllocatorRef& Allocator)
#endif
{
#if defined(TARGET_IOS)
	
	//	ios can just grab plane 0
	auto FormatCv = CVPixelBufferGetPixelFormatType( ImageBuffer );
	auto Format = Avf::GetPixelFormat( FormatCv );
	return ExtractPlaneTexture( TextureCache, ImageBuffer, TextureRef, Allocator, 0, Format );
	
#elif defined(TARGET_OSX)
	
	//	http://stackoverflow.com/questions/13933503/core-video-pixel-buffers-as-gl-texture-2d
	auto Result = CVOpenGLTextureCacheCreateTextureFromImage(Allocator,
															 TextureCache.mOpenglTextureCache.mObject,
															 ImageBuffer,
															 nullptr,
															 &TextureRef.mObject);
	
	
	if ( Result != kCVReturnSuccess || !TextureRef.mObject )
	{
		Opengl::IsOkay("Failed to CVOpenGLTextureCacheCreateTextureFromImage",false);
		std::Debug << "Failed to create texture from image " << Platform::GetCVReturnString(Result) << std::endl;
		return Opengl::TTexture();
	}
	
	auto Width = CVPixelBufferGetWidth( ImageBuffer );
	auto Height = CVPixelBufferGetHeight( ImageBuffer );
	auto RealTextureType = CVOpenGLTextureGetTarget( TextureRef.mObject );
	auto RealTextureName = CVOpenGLTextureGetName( TextureRef.mObject );
	
	//	we dont KNOW the internal format, so create a temp texture, then
	//	use it to pull out the real pixel format
	//	gr: get format from ImageBuffer!
	SoyPixelsMeta TmpMeta( Width, Height, SoyPixelsFormat::RGBA );
	Opengl::TTexture TmpTexture( RealTextureName, TmpMeta, RealTextureType );
	GLenum AutoRealTextureType = RealTextureType;
	SoyPixelsMeta Meta = TmpTexture.GetInternalMeta(AutoRealTextureType);
	Opengl::TTexture Texture( RealTextureName, Meta, RealTextureType );
	
	return Texture;
#endif
}


#if defined(ENABLE_METAL)
#if defined(TARGET_IOS)
Metal::TTexture ExtractNonPlanarTexture_Metal(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVMetalTextureRef>& TextureRef,CFAllocatorRef& Allocator)
#elif defined(TARGET_OSX)&&defined(ENABLE_METAL)
Metal::TTexture ExtractNonPlanarTexture_Metal(AvfTextureCache& TextureCache,CVImageBufferRef ImageBuffer,CFPtr<CVMetalTextureRef>& TextureRef,CFAllocatorRef& Allocator)
#endif
{
#if defined(TARGET_IOS)
	
	//	ios can just grab plane 0
	auto FormatCv = CVPixelBufferGetPixelFormatType( ImageBuffer );
	auto Format = Avf::GetPixelFormat( FormatCv );
	return ExtractPlaneTexture_Metal( TextureCache, ImageBuffer, TextureRef, Allocator, 0, Format );
	
#elif defined(TARGET_OSX)
	throw Soy::AssertException("ExtractNonPlanarTexture_Metal on osx not supported yet");
	/*
	//	http://stackoverflow.com/questions/13933503/core-video-pixel-buffers-as-gl-texture-2d
	auto Result = CVOpenGLTextureCacheCreateTextureFromImage(Allocator,
															 TextureCache.mTextureCache.mObject,
															 ImageBuffer,
															 nullptr,
															 &TextureRef.mObject);
	
	
	if ( Result != kCVReturnSuccess || !TextureRef.mObject )
	{
		Opengl::IsOkay("Failed to CVOpenGLTextureCacheCreateTextureFromImage",false);
		std::Debug << "Failed to create texture from image " << Platform::GetCVReturnString(Result) << std::endl;
		return Opengl::TTexture();
	}
	
	auto Width = CVPixelBufferGetWidth( ImageBuffer );
	auto Height = CVPixelBufferGetHeight( ImageBuffer );
	auto RealTextureType = CVOpenGLTextureGetTarget( TextureRef.mObject );
	auto RealTextureName = CVOpenGLTextureGetName( TextureRef.mObject );
	
	//	we dont KNOW the internal format, so create a temp texture, then
	//	use it to pull out the real pixel format
	//	gr: get format from ImageBuffer!
	SoyPixelsMeta TmpMeta( Width, Height, SoyPixelsFormat::RGBA );
	Opengl::TTexture TmpTexture( RealTextureName, TmpMeta, RealTextureType );
	GLenum AutoRealTextureType = RealTextureType;
	SoyPixelsMeta Meta = TmpTexture.GetInternalMeta(AutoRealTextureType);
	Opengl::TTexture Texture( RealTextureName, Meta, RealTextureType );
	
	return Texture;
	 */
#endif
}
#endif//ENABLE_METAL





void AvfPixelBuffer::Lock(ArrayBridge<Metal::TTexture>&& Textures,Metal::TContext& Context,float3x3& Transform)
{
#if defined(ENABLE_METAL)
	Soy::Assert( mDecoder!=nullptr, "Decoder expected" );
	
	auto ImageBuffer = LockImageBuffer();
	if ( !ImageBuffer )
	{
		std::Debug << "Failed to get ImageBuffer from CMSampleBuffer" << std::endl;
		Unlock();
		return;
	}
	
	CVReturn Result = CVPixelBufferLockBaseAddress( ImageBuffer, mReadOnlyLock ? kCVPixelBufferLock_ReadOnly : 0 );
	if ( Result != kCVReturnSuccess  )
	{
		Opengl::IsOkay("Failed to lock address",false);
		std::Debug << "Error locking base address of image: " << Platform::GetCVReturnString(Result) << std::endl;
		Unlock();
		return;
	}
	
	CFAllocatorRef Allocator = kCFAllocatorDefault;
	
	
#if defined(TARGET_IOS)
	
	auto PlaneCount = CVPixelBufferGetPlaneCount( ImageBuffer );
	if ( PlaneCount > 0 )
	{
		BufferArray<SoyPixelsFormat::Type,2> PlaneFormats;
		auto Format = CVPixelBufferGetPixelFormatType( ImageBuffer );
		auto SoyFormat = Avf::GetPixelFormat( Format );
		SoyPixelsFormat::GetFormatPlanes( SoyFormat, GetArrayBridge(PlaneFormats) );
		for ( int i=0;	i<PlaneCount;	i++ )
		{
			auto TextureCache = mDecoder->GetTextureCache( i, &Context );
			if ( !TextureCache )
				throw Soy::AssertException("Failed to get texture cache");
			
			//	hacky
			auto& TextureRef = (i==0) ? mMetal_LockedTexture0 : mMetal_LockedTexture1;
			
			mTextureCaches.PushBack( TextureCache );
			auto Texture = ExtractPlaneTexture_Metal( *TextureCache, ImageBuffer, TextureRef, Allocator, i, PlaneFormats[i] );
			if ( !Texture.IsValid() )
				continue;
			
			Textures.PushBack( Texture );
		}
	}
	else
	{
		auto& TextureRef = mMetal_LockedTexture0;
		auto TextureCache = mDecoder->GetTextureCache( 0, &Context );
		if ( !TextureCache )
			throw Soy::AssertException("Failed to get texture cache");
		
		mTextureCaches.PushBack( TextureCache );
		auto Texture = ExtractNonPlanarTexture_Metal( *TextureCache, ImageBuffer, TextureRef, Allocator );
		
		if ( Texture.IsValid() )
			Textures.PushBack( Texture );
	}
#elif defined(TARGET_OSX)
	
	//	on OSX, we can't pull multiple planes into opengl textures (get the "incompatible with opengl error")
	//	so we don't return anything. Caller has to use pixels -> texture instead
	auto PlaneCount = CVPixelBufferGetPlaneCount( ImageBuffer );
	if ( PlaneCount > 0 )
	{
		Unlock();
		return;
	}
	
	auto TextureCache = mDecoder->GetTextureCache(0);
	if ( !TextureCache )
		throw Soy::AssertException("Failed to get texture cache");
	
	mTextureCaches.PushBack( TextureCache );
	auto Texture = ExtractNonPlanarTexture( *TextureCache, ImageBuffer, mLockedTexture, Allocator );
	
	if ( Texture.IsValid() )
		Textures.PushBack( Texture );
#endif
#endif//ENABLE_METAL
}



void AvfPixelBuffer::Lock(ArrayBridge<Opengl::TTexture>&& Textures,Opengl::TContext& Context,float3x3& Transform)
{
	Opengl::IsOkay("LockTexture flush", false);
	
	Soy::Assert( mDecoder!=nullptr, "Decoder expected" );
	
	auto ImageBuffer = LockImageBuffer();
	/*
	 auto& Buffer = mSample.mObject;
	 mLockedImageBuffer.Retain( CMSampleBufferGetImageBuffer(Buffer) );
	 */
	if ( !ImageBuffer )
	{
		std::Debug << "Failed to get ImageBuffer from CMSampleBuffer" << std::endl;
		Unlock();
		return;
	}
	
	CVReturn Result = CVPixelBufferLockBaseAddress( ImageBuffer, mReadOnlyLock ? kCVPixelBufferLock_ReadOnly : 0 );
	if ( Result != kCVReturnSuccess  )
	{
		Opengl::IsOkay("Failed to lock address",false);
		std::Debug << "Error locking base address of image: " << Platform::GetCVReturnString(Result) << std::endl;
		Unlock();
		return;
	}
	
	CFAllocatorRef Allocator = kCFAllocatorDefault;
	
	
#if defined(TARGET_IOS)
	
	auto PlaneCount = CVPixelBufferGetPlaneCount( ImageBuffer );
	if ( PlaneCount > 0 )
	{
		BufferArray<SoyPixelsFormat::Type,2> PlaneFormats;
		auto Format = CVPixelBufferGetPixelFormatType( ImageBuffer );
		auto SoyFormat = Avf::GetPixelFormat( Format );
		SoyPixelsFormat::GetFormatPlanes( SoyFormat, GetArrayBridge(PlaneFormats) );
		for ( int i=0;	i<PlaneCount;	i++ )
		{
			auto TextureCache = mDecoder->GetTextureCache( i, nullptr );
			if ( !TextureCache )
				throw Soy::AssertException("Failed to get texture cache");
			
			//	hacky
			auto& TextureRef = (i==0) ? mLockedTexture0 : mLockedTexture1;
			
			mTextureCaches.PushBack( TextureCache );
			auto Texture = ExtractPlaneTexture( *TextureCache, ImageBuffer, TextureRef, Allocator, i, PlaneFormats[i] );
			if ( !Texture.IsValid() )
				continue;
			
			Textures.PushBack( Texture );
		}
	}
	else
	{
		auto& TextureRef = mLockedTexture0;
		auto TextureCache = mDecoder->GetTextureCache( 0, nullptr );
		if ( !TextureCache )
			throw Soy::AssertException("Failed to get texture cache");
		
		mTextureCaches.PushBack( TextureCache );
		auto Texture = ExtractNonPlanarTexture( *TextureCache, ImageBuffer, TextureRef, Allocator );
		
		if ( Texture.IsValid() )
			Textures.PushBack( Texture );
	}
#elif defined(TARGET_OSX)
	
	//	on OSX, we can't pull multiple planes into opengl textures (get the "incompatible with opengl error")
	//	so we don't return anything. Caller has to use pixels -> texture instead
	auto PlaneCount = CVPixelBufferGetPlaneCount( ImageBuffer );
	if ( PlaneCount > 0 )
	{
		Unlock();
		return;
	}
	
	auto TextureCache = mDecoder->GetTextureCache(0, nullptr);
	if ( !TextureCache )
		throw Soy::AssertException("Failed to get texture cache");
	
	mTextureCaches.PushBack( TextureCache );
	auto Texture = ExtractNonPlanarTexture( *TextureCache, ImageBuffer, mLockedTexture, Allocator );
	
	if ( Texture.IsValid() )
		Textures.PushBack( Texture );
#endif
}



AvfPixelBuffer::~AvfPixelBuffer()
{
	//	gotta WAIT for this to unlock from the other thread! NOT unlock it ourselves
	WaitForUnlock();
}



CFPixelBuffer::~CFPixelBuffer()
{
	//	gotta WAIT for this to unlock from the other thread! NOT unlock it ourselves
	WaitForUnlock();
	
	//auto RetainCount = mSample.GetRetainCount();
	//std::Debug << "Sample has " << RetainCount << " references in " << __func__ << std::endl;
	
	static bool invalidate = false;
	if ( mSample && invalidate )
		CMSampleBufferInvalidate(mSample.mObject);
	mSample.Release();
}


CVImageBufferRef CFPixelBuffer::LockImageBuffer()
{
	Soy::Assert( mLockedImageBuffer == false, "Image buffer already locked");
	
	mLockedImageBuffer.Retain( CMSampleBufferGetImageBuffer(mSample.mObject) );
	if ( mLockedImageBuffer )
		return mLockedImageBuffer.mObject;
	
	//	debug why it failed
	std::Debug << "Failed to get ImageBuffer from CMSampleBuffer... ";
	
	auto DataIsReady = CMSampleBufferDataIsReady( mSample.mObject );
	std::Debug << "Data is ready: " << DataIsReady << ", ";
	
	
	CMFormatDescriptionRef FormatDescription = CMSampleBufferGetFormatDescription( mSample.mObject );
	if ( FormatDescription == nullptr )
		std::Debug << "Format description: null,";
	else
		std::Debug << "Format description: not null,";
	
	std::Debug << std::endl;
	
	return mLockedImageBuffer.mObject;
}


void AvfPixelBuffer::LockPixels(ArrayBridge<SoyPixelsImpl*>& Planes,void* _Data,size_t BytesPerRow,SoyPixelsMeta Meta,float3x3& Transform,ssize_t DataSize)
{
	//	check for mis-alignment
	//	todo: allocate and manually clip rows... or change the meta and crop in shader? (assuming the bytes align to the same pixel bit depth)

	//	gr: 427 × 240 quicktime movie results in this
	//		Expected 427 is 448
	if ( Meta.GetRowDataSize() != BytesPerRow )
	{
		//	gr: if the data aligns to the pixelformat, then pad the image to fit the buffer and clip later in the shader(todo)
		auto Channels = Meta.GetChannels();
		if ( Channels > 0 && BytesPerRow % Channels == 0 )
		{
			//	realign image and clip with transform
			auto NewWidth = BytesPerRow / Channels;
			Transform(0,0) *= Meta.GetWidth() / static_cast<float>( NewWidth );
			//std::Debug << "Padding mis-aligned image " << Meta << " width to " << NewWidth << std::endl;
			Meta.DumbSetWidth( NewWidth );
		}
		else
		{
			std::stringstream Error;
			Error << "CVPixelBuffer for plane " << mLockedPixels.GetSize() << " (" << Meta << ") row mis-aligned, handle this. Expected " << Meta.GetRowDataSize() << " is " << BytesPerRow;
			throw Soy::AssertException( Error.str() );
		}
	}
	
	//	now apply the parent(stream) transform
	{
		auto TransformMtx = Soy::VectorToMatrix( Transform );
		auto ParentTransformMtx = Soy::VectorToMatrix( mTransform );
		TransformMtx *= ParentTransformMtx;
		Transform = Soy::MatrixToVector( TransformMtx );
	}
	
	auto* Pixels = reinterpret_cast<uint8*>(_Data);
	
	//	auto calc data size if not provided by caller
	if ( DataSize < 0 )
		DataSize = Meta.GetDataSize();
	
	SoyPixelsRemote Temp( Pixels, DataSize, Meta );
	mLockedPixels.PushBack( Temp );
	Planes.PushBack( &mLockedPixels.GetBack() );
}


void AvfPixelBuffer::Lock(ArrayBridge<SoyPixelsImpl*>&& Planes,float3x3& Transform)
{
	mLockLock.lock();
	
	//	reset
	mLockedPixels.SetAll( SoyPixelsRemote() );
	mLockedPixels.SetSize(0);
	
	auto PixelBuffer = LockImageBuffer();
	if ( !PixelBuffer )
	{
		Unlock();
		return;
	}
	
	auto Error = CVPixelBufferLockBaseAddress( PixelBuffer, mReadOnlyLock ? kCVPixelBufferLock_ReadOnly : 0 );
	if ( Error != kCVReturnSuccess )
	{
		std::Debug << "Failed to lock CVPixelBuffer address " << Platform::GetCVReturnString( Error ) << std::endl;
		Unlock();
		return;
	}
	
	//	here we diverge for multiple planes
	auto PlaneCount = CVPixelBufferGetPlaneCount( PixelBuffer );
	if ( PlaneCount >= 1 )
	{
		BufferArray<SoyPixelsFormat::Type,2> PlaneFormats;
		auto Format = CVPixelBufferGetPixelFormatType( PixelBuffer );
		auto SoyFormat = Avf::GetPixelFormat( Format );
		SoyPixelsFormat::GetFormatPlanes( SoyFormat, GetArrayBridge(PlaneFormats) );
		auto PixelBufferDataSize = CVPixelBufferGetDataSize(PixelBuffer);
		for ( size_t PlaneIndex=0;	PlaneIndex<PlaneCount;	PlaneIndex++ )
		{
			//	gr: although the blitter can split this for us, I assume there MAY be a case where planes are not contiguous, so for this platform handle it explicitly
			auto Width = CVPixelBufferGetWidthOfPlane( PixelBuffer, PlaneIndex );
			auto Height = CVPixelBufferGetHeightOfPlane( PixelBuffer, PlaneIndex );
			auto* Pixels = CVPixelBufferGetBaseAddressOfPlane( PixelBuffer, PlaneIndex );
			auto BytesPerRow = CVPixelBufferGetBytesPerRowOfPlane( PixelBuffer, PlaneIndex );
			auto PlaneFormat = PlaneFormats[PlaneIndex];
			if ( !Pixels )
			{
				std::Debug << "Image plane #" << PlaneIndex << "/" << PlaneCount << " " << Width << "x" << Height << " return null" << std::endl;
				continue;
			}
			
			//	data size here is for the whole image, so we need to calculate (ie. ASSUME) it ourselves.
			SoyPixelsMeta PlaneMeta( Width, Height, PlaneFormat );

			//	should be LESS as there are multiple plaens in the total buffer, but we'll do = just for the sake of the safety
			Soy::Assert( PlaneMeta.GetDataSize() <= PixelBufferDataSize, "Plane's calcualted data size exceeds the total buffer size" );

			//	gr: currently we only have one transform... so... only apply to main plane (and hope they're the same)
			float3x3 DummyTransform;
			float3x3& PlaneTransform = (PlaneIndex == 0) ? Transform : DummyTransform;
			
			LockPixels( Planes, Pixels, BytesPerRow, PlaneMeta, PlaneTransform );
		}
	}
	else
	{
		//	get the "non-planar" image
		auto Height = CVPixelBufferGetHeight( PixelBuffer );
		auto Width = CVPixelBufferGetWidth( PixelBuffer );
		auto* Pixels = CVPixelBufferGetBaseAddress(PixelBuffer);
		auto Format = CVPixelBufferGetPixelFormatType( PixelBuffer );
		auto DataSize = CVPixelBufferGetDataSize(PixelBuffer);
		auto SoyFormat = Avf::GetPixelFormat( Format );
		auto BytesPerRow = CVPixelBufferGetBytesPerRow( PixelBuffer );
		
		if ( !Pixels )
		{
			Unlock();
			return;
		}
		
		SoyPixelsMeta Meta( Width, Height, SoyFormat );
		LockPixels( Planes, Pixels, BytesPerRow, Meta, Transform, DataSize );

		/*
		if ( Meta.GetRowDataSize() != BytesPerRow )
		{
			std::stringstream Error;
			Error << "CVPixelBuffer (" << Meta << ") row mis-aligned, handle this. Expected " << Meta.GetRowDataSize() << " is " << BytesPerRow;
			throw Soy::AssertException( Error.str() );
		}
		 */
		
		//	gr: wierdly... with bjork, RGB data, 2048x2048... there are an extra 32 bytes... the plane split will throw an error on this, so just trim it...
		if ( DataSize > Meta.GetDataSize() )
		{
			//auto Diff = DataSize - Meta.GetDataSize();
			//std::Debug << "Warning: CVPixelBuffer data has an extra " << Diff << " bytes. Trimming..." << std::endl;
			DataSize = Meta.GetDataSize();
		}
		
		SoyPixelsRemote Temp( reinterpret_cast<uint8*>(Pixels), Width, Height, DataSize, SoyFormat );
		mLockedPixels[0] = Temp;
		Planes.PushBack( &mLockedPixels[0] );
	}
}



void AvfPixelBuffer::Unlock()
{
	//	release our use of the texture cache
	auto ClearTextureCache = [](std::shared_ptr<AvfTextureCache>& Cache)
	{
		if ( Cache )
			Cache->Flush();
		Cache.reset();
		return true;
	};
	GetArrayBridge(mTextureCaches).ForEach(ClearTextureCache);
	mTextureCaches.Clear();
	
#if defined(TARGET_IOS)
	if ( mLockedTexture0 )
		mLockedTexture0.Release();
	if ( mLockedTexture1 )
		mLockedTexture1.Release();
#elif defined(TARGET_OSX)
	if ( mLockedTexture )
		mLockedTexture.Release();
#endif
	mLockedPixels.SetAll( SoyPixelsRemote() );
	mLockedPixels.SetSize(0);
	
	UnlockImageBuffer();
	mLockLock.unlock();
}



void CFPixelBuffer::UnlockImageBuffer()
{
	if ( mLockedImageBuffer )
	{
		//	must make sure nothing is using texture before releasing it
		//glFlush();
		CVPixelBufferUnlockBaseAddress(mLockedImageBuffer.mObject, mReadOnlyLock ? kCVPixelBufferLock_ReadOnly : 0 );
		
		//std::Debug << "[c] Locked image buffer refcount before release: " << mLockedImageBuffer.GetRetainCount() << std::endl;
		mLockedImageBuffer.Release();
	}
}

void AvfPixelBuffer::WaitForUnlock()
{
	while ( !mLockLock.try_lock() )
	{
		std::Debug << "CFPixelBuffer: waiting for image buffer to unlock" << std::endl;
	}
	mLockLock.unlock();
}


#if defined(TARGET_IOS)
__export EAGLContext*	UnityGetMainScreenContextGLES();
//extern EAGLContext*	UnityGetContextEAGL();
#endif


AvfTextureCache::AvfTextureCache(Metal::TContext* MetalContext)
{
	if ( MetalContext )
	{
		AllocMetal(*MetalContext);
	}
	else
	{
		AllocOpengl();
	}
}

void AvfTextureCache::AllocOpengl()
{
	CFAllocatorRef Allocator = kCFAllocatorDefault;
	
#if defined(TARGET_IOS)
	
	auto Context = UnityGetMainScreenContextGLES();
	CVOpenGLESTextureCacheRef Cache;
	auto Result = CVOpenGLESTextureCacheCreate ( Allocator, nullptr, Context, nullptr, &Cache );
	
#elif defined(TARGET_OSX)
	
	auto Context = CGLGetCurrentContext();
	//	cannot pass null pixelformat like we can on IOS
	//	gr: not retained... assuming okay for this small amount of time
	CGLPixelFormatObj PixelFormat = CGLGetPixelFormat( Context );
	CVOpenGLTextureCacheRef Cache = nullptr;
	auto Result = CVOpenGLTextureCacheCreate ( Allocator, nullptr, Context, PixelFormat, nullptr, &Cache );
	
#endif
	
	//	gr: unnecessary additional retain?
	static bool AdditionalRetain = false;
	if ( AdditionalRetain )
		mOpenglTextureCache.Retain( Cache );
	else
		mOpenglTextureCache.SetNoRetain( Cache );
	
	
	Opengl::IsOkay("Create texture cache flush", false);
	
	if ( Result != kCVReturnSuccess )
	{
		std::stringstream Error;
		Error << "Failed to allocate texture cache " << Platform::GetCVReturnString(Result);
		throw Soy::AssertException( Error.str() );
	}
}


void AvfTextureCache::AllocMetal(Metal::TContext& Context)
{
#if defined(ENABLE_METAL)
	CFAllocatorRef Allocator = kCFAllocatorDefault;
	
	CVMetalTextureCacheRef Cache;
	auto Result = CVMetalTextureCacheCreate( Allocator, nullptr, Context.GetDevice(), nullptr, &Cache );
	
	mMetalTextureCache.SetNoRetain( Cache );
	
	if ( Result != kCVReturnSuccess )
	{
		std::stringstream Error;
		Error << "Failed to allocate texture cache " << Platform::GetCVReturnString(Result);
		throw Soy::AssertException( Error.str() );
	}
#endif
	
}


AvfTextureCache::~AvfTextureCache()
{
	Flush();
	mOpenglTextureCache.Release();
#if defined(ENABLE_METAL)
	mMetalTextureCache.Release();
#endif
}


void AvfTextureCache::Flush()
{
	//	gotta make sure all uses of texture are done before flushing
	//	gr: might not have an opengl context! must move this to the last-texture use!
	//glFlush();
#if defined(ENABLE_METAL)
	if ( mMetalTextureCache )
		CVMetalTextureCacheFlush( mMetalTextureCache.mObject, 0 );
#endif

#if defined(TARGET_IOS)
	if ( mOpenglTextureCache )
		CVOpenGLESTextureCacheFlush( mOpenglTextureCache.mObject, 0 );
#endif

#if defined(TARGET_OSX)
	if ( mOpenglTextureCache )
		CVOpenGLTextureCacheFlush( mOpenglTextureCache.mObject, 0 );
#endif
}



std::shared_ptr<AvfTextureCache> AvfDecoderRenderer::GetTextureCache(size_t Index,Metal::TContext* MetalContext)
{
	if ( Index >= mTextureCaches.GetSize() )
		mTextureCaches.SetSize( Index+1 );
	
	auto& TextureCache = mTextureCaches[Index];
	if ( TextureCache )
		return TextureCache;
	
	try
	{
		TextureCache.reset( new AvfTextureCache(MetalContext) );
	}
	catch (std::exception& e)
	{
		std::Debug << "Failed to create AvfTextureCache: " << e.what() << std::endl;
		TextureCache.reset();
	}
	
	return TextureCache;
}

