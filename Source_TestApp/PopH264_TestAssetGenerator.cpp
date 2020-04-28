#include "SoyPixels.h"
#include "SoyPng.h"
#include "SoyFileSystem.h"
#include "json11.hpp"

extern void DebugPrint(const std::string& Message);



SoyPixelsMeta GetPlaneMeta(const char* MetaJson,int PlaneIndex)
{
	std::string JsonError;
	auto Json = json11::Json::parse( MetaJson, JsonError );
	if ( !JsonError.empty() )
	throw std::runtime_error(JsonError);
	auto Planes = Json["Planes"].array_items();
	auto Plane = Planes[PlaneIndex];
	auto Width = Plane["Width"].int_value();
	auto Height = Plane["Height"].int_value();
	auto FormatStr = Plane["Format"].string_value();
	auto Format = SoyPixelsFormat::ToType(FormatStr);
	return SoyPixelsMeta(Width,Height,Format);
}



SoyPixels MakeGreyscalePixels()
{
	SoyPixels Pixels(SoyPixelsMeta(10,256,SoyPixelsFormat::RGB));
	auto Components = Pixels.GetMeta().GetChannels();
	auto& PixelsArray = Pixels.GetPixelsArray();
	for ( auto y=0;	y<Pixels.GetHeight();	y++ )
	{
		for ( auto x=0;	x<Pixels.GetWidth();	x++ )
		{
			auto i = y*Pixels.GetWidth();
			i += x;
			i *= Components;
			PixelsArray[i+0] = y;
			PixelsArray[i+1] = y;
			PixelsArray[i+2] = y;
		}
	}
	return Pixels;
}

void CompareGreyscale(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data)
{
	auto Plane0Meta = GetPlaneMeta(MetaJson,0);
	auto Plane0DataSize = Plane0Meta.GetDataSize();
	SoyPixelsRemote Plane0( Plane0Data, Plane0DataSize, Plane0Meta );
	
	//	compare with original
	SoyPixels Original = MakeGreyscalePixels();

	//	debug first column (for the greyscale test image)
	{
		std::stringstream Debug;
		auto Width = Plane0Meta.GetWidth();
		auto Height = Plane0Meta.GetHeight();
		auto Channels = Plane0Meta.GetChannels();
		for ( auto y=0;	y<Height;	y++ )
		{
			auto x = 0;
			auto Old = Original.GetPixel(x,y,0);
			auto New = Plane0.GetPixel(x,y,0);
			auto Diff = New-Old;
			Debug << static_cast<int>(Diff) << ' ';
		}
		DebugPrint(Debug.str());
	}
}

void MakeGreyscalePng(const char* Filename)
{
	auto Pixels = MakeGreyscalePixels();
	Array<uint8_t> PngData;
	TPng::GetPng( Pixels, GetArrayBridge(PngData), 0 );
	Soy::ArrayToFile( GetArrayBridge(PngData), Filename);
	Platform::ShowFileExplorer(Filename);
}

void GetRainbowRgb(int y,uint8_t& r,uint8_t& g,uint8_t& b)
{
	auto GetYNorm = [&](int NextSection)
	{
		auto f = y - ( 255 * (NextSection-1) );
		return f;
	};
	if ( y <= 255*1 )
	{
		auto f = GetYNorm(1);
		r = 255;	g = f;	b = 0;
	}
	else if ( y <= 255*2 )
	{
		auto f = GetYNorm(2);
		r = 255-f;	g = 255;	b = 0;
	}
	else if ( y <= 255*3 )
	{
		auto f = GetYNorm(3);
		r = 0;	g = 255;	b = f;
	}
	else if ( y <= 255*4 )
	{
		auto f = GetYNorm(4);
		r = 0;	g = 255-f;	b = 255;
	}
	else if ( y <= 255*5 )
	{
		auto f = GetYNorm(5);
		r = f;	g = 0;	b = 255;
	}
	else if ( y <= 255*6 )
	{
		auto f = GetYNorm(6);
		r = 255;	g = 0;	b = 255-f;
	}
	else
	{
		r = 0;	g=0;	b=0;
	}
}

SoyPixels MakeRainbowPixels()
{
	//	this size is too big for h264, so section
	int SectionWidth = 16;
	SoyPixels Pixels(SoyPixelsMeta(6*SectionWidth,256,SoyPixelsFormat::RGB));
	auto Components = Pixels.GetMeta().GetChannels();
	auto& PixelsArray = Pixels.GetPixelsArray();
	for ( auto y=0;	y<Pixels.GetHeight();	y++ )
	{
		for ( auto x=0;	x<Pixels.GetWidth();	x++ )
		{
			int Section = x / SectionWidth;
			int Row = y + (256*Section);
			
			uint8_t r,g,b;
			GetRainbowRgb( Row, r, g, b );

			auto i = y*Pixels.GetWidth();
			i += x;
			i *= Components;
			PixelsArray[i+0] = r;
			PixelsArray[i+1] = g;
			PixelsArray[i+2] = b;
		}
	}
	return Pixels;
}

void CompareRainbow(const char* MetaJson,uint8_t* Plane0Data,uint8_t* Plane1Data,uint8_t* Plane2Data)
{
	auto Plane0Meta = GetPlaneMeta(MetaJson,0);
	auto Plane1Meta = GetPlaneMeta(MetaJson,0);
	auto Plane2Meta = GetPlaneMeta(MetaJson,0);
	auto Plane0DataSize = Plane0Meta.GetDataSize();
	auto Plane1DataSize = Plane1Meta.GetDataSize();
	auto Plane2DataSize = Plane2Meta.GetDataSize();
	SoyPixelsRemote Plane0( Plane0Data, Plane0DataSize, Plane0Meta );
	SoyPixelsRemote Plane1( Plane1Data, Plane1DataSize, Plane1Meta );
	SoyPixelsRemote Plane2( Plane2Data, Plane2DataSize, Plane2Meta );

	//	compare with original
	SoyPixels Original = MakeRainbowPixels();
	
	//	debug first column (for the greyscale test image)
	{
		std::stringstream Debug;
		auto Width = Plane0Meta.GetWidth();
		auto Height = Plane0Meta.GetHeight();
		auto Channels = Plane0Meta.GetChannels();
		for ( auto y=0;	y<Height;	y++ )
		{
			auto x = 0;
			auto Old = Original.GetPixel(x,y,0);
			auto New = Plane0.GetPixel(x,y,0);
			auto Diff = New-Old;
			Debug << static_cast<int>(Diff) << ' ';
		}
		DebugPrint(Debug.str());
	}
}

void MakeRainbowPng(const char* Filename)
{
	auto Pixels = MakeRainbowPixels();
	Array<uint8_t> PngData;
	TPng::GetPng( Pixels, GetArrayBridge(PngData), 0 );
	Soy::ArrayToFile( GetArrayBridge(PngData), Filename);
	Platform::ShowFileExplorer(Filename);
}
