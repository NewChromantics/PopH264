#include "SoyPixels.h"
#include "SoyPng.h"
#include "SoyFileSystem.h"
#include "json11.hpp"

extern void DebugPrint(const std::string& Message);


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

void CompareGreyscale(const char* MetaJson,uint8_t* Plane0Data)
{
	auto Plane0Meta = GetPlaneMeta(MetaJson,0);
	auto Plane0DataSize = Plane0Meta.GetDataSize();
	SoyPixelsRemote Plane0( Plane0Data, Plane0DataSize, Plane0Meta );
	
	//	compare with original
	SoyPixels Greyscale = MakeGreyscalePixels();

	//	debug first column (for the greyscale test image)
	{
		std::stringstream Debug;
		auto Width = Plane0Meta.GetWidth();
		auto Height = Plane0Meta.GetHeight();
		auto Channels = Plane0Meta.GetChannels();
		for ( auto y=0;	y<Height;	y++ )
		{
			auto x = 0;
			auto Old = Greyscale.GetPixel(x,y,0);
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

