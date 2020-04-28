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
	auto Plane1Meta = GetPlaneMeta(MetaJson,1);
	auto Plane2Meta = GetPlaneMeta(MetaJson,2);
	auto Plane0DataSize = Plane0Meta.GetDataSize();
	auto Plane1DataSize = Plane1Meta.GetDataSize();
	auto Plane2DataSize = Plane2Meta.GetDataSize();
	SoyPixelsRemote Plane0( Plane0Data, Plane0DataSize, Plane0Meta );
	SoyPixelsRemote Plane1( Plane1Data, Plane1DataSize, Plane1Meta );
	SoyPixelsRemote Plane2( Plane2Data, Plane2DataSize, Plane2Meta );

	//	compare with original
	SoyPixels Original = MakeRainbowPixels();
	SoyPixels Converted = Original;

	auto GetYuv = [&](int x,int y,uint8_t& Luma,uint8_t& ChromaU,uint8_t& ChromaV)
	{
		Luma = Plane0.GetPixel(x,y,0);
		ChromaU = Plane1.GetPixel(x/2,y/2,0);
		ChromaV = Plane2.GetPixel(x/2,y/2,0);
	};
	auto YuvToRgb = [&](int x,int y,int& r,int& g,int& b)
	{
		uint8_t l,u,v;
		GetYuv( x,y,l,u,v );
		
		//	convert
		float ChromaU = (u/255.f) - 0.5f;
		float ChromaV = (v/255.f) - 0.5f;
		float Luma = (l/255.f);
		float ChromaVRed = 1.5958f;
		float ChromaUGreen = -0.39173f;
		float ChromaVGreen = -0.81290f;
		float ChromaUBlue = 2.017f;
		float rf = Luma + (ChromaVRed * ChromaV);
		float gf = Luma + (ChromaUGreen * ChromaU) + (ChromaVGreen * ChromaV);
		float bf = Luma + (ChromaUBlue * ChromaU);
		r = rf * 255;
		g = gf * 255;
		b = bf * 255;
	};
	
	//	debug first column (for the greyscale test image)
	{
		int SectionWidth = 16;
		std::stringstream Debug;
		auto Width = Plane0Meta.GetWidth();
		auto Height = Plane0Meta.GetHeight();
		auto Channels = Plane0Meta.GetChannels();
		int TotalDiff[3] = {0,0,0};
		int MinDiff[3] = {999,999,999};
		int MaxDiff[3] = {-999,-999,-999};
		int DiffCount = 0;
		for ( auto y=0;	y<Height;	y++ )
		{
			for ( auto s=0;	s<Width/SectionWidth;	s++ )
			{
				auto x = s*SectionWidth;
				x += SectionWidth/2;	//	sample from middle
				int Old[3];
				int New[3];
				Old[0] = Original.GetPixel(x,y,0);
				Old[1] = Original.GetPixel(x,y,1);
				Old[2] = Original.GetPixel(x,y,2);
				YuvToRgb( x,y,New[0], New[1], New[2] );
				
				std::clamp(New[0],0,255);
				std::clamp(New[1],0,255);
				std::clamp(New[2],0,255);
				
				auto Diffr = (New[0]-Old[0]);
				auto Diffg = (New[1]-Old[1]);
				auto Diffb = (New[2]-Old[2]);
				
				//	ignore outliers
				auto OutlierDiff = 30;
				if ( abs(Diffr) > OutlierDiff || abs(Diffg) > OutlierDiff || abs(Diffb) > OutlierDiff )
				{
			
				}
				else
				{
					TotalDiff[0] += Diffr;
					TotalDiff[1] += Diffg;
					TotalDiff[2] += Diffb;
					MinDiff[0] = std::min(MinDiff[0],Diffr);
					MinDiff[1] = std::min(MinDiff[1],Diffg);
					MinDiff[2] = std::min(MinDiff[2],Diffb);
					MaxDiff[0] = std::max(MaxDiff[0],Diffr);
					MaxDiff[1] = std::max(MaxDiff[1],Diffg);
					MaxDiff[2] = std::max(MaxDiff[2],Diffb);
				}
			
				DiffCount++;
				Debug << " " << Diffr << "," << Diffg << "," << Diffb << " ";
				
				//	write the converted for testing
				for ( auto sx=0;	sx<SectionWidth;	sx++ )
				{
					int ox = (s*SectionWidth) + sx;
					std::clamp(New[0],0,255);
					std::clamp(New[1],0,255);
					std::clamp(New[2],0,255);
					Converted.SetPixel(ox, y, 0, New[0] );
					Converted.SetPixel(ox, y, 1, New[1] );
					Converted.SetPixel(ox, y, 2, New[2] );
				}
			}
		}
		float AverageDiff[3];
		AverageDiff[0] = TotalDiff[0] / static_cast<float>(DiffCount);
		AverageDiff[1] = TotalDiff[1] / static_cast<float>(DiffCount);
		AverageDiff[2] = TotalDiff[2] / static_cast<float>(DiffCount);
		Debug << std::endl << std::endl;
		Debug << "Average diff " << AverageDiff[0] << "," << AverageDiff[1] << "," << AverageDiff[2] << std::endl;
		Debug << "Min diff " << MinDiff[0] << "," << MinDiff[1] << "," << MinDiff[2] << std::endl;
		Debug << "Max diff " << MaxDiff[0] << "," << MaxDiff[1] << "," << MaxDiff[2] << std::endl;

		DebugPrint(Debug.str());
		
	}
	
	Array<uint8_t> PngData;
	TPng::GetPng( Converted, GetArrayBridge(PngData), 0 );
	auto* Filename ="RainbowConvert.png";
	Soy::ArrayToFile( GetArrayBridge(PngData), Filename);
	Platform::ShowFileExplorer(Filename);

}

void MakeRainbowPng(const char* Filename)
{
	auto Pixels = MakeRainbowPixels();
	Array<uint8_t> PngData;
	TPng::GetPng( Pixels, GetArrayBridge(PngData), 0 );
	Soy::ArrayToFile( GetArrayBridge(PngData), Filename);
	Platform::ShowFileExplorer(Filename);
}
