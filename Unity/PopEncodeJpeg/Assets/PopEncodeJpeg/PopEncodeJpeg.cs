using UnityEngine;
using System.Collections;					// required for Coroutines
using System.Runtime.InteropServices;		// required for DllImport
using System;								// requred for IntPtr
using System.Text;
using System.Collections.Generic;



/// <summary>
///	Low level interface
/// </summary>
public static class PopEncodeJpeg {

	private const string PluginName = "PopEncodeJpeg";

	[DllImport (PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern System.IntPtr	PopDebugString();

	[DllImport (PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void		ReleaseDebugString(System.IntPtr String);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int		EncodeJpeg (byte[] JpegData, int JpegDataSize, int JpegQuality, byte[] ImageData, int ImageDataSize, int ImageWidth, int ImageHeight, int ImageComponents,int IsRgb);


	public static void EncodeToJpeg(Texture2D Image,ref byte[] JpegData,ref int JpegDataSize)
	{
		var Width = Image.width;
		var Height = Image.height;
		var Pixels = Image.GetPixels32 ();
		var ComponentCount = 4;
		var Rgb = true;

		var PixelBytes = new byte[ComponentCount * Width * Height];
		for (int i = 0;	i < Pixels.Length;	i++) {
			var Colour = Pixels [i];
			int p = i * ComponentCount;
			PixelBytes [p + 0] = Colour.r;
			PixelBytes [p + 1] = Colour.g;
			PixelBytes [p + 2] = Colour.b;
			PixelBytes [p + 3] = Colour.a;
		}

		//	try and encode, returns number of bytes used. if the number is bigger than allocated, we need a bigger buffer
		if (JpegData == null) {
			JpegData = new byte[ComponentCount * Width * Height];
		}
		int Quality = 1;
		JpegDataSize = EncodeJpeg (JpegData, JpegData.Length, Quality, PixelBytes, PixelBytes.Length, Width, Height, ComponentCount, Rgb?1:0 );
		if (JpegDataSize > JpegData.Length)
			throw new System.Exception ("Didn't allocate enough bytes for JPEG. " + JpegData.Length + "/" + JpegDataSize);
	}

	public static void EncodeToJpeg(byte[] PixelBytes,int Width,int Height,int ComponentCount,bool Rgb,ref byte[] JpegData,ref int JpegDataSize)
	{
		//	try and encode, returns number of bytes used. if the number is bigger than allocated, we need a bigger buffer
		if (JpegData == null) {
			JpegData = new byte[ComponentCount * Width * Height];
		}
		int Quality = 1;
		JpegDataSize = EncodeJpeg (JpegData, JpegData.Length, Quality, PixelBytes, PixelBytes.Length, Width, Height, ComponentCount, Rgb?1:0 );
		if (JpegDataSize > JpegData.Length)
			throw new System.Exception ("Didn't allocate enough bytes for JPEG. " + JpegData.Length + "/" + JpegDataSize);
	}

	

	public static byte[] EncodeToJpeg(byte[] PixelBytes,int Width,int Height,int ComponentCount,bool Rgb)
	{
		byte[] JpegData = null;
		int JpegDataSize = 0;
		EncodeToJpeg (PixelBytes, Width, Height,ComponentCount, Rgb, ref JpegData, ref JpegDataSize);

		var ShrunkJpegData = new Byte[JpegDataSize];
		for (int i = 0;	i < ShrunkJpegData.Length;	i++)
			ShrunkJpegData [i] = JpegData [i];

		return ShrunkJpegData;
	}

	public static byte[] EncodeToJpeg(Texture2D Image)
	{
		byte[] JpegData = null;
		int JpegDataSize = 0;
		EncodeToJpeg (Image, ref JpegData, ref JpegDataSize);

		var ShrunkJpegData = new Byte[JpegDataSize];
		for (int i = 0;	i < ShrunkJpegData.Length;	i++)
			ShrunkJpegData [i] = JpegData [i];

		return ShrunkJpegData;
	}


}
