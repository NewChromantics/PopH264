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

	#if UNITY_IOS && !UNITY_EDITOR_OSX && !UNITY_EDITOR_WIN
	private const string PluginName = "__Internal";
	#else
	private const string PluginName = "PopEncodeJpeg";
	#endif

	[DllImport (PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern System.IntPtr	PopDebugString();

	[DllImport (PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void		ReleaseDebugString(System.IntPtr String);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int		EncodeJpeg (byte[] JpegData, int JpegDataSize, int JpegQuality, byte[] ImageData, int ImageDataSize, int ImageWidth, int ImageHeight, int ImageComponents);



	public static byte[] EncodeToJpeg(Texture2D Image)
	{
		var Width = Image.width;
		var Height = Image.height;
		var Pixels = Image.GetPixels32 ();
		var ComponentCount = 4;

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
		var JpegData = new byte[ComponentCount * Width * Height];
		int Quality = 1;
		var BytesWritten = EncodeJpeg (JpegData, JpegData.Length, Quality, PixelBytes, PixelBytes.Length, Width, Height, ComponentCount);
		if (BytesWritten > JpegData.Length)
			throw new System.Exception ("Didn't allocate enough bytes for JPEG. " + JpegData.Length + "/" + BytesWritten);

		var ShrunkJpegData = new Byte[BytesWritten];
		for (int i = 0;	i < ShrunkJpegData.Length;	i++)
			ShrunkJpegData [i] = JpegData [i];

		return ShrunkJpegData;
	}


}
