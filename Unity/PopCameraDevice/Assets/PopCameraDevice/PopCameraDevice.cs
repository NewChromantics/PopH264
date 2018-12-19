using UnityEngine;
using System.Collections;					// required for Coroutines
using System.Runtime.InteropServices;		// required for DllImport
using System;								// requred for IntPtr
using System.Text;
using System.Collections.Generic;



/// <summary>
///	Low level interface
/// </summary>
public static class PopCameraDevice
{
	private const string PluginName = "PopCameraDevice";

	//	use byte as System.Char is a unicode char (2 bytes), then convert to Unicode Char
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void		EnumCameraDevices([In, Out] byte[] StringBuffer,int StringBufferLength);
	
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int		CreateCameraDevice(byte[] Name);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void		FreeCameraDevice(int Instance);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void		GetMeta(int Instance,int[] MetaValues,int MetaValuesCount);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int		PopFrame(int Instance,byte[] Plane0,int Plane0Size,byte[] Plane1,int Plane1Size,byte[] Plane2,int Plane2Size);



	private enum MetaIndex
	{
		PlaneCount = 0,

		Plane0_Width,
		Plane0_Height,
		Plane0_ComponentCount,
		Plane0_SoyPixelsFormat,
		Plane0_PixelDataSize,

		Plane1_Width,
		Plane1_Height,
		Plane1_ComponentCount,
		Plane1_SoyPixelsFormat,
		Plane1_PixelDataSize,

		Plane2_Width,
		Plane2_Height,
		Plane2_ComponentCount,
		Plane2_SoyPixelsFormat,
		Plane2_PixelDataSize,
	};

	public static List<string> EnumCameraDevices()
	{
		var StringBuffer = new byte[1000];
		EnumCameraDevices( StringBuffer, StringBuffer.Length );

		//	split strings
		var Delin = StringBuffer[0];
		string CurrentName = "";
		var Names = new List<string>();
		System.Action FinishCurrentName = ()=>
		{
			if ( String.IsNullOrEmpty(CurrentName) )
				return;
			Names.Add(CurrentName);
			CurrentName = "";
		};

		//	split at delin or when we hit a terminator
		for ( var i=1;	i<StringBuffer.Length;	i++ )
		{
			var Char8 = StringBuffer[i];
			if ( Char8 == '\0' )
			{
				FinishCurrentName();
				break;
			}
			if ( Char8 == Delin )
			{
				FinishCurrentName();
				continue;
			}
			var UnicodeChar = System.Convert.ToChar(Char8);
			CurrentName += UnicodeChar;
		}
		FinishCurrentName();

		return Names;
	}
	

	public class Device : IDisposable
	{
		int? Instance = null;

		public Device(string DeviceName)
		{
			var DeviceNameAscii = System.Text.ASCIIEncoding.ASCII.GetBytes(DeviceName);
			Instance = CreateCameraDevice(DeviceNameAscii);
			if ( Instance.Value <= 0 )
				throw new System.Exception("Failed to create Camera device with name " + DeviceName);
		}
		~Device()
		{
			Dispose();
		}

		public void Dispose()
		{
			if ( Instance.HasValue )
				FreeCameraDevice( Instance.Value );
			Instance = null;
		}

		TextureFormat GetTextureFormat(int ComponentCount)
		{
			switch(ComponentCount)
			{
				case 1:	return TextureFormat.R8;
				case 2:	return TextureFormat.RG16;
				case 3:	return TextureFormat.RGB24;
				case 4:	return TextureFormat.ARGB32;
				default:
					throw new System.Exception("Don't know what format to use for component count " + ComponentCount);
			}
		}
		void AllocTexture(ref Texture2D Plane,int Width, int Height,int ComponentCount)
		{
			var Format = GetTextureFormat( ComponentCount );
			if ( Plane != null )
			{
				if ( Plane.width != Width )
					Plane = null;
				else if ( Plane.height != Height )
					Plane = null;
				else if ( Plane.format != Format )
					Plane = null;
			}

			if ( !Plane )
			{
				var MipMap = false;
				Plane = new Texture2D( Width, Height, Format, MipMap );
			}
		}

		//	returns if changed
		public bool GetNextFrame(ref Texture2D Plane0,ref Texture2D Plane1,ref Texture2D Plane2)
		{
			var MetaValues = new int[100];
			GetMeta( Instance.Value, MetaValues, MetaValues.Length );
			var PlaneCount = MetaValues[(int)MetaIndex.PlaneCount];
			if ( PlaneCount <= 0 )
			{
				Debug.Log("No planes (" + PlaneCount +")");
				return false;
			}

			var Plane0Size = (PlaneCount >= 1) ? MetaValues[(int)MetaIndex.Plane0_PixelDataSize] : 0;
			var Plane1Size = (PlaneCount >= 2) ? MetaValues[(int)MetaIndex.Plane1_PixelDataSize] : 0;
			var Plane2Size = (PlaneCount >= 3) ? MetaValues[(int)MetaIndex.Plane2_PixelDataSize] : 0;

			//	alloc textures so we have data to write to
			if ( PlaneCount >= 1 )	AllocTexture( ref Plane0, MetaValues[(int)MetaIndex.Plane0_Width], MetaValues[(int)MetaIndex.Plane0_Height], MetaValues[(int)MetaIndex.Plane0_ComponentCount] );
			if ( PlaneCount >= 2 )	AllocTexture( ref Plane1, MetaValues[(int)MetaIndex.Plane1_Width], MetaValues[(int)MetaIndex.Plane1_Height], MetaValues[(int)MetaIndex.Plane1_ComponentCount] );
			if ( PlaneCount >= 3 )	AllocTexture( ref Plane2, MetaValues[(int)MetaIndex.Plane2_Width], MetaValues[(int)MetaIndex.Plane2_Height], MetaValues[(int)MetaIndex.Plane2_ComponentCount] );

			var UnusedData = new byte[1];
			var Plane0Data = Plane0 ? Plane0.GetRawTextureData() : UnusedData;
			var Plane1Data = Plane1 ? Plane1.GetRawTextureData() : UnusedData;
			var Plane2Data = Plane2 ? Plane2.GetRawTextureData() : UnusedData;

			var PopResult = PopFrame( Instance.Value, Plane0Data, Plane0Data.Length, Plane1Data, Plane1Data.Length, Plane2Data, Plane2Data.Length );
			if ( PopResult == 0 )
				return false;
			
			if ( Plane0 )
			{
				Plane0.LoadRawTextureData( Plane0Data );
				Plane0.Apply();
			}
			if ( Plane1 )
			{
				Plane1.LoadRawTextureData( Plane1Data );
				Plane1.Apply();
			}
			if ( Plane2 )
			{
				Plane2.LoadRawTextureData( Plane2Data );
				Plane2.Apply();
			}
			return true;
		}

	}
}

