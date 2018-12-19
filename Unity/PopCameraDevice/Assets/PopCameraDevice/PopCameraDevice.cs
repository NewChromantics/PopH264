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
	

}
