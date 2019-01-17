using UnityEngine;
using System.Collections;					// required for Coroutines
using System.Runtime.InteropServices;		// required for DllImport
using System;								// requred for IntPtr
using System.Text;
using System.Collections.Generic;



/// <summary>
///	Low level interface
/// </summary>
public static class PopH264
{
	private const string PluginName = "PopH264";

	//	use byte as System.Char is a unicode char (2 bytes), then convert to Unicode Char
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	public static extern int		GetTestInteger();
	
}

