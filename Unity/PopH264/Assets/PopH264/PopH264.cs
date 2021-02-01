#define POPH264_AS_PACKAGE	//	disable this when testing Poph264 builds (which are placed in /Assets/PopH264)
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
#if UNITY_UWP
	private const string PluginName = "PopH264.Uwp";
#error building uwp
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
#if POPH264_AS_PACKAGE
	private const string PluginFrameworkPath = "Packages/com.newchromantics.poph264/PopH264.xcframework/macos-x86_64/";
#else
	// universal xcframework
	//private const string PluginFrameworkPath = "Assets/PopH264/PopH264.xcframework/macos-x86_64/";
	private const string PluginFrameworkPath = "Assets/PopH264/";
#endif
	private const string PluginExecutable = "PopH264_Osx.framework/PopH264_Osx";
	//private const string PluginExecutable = "PopH264_Osx.framework/Versions/A/PopH264_Osx";
	private const string PluginName = PluginFrameworkPath+PluginExecutable;
#elif UNITY_IPHONE
	[DllImport("__Internal")]
#else
	private const string PluginName = "PopH264";
#endif
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_GetVersion();

    //  returns decoder instance id, 0 on error.
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_CreateDecoder(byte[] OptionsJson, [In, Out] byte[] ErrorBuffer, Int32 ErrorBufferLength);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void PopH264_DestroyDecoder(int Instance);

    //  returns 0 on success or -1 on error
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PushData(int Instance,byte[] Data,int DataSize,int FrameNumber);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void	PopH264_PeekFrame(int Instance, byte[] JsonBuffer, int JsonBufferSize);

    //  returns frame number or -1
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PopFrame(int Instance,byte[] Plane0,int Plane0Size,byte[] Plane1,int Plane1Size,byte[] Plane2,int Plane2Size);


	//	gr: these numbers don't matter in PopH264, need a better way to map these across depedencies
	//		other than matching strings
	//	for use with PopYuv shader, these enum values should match the shader
	public enum PixelFormat
	{
		Debug			=999,
		Invalid			=0,
		Greyscale		=1,
		RGB				=2,
		RGBA			=3,
		BGRA			=4,
		BGR				=5,
		YYuv_8888_Full	=6,
		YYuv_8888_Ntsc	=7,
		Depth16mm		=8,
		Chroma_U		=9,
		Chroma_V		=10,
		ChromaUV_88		=11,
		ChromaVU_88		=12,
		Luma_Ntsc		=13,


		ChromaU_8 = Chroma_U,
		ChromaV_8 = Chroma_V,
	}


	[System.Serializable]
	public struct PlaneMeta
	{
		public PixelFormat		PixelFormat { get { return (PixelFormat)Enum.Parse(typeof(PixelFormat), Format); } }
		public string			Format;
		public int				Width;
		public int				Height;
		public int				DataSize;
		public int				Channels;
	};

	[System.Serializable]
	public struct FrameMeta
	{
		public List<PlaneMeta>	Planes;
		public int				PlaneCount { get { return Planes!=null ? Planes.Count : 0; } }

		public bool				EndOfStream;
		public int 				FrameNumber;
		public int				FramesQueued;
	};
	
	static public string GetString(byte[] Ascii)
	{
		var String = System.Text.ASCIIEncoding.ASCII.GetString(Ascii);
		var TerminatorPos = String.IndexOf('\0');
		if (TerminatorPos >= 0)
			String = String.Substring(0, TerminatorPos);
		return String;
	}

	public struct FrameInput
	{
		public byte[] Bytes;
		public int FrameNumber;
		public bool EndOfStream { get { return Bytes == null; } }	//	marker/command to tell decoder stream has ended
	};

	[System.Serializable]
	public struct DecoderParams
	{
		//	Avf, Broadway, MediaFoundation, MagicLeap, Intel etc
		//	empty string defaults to "best" (hardware where possible)
		//	todo: PopH264_EnumDecoders which will return a list of all possible decoders
		//	ie. low level specific decoders/codecs installed on the system, including say MediaFoundation_NvidiaHardwareH264, or MagicLeap_GoogleSoftware
		public string Decoder;
		
		//	print extra debug info (all decoders)
		public bool VerboseDebug;
	};

	public class Decoder : IDisposable
	{
		int? Instance = null;

		//	cache once to avoid allocating each frame
		List<byte[]> PlaneCaches;
		byte[] UnusedBuffer = new byte[1];
		bool ThreadedDecoding = true;
		System.Threading.Thread InputThread;
		List<FrameInput> InputQueue;
		int? InputThreadResult = 0;
		public bool HadEndOfStream = false;
		
		//	reuse/alloc once a json buffer
		byte[] JsonBufferPrealloc;
		byte[] JsonBuffer
		{
			get
			{
				if (JsonBufferPrealloc == null)
					JsonBufferPrealloc = new byte[1000];
				return JsonBufferPrealloc;
			}
		}

		public Decoder(DecoderParams? DecoderParams,bool ThreadedDecoding)
		{
			//	show version on first call
			var Version = PopH264_GetVersion();
			Debug.Log("PopH264 version " + Version);
			
			this.ThreadedDecoding = ThreadedDecoding;

			//	alloc defaults
			if (!DecoderParams.HasValue)
				DecoderParams = new DecoderParams();

			var ParamsJson = JsonUtility.ToJson(DecoderParams.Value);
			var ParamsJsonAscii = System.Text.ASCIIEncoding.ASCII.GetBytes(ParamsJson + "\0");
			var ErrorBuffer = new byte[200];
			Instance = PopH264_CreateDecoder(ParamsJsonAscii, ErrorBuffer, ErrorBuffer.Length);
			var Error = GetString(ErrorBuffer);
			if (Instance.Value <= 0)
				throw new System.Exception("Failed to create decoder instance;" + Error);
			if (!String.IsNullOrEmpty(Error))
			{
				Debug.LogWarning("Created PopH264 decoder (" + Instance.Value + ") but error was not empty (length = " + Error.Length + ") " + Error);
			}
		}
		~Decoder()
		{
			Dispose();
		}

		public void Dispose()
		{
			//	stop thread before killing decoder
			InputQueue = null;
			if (InputThread != null)
			{
				//	I think we can safely abort, might need to check. If we don't, depending on how much data we've thrown at the decoder, this could take ages to finish
				InputThread.Abort();
				InputThread.Join();
				InputThread = null;
			}

			if (Instance.HasValue)
				PopH264_DestroyDecoder(Instance.Value);
			Instance = null;
		}

		TextureFormat GetTextureFormat(int ComponentCount)
		{
			switch (ComponentCount)
			{
				case 1: return TextureFormat.R8;
				case 2: return TextureFormat.RG16;
				case 3: return TextureFormat.RGB24;
				case 4: return TextureFormat.RGBA32;
				default:
					throw new System.Exception("Don't know what format to use for component count " + ComponentCount);
			}
		}

		Texture2D AllocTexture(Texture2D Plane,PlaneMeta Meta)
		{
			var Format = GetTextureFormat(Meta.Channels);
			if (Plane != null)
			{
				if (Plane.width != Meta.Width)
					Plane = null;
				else if (Plane.height != Meta.Height)
					Plane = null;
				else if (Plane.format != Format)
					Plane = null;
			}

			if (!Plane)
			{
				var MipMap = false;
				var Linear = true;
				try
				{
					Plane = new Texture2D(Meta.Width, Meta.Height, Format, MipMap, Linear);
					Plane.filterMode = FilterMode.Point;
				}
				catch(System.Exception e)
				{
					Debug.LogError("Create texture2d(" + Meta.Width + "," + Meta.Height + " " + Format + ")");
					throw e;
				}
			}

			return Plane;
		}

		void AllocListToSize<T>(ref List<T> Array, int Size)
		{
			if (Array == null)
				Array = new List<T>();
			while (Array.Count < Size)
				Array.Add(default(T));
		}

		void ThreadPushQueue()
		{
			while (InputQueue != null)
			{
				if (InputQueue.Count == 0)
				{
					System.Threading.Thread.Sleep(100);
					//	make thread idle properly
					//PushByteThread.Suspend();
					continue;
				}

				//	pop off the data
				FrameInput Frame;
				lock (InputQueue)
				{
					Frame = InputQueue[0];
					InputQueue.RemoveRange(0, 1);
				}
				var Length = (Frame.Bytes == null) ? 0 : Frame.Bytes.Length;
				InputThreadResult = PopH264_PushData(Instance.Value, Frame.Bytes, Length, Frame.FrameNumber );
			}
		}

		void CheckH264Frame(FrameInput Frame)
		{
			//	if we're getting raw fragmented packets (eg. from udp)
			//	then the packets may not be real frames. 
			//	maybe don't need to waste time checking any more, but certainly skip ultra small ones
			if (Frame.Bytes.Length < 4)
				return;
			
			/*	gr: removed this check for now to remove dependencies
			try
			{
				var NaluHeaderLength = PopX.H264.GetNaluHeaderSize(Frame.Bytes);
				var PacketType = PopX.H264.GetNaluType(Frame.Bytes[NaluHeaderLength]);
				if ( PacketType == PopX.H264.NaluType.SPS )
				{
					var HeaderBytes = Frame.Bytes.SubArray(NaluHeaderLength, Frame.Bytes.Length - NaluHeaderLength);
					var Header = PopX.H264.ParseAvccProfile(HeaderBytes);
					if ( Header.Profile!=PopX.H264.Profile.Baseline || Header.Level > 3 )
					{
						Debug.LogWarning("H264 SPS version " + Header.Profile + " " + Header.Level + " higher than supported (Baseline 3.0)"); 
					}
				}
			}
			catch(System.Exception e)
			{
				Debug.LogException(e);
			}
			*/
		}

		public int PushFrameData(FrameInput Frame)
		{
			//CheckH264Frame(Frame);
			//Debug.Log(BitConverter.ToString(Frame.Bytes.SubArray(0, 8)));

			if ( !ThreadedDecoding )
			{
				var Length = (Frame.Bytes==null) ? 0 : Frame.Bytes.Length;
				return PopH264_PushData(Instance.Value, Frame.Bytes, Length, Frame.FrameNumber);
			}

			if (InputThread == null )
			{
				InputQueue = new List<FrameInput>();
				InputThread = new System.Threading.Thread(new System.Threading.ThreadStart(ThreadPushQueue));
				InputThread.Start();
			}

			//	add data and wake up the thread in case we need to
			lock (InputQueue)
			{
				InputQueue.Add(Frame);
				//PushByteThread.Resume();
			}

			//	check for 
			return InputThreadResult.HasValue ? InputThreadResult.Value : 0;
		}

		public int PushFrameData(byte[] H264Data, int FrameNumber)
		{
			var NewFrame = new FrameInput();
			NewFrame.FrameNumber = FrameNumber;
			NewFrame.Bytes = H264Data;
			return PushFrameData(NewFrame);
		}

		//	returns frame time
		public int? GetNextFrame(ref List<Texture2D> Planes, ref List<PixelFormat> PixelFormats)
		{
			PopH264_PeekFrame(Instance.Value, JsonBuffer, JsonBuffer.Length);
			var Json = GetString(JsonBuffer);
			var Meta = JsonUtility.FromJson<FrameMeta>(Json);
			var PlaneCount = Meta.PlaneCount;

			if (Meta.EndOfStream)
				HadEndOfStream = true;

			//Debug.Log("Meta " + Json);
			if (PlaneCount <= 0)
			{
				//Debug.Log("No planes (" + PlaneCount +")");
				PixelFormats = null;
				return null;
			}

			//	not going to extract a new frame, so skip buffer/texture allocations
			if (Meta.FrameNumber <0)
				return null;
				
			AllocListToSize(ref Planes, PlaneCount);
			AllocListToSize(ref PixelFormats, PlaneCount);
			AllocListToSize(ref PlaneCaches, PlaneCount);
	
			if (PlaneCount >= 1) PixelFormats[0] = Meta.Planes[0].PixelFormat;
			if (PlaneCount >= 2) PixelFormats[1] = Meta.Planes[1].PixelFormat;
			if (PlaneCount >= 3) PixelFormats[2] = Meta.Planes[2].PixelFormat;

			//	alloc textures so we have data to write to
			if (PlaneCount >= 1) Planes[0] = AllocTexture(Planes[0], Meta.Planes[0]);
			if (PlaneCount >= 2) Planes[1] = AllocTexture(Planes[1], Meta.Planes[1]);
			if (PlaneCount >= 3) Planes[2] = AllocTexture(Planes[2], Meta.Planes[2]);

			for (var p = 0; p < PlaneCount; p++)
			{
				if (PlaneCaches[p] != null)
					continue;
				if (!Planes[p])
					continue;
				PlaneCaches[p] = Planes[p].GetRawTextureData();
			}

			//	read frame bytes
			var Plane0Data = (PlaneCaches.Count >= 1 && PlaneCaches[0] != null) ? PlaneCaches[0] : UnusedBuffer;
			var Plane1Data = (PlaneCaches.Count >= 2 && PlaneCaches[1] != null) ? PlaneCaches[1] : UnusedBuffer;
			var Plane2Data = (PlaneCaches.Count >= 3 && PlaneCaches[2] != null) ? PlaneCaches[2] : UnusedBuffer;
			var PopResult = PopH264_PopFrame(Instance.Value, Plane0Data, Plane0Data.Length, Plane1Data, Plane1Data.Length, Plane2Data, Plane2Data.Length);
			if (PopResult < 0)
			{
				//Debug.Log("PopFrame returned " + PopResult);
				return null;
			}

			//	update texture
			for (var p = 0; p < PlaneCount; p++)
			{
				if (!Planes[p])
					continue;

				Planes[p].LoadRawTextureData(PlaneCaches[p]);
				Planes[p].Apply();
			}

			return PopResult;
		}

	}
}

