using UnityEngine;
using System.Collections;					// required for Coroutines
using System.Runtime.InteropServices;		// required for DllImport
using System;								// requred for IntPtr
using System.Text;
using System.Collections.Generic;
using PopX;     //	for PopX.PixelFormat, replace this and provide your own pixelformat if you want to remove the dependency



/// <summary>
///	Low level interface
/// </summary>
public static class PopH264
{
#if UNITY_UWP
	private const string PluginName = "PopH264.Uwp";
#error building uwp
#elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
	private const string PluginName = "Assets/PopH264/PopH264_Osx.framework/Versions/A/PopH264_Osx";
#elif UNITY_IPHONE
	[DllImport("__Internal")]
#else
	private const string PluginName = "PopH264";
#endif
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_GetVersion();

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_CreateInstance(int Mode);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void	PopH264_DestroyInstance(int Instance);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PushData(int Instance,byte[] Data,int DataSize,int FrameNumber);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void	PopH264_PeekFrame(int Instance, byte[] JsonBuffer, int JsonBufferSize);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PopFrame(int Instance,byte[] Plane0,int Plane0Size,byte[] Plane1,int Plane1Size,byte[] Plane2,int Plane2Size);

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
	};

	public enum DecoderMode
	{
		Software = 0,
		MagicLeap_NvidiaSoftware = 1,
		MagicLeap_GoogleSoftware = 2,
		MagicLeap_NvidiaHardware = 3,
		MagicLeap_GoogleHardware = 4,
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

		public Decoder(DecoderMode DecoderMode,bool ThreadedDecoding)
		{
			//	show version on first call
			var Version = PopH264_GetVersion();
			Debug.Log("PopH264 version " + Version);
			
			this.ThreadedDecoding = ThreadedDecoding;
			int Mode = (int)DecoderMode;
			Instance = PopH264_CreateInstance(Mode);
			if (Instance.Value <= 0)
				throw new System.Exception("Failed to create decoder instance");
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
				PopH264_DestroyInstance(Instance.Value);
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
				Plane = new Texture2D(Meta.Width, Meta.Height, Format, MipMap,Linear);
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
				InputThreadResult = PopH264_PushData(Instance.Value, Frame.Bytes, Frame.Bytes.Length, Frame.FrameNumber );
			}
		}

		void CheckH264Frame(FrameInput Frame)
		{
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
		}

		public int PushFrameData(FrameInput Frame)
		{
			CheckH264Frame(Frame);

			if ( !ThreadedDecoding )
			{
				return PopH264_PushData(Instance.Value, Frame.Bytes, Frame.Bytes.Length, Frame.FrameNumber);
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
			var JsonBuffer = new Byte[1000];
			PopH264_PeekFrame(Instance.Value, JsonBuffer, JsonBuffer.Length);
			var Json = GetString(JsonBuffer);
			var Meta = JsonUtility.FromJson<FrameMeta>(Json);
			var PlaneCount = Meta.PlaneCount;

			if (PlaneCount <= 0)
			{
				//Debug.Log("No planes (" + PlaneCount +")");
				PixelFormats = null;
				return null;
			}

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

