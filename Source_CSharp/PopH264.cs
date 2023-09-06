#if UNITY_EDITOR_WIN
	//	not supported
#elif UNITY_EDITOR_OSX
	#define ENCODER_SUPPORTS_RGBA_INPUT
#elif UNITY_IPHONE || UNITY_STANDALONE_OSX
	#define ENCODER_SUPPORTS_RGBA_INPUT
#else
	//	not supported
#endif


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
//	need to place editor defines first as we can be in mac editor, but windows target.
#if UNITY_EDITOR_WIN
	private const string PluginName = "PopH264";	//	PopH264.dll
#elif UNITY_EDITOR_OSX
	private const string PluginName = "PopH264";	//	libPopH264.dylib
#elif UNITY_STANDALONE_OSX
	private const string PluginName = "PopH264";	//	libPopH264.dylib
#elif UNITY_WSA
	private const string PluginName = "PopH264.Uwp.dll";	//	PopH264.Uwp.dll
#elif UNITY_IPHONE
	private const string PluginName = "__Internal";
#else
	private const string PluginName = "PopH264";
#endif
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_GetVersion();

	//	returns decoder instance id, 0 on error.
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_CreateDecoder(byte[] OptionsJson, [In, Out] byte[] ErrorBuffer, Int32 ErrorBufferLength);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void PopH264_DestroyDecoder(int Instance);

	//	returns 0 on success or -1 on error
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PushData(int Instance,byte[] Data,int DataSize,int FrameNumber);

	//	wrapper for PopH264_PushData(null) - send an EndOFStream packet to the decoder to make it finish off any unprocessed packets
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PushEndOfStream(int Instance);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void	PopH264_PeekFrame(int Instance, byte[] JsonBuffer, int JsonBufferSize);

	//	returns frame number or -1
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int	PopH264_PopFrame(int Instance,byte[] Plane0,int Plane0Size,byte[] Plane1,int Plane1Size,byte[] Plane2,int Plane2Size);




	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int			PopH264_CreateEncoder(byte[] OptionsJson,byte[] ErrorBuffer,int ErrorBufferSize);
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void			PopH264_DestroyEncoder(int Instance);
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void			PopH264_EncoderPushFrame(int Instance,byte[] MetaJson,byte[] LumaData,byte[] ChromaUData,byte[] ChromaVData,byte[] ErrorBuffer,int ErrorBufferSize);
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void			PopH264_EncoderEndOfStream(int Instance);
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int			PopH264_EncoderPopData(int Instance,byte[] DataBuffer,int DataBufferSize);
	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern void			PopH264_EncoderPeekData(int Instance,byte[] MetaJsonBuffer,int MetaJsonBufferSize);

	[DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
	private static extern int			PopH264_GetTestData(byte[] Name,byte[] Buffer,int BufferSize);



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
		YYuv_8888_Full	=6,	//	poph264 doesnt like this as an input
		YYuv_8888_Ntsc	=7,
		Depth16mm		=8,
		Chroma_U		=9,
		Chroma_V		=10,
		ChromaUV_88		=11,
		ChromaVU_88		=12,
		Luma_Ntsc		=13,

		Yuv_8_8_8 = 100,
		Yuv_8_88 = 101,


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

		public string			Error;
		public string			Decoder;				//	internal name of codec (if provided by API/OS)
		public bool				HardwareAccelerated;	//	are we using a hardware accelerated decoder. DO NOT rely on this information as if not provided cs defaults to false. Currently MediaFoundation only
		public bool				EndOfStream;
		public int 				FrameNumber;
		public int				FramesQueued;	//	number of subsequent frames already decoded and buffered up

        // optional meta output by decoder
		public int				Rotation;   //  clockwise rotation in degrees
		public string			YuvColourMatrixName;    //	todo: enum this
		public int				AverageBitsPerSecondRate;
		public int				RowStrideBytes;
		public bool				Flipped;
		public int				ImageWidth;
		public int				ImageHeight;
		public int[]			ImageRect;		//	some decoders will output an image aligned to say, 16 (macro blocks, or byte alignment etc) If the image is padded, we should have a [x,y,w,h] array here
	};
	
	//	if we make this public for re-use, give it a name that doesn't suggest this is an API function
	//	make sure we use this! using GetString without all 0's will crash unity as console tries to print it
	static private string GetString(byte[] Ascii)
	{
		if ( Ascii[0] == 0 )
			return null;
		var String = System.Text.ASCIIEncoding.ASCII.GetString(Ascii);
		
		//	clip string as unity doesn't cope well with large terminator strings
		var TerminatorPos = String.IndexOf('\0');
		if (TerminatorPos >= 0)
			String = String.Substring(0, TerminatorPos);
		return String;
	}


	//	nice wrappers for raw CAPI calls
	static public string		GetVersion()
	{
		var Version = PopH264_GetVersion();
		var Major = (Version / (100 * 100000));
		var Minor = (Version / 100000) % 100;
		var Patch = (Version) % 100000;
		return $"{Major}.{Minor}.{Patch}";
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

		public bool AllowBuffering;			//	by default poph264 tries to reduce amount of buffering decoders do and deliver frames ASAP
		public bool DoubleDecodeKeyframe;	//	Hack for broadway & MediaFoundation, process a keyframe twice to instantly decode instead of buffering
		public bool DrainOnKeyframe;		//	Debug for MediaFoundation, trigger drain command on a keyfrae
		public bool LowPowerMode;
		public bool DropBadFrames;
		public bool DecodeSei;
		
		//	gr: only set these for testing. 0 means no hint will be set
		//public int	Width;
		//public int	Height;
		//public int	InputSize;
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
		bool? InputThreadResult = null;
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
			Debug.Log($"PopH264 version {GetVersion()}");
			
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
				var Result = PopH264_PushData(Instance.Value, Frame.Bytes, Length, Frame.FrameNumber);
				InputThreadResult = (Result==0);
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

		public bool PushFrameData(FrameInput Frame)
		{
			//CheckH264Frame(Frame);
			//Debug.Log(BitConverter.ToString(Frame.Bytes.SubArray(0, 8)));

			if ( !ThreadedDecoding )
			{
				var Length = (Frame.Bytes==null) ? 0 : Frame.Bytes.Length;
				var Result = PopH264_PushData(Instance.Value, Frame.Bytes, Length, Frame.FrameNumber);
				return (Result == 0);
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

			//	look out for previous errors
			return InputThreadResult.HasValue ? InputThreadResult.Value : true;
		}

		public bool PushFrameData(byte[] H264Data, int FrameNumber)
		{
			var NewFrame = new FrameInput();
			NewFrame.FrameNumber = FrameNumber;
			NewFrame.Bytes = H264Data;
			return PushFrameData(NewFrame);
		}

		public bool PushEndOfStream()
		{
			var NewFrame = new FrameInput();
			return PushFrameData(NewFrame);
		}

		public FrameMeta? GetNextFrameAndMeta(ref List<Texture2D> Planes, ref List<PixelFormat> PixelFormats)
		{
			PopH264_PeekFrame(Instance.Value, JsonBuffer, JsonBuffer.Length);
			var Json = GetString(JsonBuffer);
			//Debug.Log("PopH264 frame meta: " + Json);
			var Meta = JsonUtility.FromJson<FrameMeta>(Json);
			var PlaneCount = Meta.PlaneCount;

			//	an error has been reported
			//	gr: if there is no frame, we can assume it's a fatal error. but that is currently ambiguious against frame0 specific error
			//	update this handling based on user feedback!
			if ( !String.IsNullOrEmpty(Meta.Error) )
				throw new System.Exception("PopH264 decode error: "+Meta.Error);

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

			//	gr: this shouldn't have changed. But just in case...
			Meta.FrameNumber = PopResult;
			return Meta;
		}

		//	old interface. May be deprecated for GetNextFrameAndMeta entirely as the ImageRect can't really be ignored
		public int? GetNextFrame(ref List<Texture2D> Planes, ref List<PixelFormat> PixelFormats)
		{
			var FrameMeta = GetNextFrameAndMeta( ref Planes, ref PixelFormats );
			if (!FrameMeta.HasValue)
				return null;

			return FrameMeta.Value.FrameNumber;
		}

	}
	
	[System.Serializable]
	public struct EncoderParams
	{
//	public string	Encoder = "avf"|"x264"
//	public int		Quality = [0..9]				x264
//	public int		AverageKbps = int				avf kiloBYTES
//	public int		MaxKbps = int					avf kiloBYTES
//	public bool		Realtime = true				avf: kVTCompressionPropertyKey_RealTime
//	.MaxFrameBuffers = undefined	avf: kVTCompressionPropertyKey_MaxFrameDelayCount
//	.MaxSliceBytes = number			avf: kVTCompressionPropertyKey_MaxH264SliceBytes
//	.MaximisePowerEfficiency = true	avf: kVTCompressionPropertyKey_MaximizePowerEfficiency
//	public int		ProfileLevel = 30(int)			Baseline only at the moment. 30=3.0, 41=4.1 etc this also matches the number in SPS. Default will try and pick correct for resolution or 3.0
	};

	[System.Serializable]
	public struct EncoderFrameMeta
	{
		public int			Width;
		public int			Height;
		public int			LumaSize;	//	bytes
		public int			ChromaUSize;	//	bytes
		public int			ChromaVSize;	//	bytes
		public bool			Keyframe;
		public string		Format;
		public PixelFormat	PixelFormat
		{
			get { PixelFormat value = PixelFormat.Invalid;	PixelFormat.TryParse(this.Format,out value);	return value; }
			set { this.Format = value.ToString(); }
		}
	}
	
	[System.Serializable]
	public struct H264Frame
	{
		public byte[]		H264Data;
		public EncodedFrameMeta	Meta;
	}
	
	//	data coming out of PopH264_EncoderPeekData
	[System.Serializable]
	public struct EncodedFrameMeta
	{
	}
	
	//	data coming out of PopH264_EncoderPeekData
	[System.Serializable]
	public struct PoppedFrameMeta
	{
		public int					DataSize;	//	bytes
		public EncodedFrameMeta		Meta;	//	all the meta sent to PopH264_EncoderPushFrame
		public int?					EncodeDurationMs;	//	time it took to encode
		public int?					DelayDurationMs;	//	time spent in queue before encoding (lag)
		public int					OutputQueueCount;	//	time spent in queue before encoding (lag)
	}

	public class Encoder : IDisposable
	{
		int? Instance = null;
		
		public Encoder(EncoderParams? EncoderParams)
		{
			if ( !EncoderParams.HasValue )
				EncoderParams = new EncoderParams();
			
			var ParamsJson = JsonUtility.ToJson(EncoderParams.Value);
			var ParamsJsonAscii = System.Text.ASCIIEncoding.ASCII.GetBytes(ParamsJson + "\0");
			var ErrorBuffer = new byte[200];
			Instance = PopH264_CreateEncoder(ParamsJsonAscii, ErrorBuffer, ErrorBuffer.Length);
			var Error = GetString(ErrorBuffer);
			if (Instance.Value <= 0)
				throw new System.Exception("Failed to create decoder instance;" + Error);
			if (!String.IsNullOrEmpty(Error))
			{
				Debug.LogWarning("Created PopH264 decoder (" + Instance.Value + ") but error was not empty (length = " + Error.Length + ") " + Error);
			}
		}
		
		~Encoder()
		{
			Dispose();
		}
		
		public void Dispose()
		{
			if (Instance.HasValue)
				PopH264_DestroyEncoder(Instance.Value);
			Instance = null;
		}
		
		public void PushFrame(byte[] PixelData,int Width,int Height,TextureFormat Format,bool Keyframe=false)
		{
			if ( Format == TextureFormat.RGBA32 )
				PushFrameRGBA(PixelData,Width,Height);
			else if ( Format == TextureFormat.R8)
				PushFrameGreyscale(PixelData,Width,Height);
			else
				throw new Exception($"Unhandled input format {Format}");
		}
		
		public void PushFrameGreyscale(byte[] Luma,int Width,int Height,bool Keyframe=false)
		{
			PushFrame( Luma, null, null, Width, Height, PixelFormat.Greyscale, Keyframe );
		}

		void RgbToYuv(byte r,byte g,byte b,out byte Luma,out byte ChromaU,out byte ChromaV)
		{
			var rf = r / 255.0;
			var gf = g / 255.0;
			var bf = b / 255.0;
			//	very quick conversion, should clarify if BT709 or what colour space it is
			var lf = rf * 0.299 + gf * 0.587 + bf * 0.114;
			var uf = rf * (-0.14713) - gf * 0.28886 + bf * 0.436;
			var vf = rf * 0.615 - gf * 0.51499 - bf * 0.10001;
			Luma = (byte)(lf * 255.0);
			ChromaU = (byte)((uf * 255.0) + 127.0);
			ChromaV = (byte)((vf * 255.0) + 127.0);
		}
		
		public void PushFrameRGBA(byte[] Rgba,int Width,int Height,bool Keyframe=false)
		{
			//	macos & ios supports RGBA input
#if ENCODER_SUPPORTS_RGBA_INPUT
			//	unity texture2D data is often... the wrong size.
			if ( Rgba.Length > (Width*Height*4) )
			{
				var RgbaSlice = new ArraySegment<byte>( Rgba, 0, Width*Height*4 );
				var RgbaClip = RgbaSlice.ToArray();
				PushFrame( RgbaClip, null, null, Width, Height, PixelFormat.RGBA, Keyframe );
			}
			else
			{
				PushFrame( Rgba, null, null, Width, Height, PixelFormat.RGBA, Keyframe );
			}
#else
			//	UWP only seems to accept 8_88/NV12
			//	apple supports either
			//	windows supports 8_8_8/nv21
			#if UNITY_WSA
			PushFrameRGBAToYuv8_88(Rgba,Width,Height,Keyframe);
			#else
			PushFrameRGBAToYuvI420(Rgba,Width,Height,Keyframe);
			#endif
#endif
		}
		
		//	I420 or NV12 - 3 planes, luma, chroma, chroma
		public void PushFrameRGBAToYuvI420(byte[] Rgba,int Width,int Height,bool Keyframe=false)
		{
			var HalfWidth = Width/2;
			var HalfHeight = Height/2;
			var LumaSize = Width * Height;
			var ChromaSize = HalfWidth * HalfHeight;
			var YuvPixels = new byte[LumaSize+ChromaSize+ChromaSize];
			var LumaPlane = new ArraySegment<byte>(YuvPixels, 0, LumaSize);
			var ChromaUPlane = new ArraySegment<byte>(YuvPixels, LumaSize, ChromaSize);
			var ChromaVPlane = new ArraySegment<byte>(YuvPixels, LumaSize+ChromaSize, ChromaSize);
			byte Luma,ChromaU,ChromaV;
			for ( var i=0;	i<Width*Height;	i++ )
			{
				var RgbaIndex = i * 4;
				var r = Rgba[RgbaIndex+0];
				var g = Rgba[RgbaIndex+1];
				var b = Rgba[RgbaIndex+2];
				var a = Rgba[RgbaIndex+3];
				RgbToYuv( r,g,b,out Luma,out ChromaU,out ChromaV);
				var x = i % Width;
				var y = i / Width;
				var Chromax = x / 2;
				var Chromay = y/2;
				var Chromai = Chromax + (Chromay * HalfWidth );
				LumaPlane[i] = Luma;
				ChromaUPlane[Chromai] = ChromaU;
				ChromaVPlane[Chromai] = ChromaV;
			}
			//	push as one plane with a format (mediafoundation & avf support this better)
			PushFrameYuvI420( YuvPixels, Width, Height, Keyframe );
			//PushFrame( LumaPlane, ChromaUPlane, ChromaVPlane, Width, Height, PixelFormat.Yuv_8_8_8, Keyframe );
		}
		
		//	420O or NV21 - 2 planes, chroma interleaved
		public void PushFrameRGBAToYuv8_88(byte[] Rgba,int Width,int Height,bool Keyframe=false)
		{
			var HalfWidth = Width/2;
			var HalfHeight = Height/2;
			var LumaSize = Width * Height;
			var ChromaSize = HalfWidth * HalfHeight;
			var YuvPixels = new byte[LumaSize+ChromaSize+ChromaSize];
			var LumaPlane = new ArraySegment<byte>(YuvPixels, 0, LumaSize);
			var ChromaUVPlane = new ArraySegment<byte>(YuvPixels, LumaSize, ChromaSize+ChromaSize);
			byte Luma,ChromaU,ChromaV;
			for ( var i=0;	i<Width*Height;	i++ )
			{
				var RgbaIndex = i * 4;
				var r = Rgba[RgbaIndex+0];
				var g = Rgba[RgbaIndex+1];
				var b = Rgba[RgbaIndex+2];
				var a = Rgba[RgbaIndex+3];
				RgbToYuv( r,g,b,out Luma,out ChromaU,out ChromaV);
				var x = i % Width;
				var y = i / Width;
				var Chromax = x / 2;
				var Chromay = y / 2;
				var Chromai = Chromax + (Chromay * HalfWidth);
				Chromai *= 2;
				
				LumaPlane[i] = Luma;
				ChromaUVPlane[Chromai+0] = ChromaU;
				ChromaUVPlane[Chromai+1] = ChromaV;
			}
			//	push as one plane with a format (mediafoundation & avf support this better)
			PushFrameYuv8_88( YuvPixels, Width, Height, Keyframe );
		}

		public void PushFrameYuvI420(byte[] Yuv8_8_8,int Width,int Height,bool Keyframe=false)
		{
			PushFrame( Yuv8_8_8, null, null, Width, Height, PixelFormat.Yuv_8_8_8, Keyframe );
		}
		
		public void PushFrameYuv8_88(byte[] Yuv8_88,int Width,int Height,bool Keyframe=false)
		{
			PushFrame( Yuv8_88, null, null, Width, Height, PixelFormat.Yuv_8_88, Keyframe );
		}

		public void PushFrame(byte[] Plane0,byte[] Plane1,byte[] Plane2,int Width,int Height,PixelFormat Format,bool Keyframe=false)
		{
			EncoderFrameMeta FrameMeta;
			FrameMeta.Width = Width;
			FrameMeta.Height = Height;
			FrameMeta.Keyframe = Keyframe;
			FrameMeta.LumaSize = (Plane0!=null) ? Plane0.Length : 0;
			FrameMeta.ChromaUSize = (Plane1!=null) ? Plane1.Length : 0;
			FrameMeta.ChromaVSize = (Plane2!=null) ? Plane2.Length : 0;
			//FrameMeta.PixelFormat = Format;
			FrameMeta.Format = Format.ToString();

			
			var MetaJson = JsonUtility.ToJson(FrameMeta);
			var MetaJsonAscii = System.Text.ASCIIEncoding.ASCII.GetBytes(MetaJson + "\0");
			var ErrorBuffer = new byte[200];
			PopH264_EncoderPushFrame( Instance.Value, MetaJsonAscii, Plane0, Plane1, Plane2, ErrorBuffer, ErrorBuffer.Length );

			var Error = GetString(ErrorBuffer);
			if ( !String.IsNullOrEmpty(Error) )
				throw new Exception($"PopH264.Encoder.PushFrame error {Error}");
		}
		
		public void PushEndOfStream()
		{
			PopH264_EncoderEndOfStream( Instance.Value );
		}
		
		public H264Frame? PopFrame()
		{
			var MetaJsonBuffer = new byte[1024*20];
			PopH264_EncoderPeekData(Instance.Value, MetaJsonBuffer,MetaJsonBuffer.Length);
			var MetaJson = GetString(MetaJsonBuffer);
			var PoppedFrameMeta = JsonUtility.FromJson<PoppedFrameMeta>(MetaJson);
			
			Debug.Log($"PopFrame() -> {MetaJson}");
			//	any data pending?
			////	gr: how do we know stream is finished?
			if ( PoppedFrameMeta.DataSize == 0 )
			{
				Debug.Log($"No pending frame to pop-> {MetaJson}");
				return null;
			}
			
			H264Frame Frame;
			//	todo: pool these buffers
			Frame.H264Data = new byte[PoppedFrameMeta.DataSize];
			Frame.Meta = PoppedFrameMeta.Meta;
			
			var BytesWritten = PopH264_EncoderPopData( Instance.Value, Frame.H264Data, Frame.H264Data.Length );
			//	returns 0 if there is no Data to pop.
			if ( BytesWritten == 0 )
			{
				Debug.LogWarning($"Popped 0 bytes for frame; but frame expected; Meta={MetaJson}");
				return null;
			}
			if ( BytesWritten < 0 )
				throw new Exception($"Error from PopH264_EncoderPopData; {BytesWritten}; Meta={MetaJson}");

			return Frame;
		}
	}

	public static byte[] GetH264TestData(string TestDataName)
	{
		var H264DataBuffer = new Byte[1024 * 1024 * 1];
		var TestDataNameAscii = System.Text.ASCIIEncoding.ASCII.GetBytes(TestDataName + "\0");
		var DataSize = PopH264_GetTestData(TestDataNameAscii,H264DataBuffer,H264DataBuffer.Length);
		if ( DataSize == 0 )
			throw new Exception($"No such test data named {TestDataName}");
		if ( DataSize < 0 )
			throw new Exception($"Error getting test data named {TestDataName}");
			
		var ClippedDataView = new ArraySegment<byte>(H264DataBuffer,0,DataSize);
		var ClippedData = ClippedDataView.ToArray();
		return ClippedData;
	}
}

