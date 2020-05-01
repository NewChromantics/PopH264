using System.IO;
using Unity.Collections;
using UnityEngine;
using System.Runtime.InteropServices;

struct PointCloudStreamElement
{
	public float x;
	public float y;
	public float z;
	public float Pad;
	public byte b;
	public byte g;
	public byte r;
	public byte a;
}



[System.Serializable]
public class UnityEvent_Texture : UnityEngine.Events.UnityEvent<Texture> { };


[RequireComponent(typeof(FileReaderBase))]
public class PointcloudStreamParser : MonoBehaviour
{
	const int ElementSize = 20;
	[Range(0, 1024)]
	public int ImageWidth = 256;
	public int FrameWidth = 40939;
	public int FrameHeight = 1;
	public int ElementCount { get { return FrameWidth * FrameHeight; } }
	public int FrameSize { get { return ElementCount * ElementSize; } }
	public int TextureWidth
	{
		get
		{
			var w = (ImageWidth == 0) ? (int)Mathf.Sqrt(ElementCount) : ImageWidth;
			w = Mathf.NextPowerOfTwo(w);
			return w;
		}
	}

	public Texture2D PositionImage;
	public UnityEvent_Texture OnNewPositionImage;
	public Texture2D ColourImage;
	public UnityEvent_Texture OnNewColourImage;
	NativeArray<float> PositionBuffer;
	NativeArray<byte> ColourBuffer;

	bool BufferChanged = false;
	int PositionChannels;
	int ColourChannels;
	public bool TransformRosToUnity = false;
	public bool Flip = true;

	System.Func<long, long, byte[]> ReadBytes;
	long BytesRead = 0;
	[Header("Use this with network buffer reader")]
	public bool FrameSizeIsKnownSize = false;
	[Header("Mostly to debug performance, go back to start of stream")]
	public bool LoopStream = false;
	[Header("Mostly to debug with above")]
	public bool EndOfStreamOnError = false;
	long? KnownSize = null;
	public bool UseThread = true;
	System.Threading.Thread ReadThread;
	[Range(0, 1000)]
	public int ThreadSleepMillisecs = 30;
	[Range(0, 1000)]
	public int ThreadQueuedTextureSleepMillisecs = 100;
	[Header("To avoid tearing, lock textures. This may not be neccessary though")]
	public bool LockTextures = false;
	[Header("Shouldn't be needed, but debugging problems")]
	public bool ExplicitlyLoadTextureData = false;

	void OnEnable()
	{
		DestroyThread();
		BytesRead = 0;
		KnownSize = 0;

		if (Marshal.SizeOf(typeof(PointCloudStreamElement)) != ElementSize)
			throw new System.Exception("Struct size should be 20");

		var Reader = GetComponent<FileReaderBase>();
		if (Reader is BufferReader)
			FrameSizeIsKnownSize = true;
		
		ReadBytes = Reader.GetReadFileFunction();

		//	work out image size for alignment
		var TexWidth = TextureWidth;
		var TexHeight = Mathf.NextPowerOfTwo(ElementCount / TexWidth);

		PositionImage = new Texture2D(TexWidth, TexHeight, TextureFormat.RGBAFloat,false,false);
		ColourImage = new Texture2D(TexWidth, TexHeight, TextureFormat.RGBA32, false, false);
		PositionImage.name = "Cloud Positions";
		ColourImage.name = "Cloud Colours";
		PositionImage.filterMode = FilterMode.Point;
		ColourImage.filterMode = FilterMode.Point;
		PositionBuffer = PositionImage.GetRawTextureData<float>();
		ColourBuffer = ColourImage.GetRawTextureData<byte>();
		PositionChannels = 4;
		ColourChannels = 4;
	}

	void OnEndOfStream()
	{
		if (LoopStream)
		{
			BytesRead = 0;
		}
	}


	//	threaded call
	void ReadNextIteration()
	{
		while (true)
		{
			try
			{
				if (!ReadNextPacket())
				{
					OnEndOfStream();
				}
			}
			catch(System.Exception e)
			{
				Debug.LogException(e);
				if (EndOfStreamOnError)
					OnEndOfStream();
			}
			System.Threading.Thread.Sleep(ThreadSleepMillisecs);

			//	if waiting for last frame to be consumed, sleep longer
			if ( BufferChanged )
			{
				Debug.Log("Thread waiting for last frame to be used");
				System.Threading.Thread.Sleep(ThreadQueuedTextureSleepMillisecs);
			}
		}
	}

	void CreateThread()
	{
		if (ReadThread != null)
			return;

		Debug.Log("Creating thread", this);
		ReadThread = new System.Threading.Thread(new System.Threading.ThreadStart(ReadNextIteration));
		ReadThread.Start();
	}


	void DestroyThread()
	{
		if (ReadThread==null)
			return;

		Debug.Log("Destroying thread",this);
		ReadThread.Abort();
		ReadThread.Join();
		ReadThread = null;
	}

	void OnDisable()
	{
		DestroyThread();
	}

	void Update()
    {
		//	update the known size
		var FileReader = GetComponent<FileReaderBase>();
		KnownSize = FileReader.GetKnownFileSize();


		if ( PositionImage != null && PositionImage.width != TextureWidth && ImageWidth!=0)
		{
			OnEnable();
		}

		if (UseThread)
		{
			//	startup thread
			CreateThread();
		}
		else
		{
			DestroyThread();    //	so we can toggle at runtime
			try
			{
				UnityEngine.Profiling.Profiler.BeginSample("ReadNextPacket");
				if (BufferChanged)
				{
					Debug.Log("Waiting for previous frame to be processed");
				}
				else
				{
					if (!ReadNextPacket())
					{
						OnEndOfStream();
					}
				}
			}
			catch (System.Exception e)
			{
				Debug.LogException(e);
				if (EndOfStreamOnError)
					OnEndOfStream();
			}
			UnityEngine.Profiling.Profiler.EndSample();
		}

		//	texture data needs updating
		if ( BufferChanged )
		{
			if (LockTextures)
			{
				lock (PositionImage)
				{
					UpdateTexture();
				}
			}
			else
			{
				UpdateTexture();
			}
		}
	}

	void UpdateTexture()
	{
		Debug.Log("Updating texture", this);
		UnityEngine.Profiling.Profiler.BeginSample("Update Position Texture");
		//	gr: writing directly into the native array means we don't need to re-upload
		if ( ExplicitlyLoadTextureData )
			PositionImage.LoadRawTextureData(PositionBuffer);
		PositionImage.Apply();
		OnNewPositionImage.Invoke(PositionImage);
		UnityEngine.Profiling.Profiler.EndSample();

		UnityEngine.Profiling.Profiler.BeginSample("Colour Position Texture");
		//	gr: writing directly into the native array means we don't need to re-upload
		if (ExplicitlyLoadTextureData)
			ColourImage.LoadRawTextureData(ColourBuffer);
		ColourImage.Apply();
		OnNewColourImage.Invoke(ColourImage);
		UnityEngine.Profiling.Profiler.EndSample();

		BufferChanged = false;
	}


	static byte GetByteN(uint abcd, int Index)
	{
		var Byte = abcd >> 8 * Index;
		return (byte)(Byte & 0xff);
	}


	//	returns if we have more data to read
	bool ReadNextPacket()
	{
		//	havent been able to get known size yet
		if (!KnownSize.HasValue)
			return true;

		//	no more data to read
		//	note- this may just be waiting for more bytes from the network
		//	read up to the end of the file
		if (BytesRead >= KnownSize.Value)
			return false;

		//	gr: known size is now currentpos + next chunk
		var PacketSize = FrameSizeIsKnownSize ? KnownSize.Value- BytesRead : FrameSize;
		var PacketElementCount = FrameSizeIsKnownSize ? (PacketSize / ElementSize) : ElementCount;

		//	read (throws if no enough)
		var FrameBytes = ReadBytes(BytesRead, PacketSize);
		BytesRead += FrameBytes.Length;
		Debug.Log("Stream Parser read " + FrameBytes.Length + " bytes");

		var Element = default(PointCloudStreamElement);


		/* gr: this is faster, but the intptr alloc is expensive
		var ArrayPointer = Marshal.AllocHGlobal(FrameSize);
		Marshal.Copy(FrameBytes, 0, ArrayPointer, FrameBytes.Length);
		for (var i = 0; i < ElementCount; i++)
		{
			var StructPtr = new System.IntPtr(ArrayPointer.ToInt64() + Marshal.SizeOf(typeof(PointCloudStreamElement)));
			Element = (PointCloudStreamElement)Marshal.PtrToStructure(ArrayPointer, typeof(PointCloudStreamElement));
		}
		Marshal.FreeHGlobal(ArrayPointer);
		return true;
		*/

		//	binary reader should be really fast, but its still 50ms for 800kb which is too slow
		var Reader = new BinaryReader(new MemoryStream(FrameBytes));

		//	real loop profiling
		UnityEngine.Profiling.Profiler.BeginSample("Parse elements");
		lock (PositionImage)
		{

			for (var i = 0; i < PacketElementCount; i++)
			{
				var bi = i * ElementSize;

				Element.x = Reader.ReadSingle();
				Element.y = Reader.ReadSingle();
				Element.z = Reader.ReadSingle();
				Element.Pad = Reader.ReadSingle();
				//	speed up - makes no difference
				/*
				var bgra = Reader.ReadUInt32();
				Element.b = GetByteN(bgra, 0);
				Element.g = GetByteN(bgra, 1);
				Element.r = GetByteN(bgra, 2);
				Element.a = GetByteN(bgra, 3);
				*/
				Element.b = Reader.ReadByte();
				Element.g = Reader.ReadByte();
				Element.r = Reader.ReadByte();
				Element.a = Reader.ReadByte();

				//	can we check for frame alignment vs image alignment here, or do it in shader?
				var pi = i * PositionChannels;
				var ci = i * ColourChannels;

				if (Flip)
					Element.y = -Element.y;

				if (TransformRosToUnity)
				{
					PositionBuffer[pi + 0] = -Element.y;
					PositionBuffer[pi + 1] = Element.z;
					PositionBuffer[pi + 2] = Element.x;
				}
				else
				{
					PositionBuffer[pi + 0] = Element.x;
					PositionBuffer[pi + 1] = Element.y;
					PositionBuffer[pi + 2] = Element.z;
				}

				PositionBuffer[pi + 3] = Element.Pad;

				ColourBuffer[ci + 0] = Element.r;
				ColourBuffer[ci + 1] = Element.g;
				ColourBuffer[ci + 2] = Element.b;
				ColourBuffer[ci + 3] = Element.a;
			}
			UnityEngine.Profiling.Profiler.EndSample();
		}
		BufferChanged = true;

		return true;
	}

}
