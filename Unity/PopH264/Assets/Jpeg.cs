using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;


public class Jpeg : MonoBehaviour {

	enum PacketFormat
	{
		Avcc,
		AnnexB
	};


	byte HexCharToNibble(char HexChar)
	{
		if (HexChar >= 'A' && HexChar <= 'F') return (byte)((HexChar - 'A') + 10);
		if (HexChar >= 'a' && HexChar <= 'f') return (byte)((HexChar - 'a') + 10);
		if (HexChar >= '0' && HexChar <= '9') return (byte)((HexChar - '0'));
		throw new System.Exception(HexChar + " not a hex char");
	}

	byte[] HexStringToBytes(string HexString)
	{
		if (string.IsNullOrEmpty(HexString))
			return null;

		var Bytes = new List<byte>();
		for (int i = 0; i < HexString.Length;	i+=2)
		{
			var a = HexCharToNibble(HexString[i + 0]);
			var b = HexCharToNibble(HexString[i + 1]);
			var Byte = (byte)((a<<4) | b);
			Bytes.Add(Byte);
		}
		return Bytes.ToArray();
	}

	//	now sps & pps
	public string Preconfigured_SPS_HexString = "";
	public byte[] Preconfigured_SPS_Bytes	{ get { return HexStringToBytes(Preconfigured_SPS_HexString); } }

	public bool VerboseDebug = false;

	[Header("Make this false to sync DecodeToVideoTime manually")]
	public bool AutoTrackTimeOnEnable = true;
	float? StartTime = null;    //	set this to time.time on enable if AutoTrackTimeOnEnable

	public UnityEngine.Events.UnityEvent OnFinished;

	[Range(0, 20)]
	public float DecodeToVideoTime = 0;
	public int DecodeToVideoTimeMs { get { return (int)(DecodeToVideoTime * 1000.0f); } }

	public PopH264.DecoderMode HardwareDecoding = PopH264.DecoderMode.Software;
	public bool DecodeOnSeperateThread = true;

	//	these values dictate how much processing time we give to each step
	[Range(0, 20)]
	public int DecodeImagesPerFrame = 1;
	[Range(0, 20)]
	public int DecodeSamplesPerFrame = 1;
	[Range(0, 20)]
	public int DecodePacketsPerFrame = 1;
	[Range(0, 20)]
	public int DecodeMp4AtomsPerFrame = 1;


	List<PopH264.FrameInput> PendingInputFrames;	//	h264 packets per-frame
	List<int> PendingOutputFrameTimes;              //	frames we've submitted to the h264 decoder, that we're expecting to come out
	int? LastFrameTime = null;                      //	maybe a better way of storing this


	long Mp4BytesRead = 0;                          //	amount of data we've processed from the start of the asset, so we know correct file offsets
	System.Func<long, long, byte[]> ReadFileFunction;   //	if set, we use this to read data (eg, from memory buffer). Other

	//	assuming only one for now
	int? H264TrackIndex = null;    
	H264.AvccHeader? H264TrackHeader = null;

	public string MaterialUniform_LumaTexture = "LumaTexture";
	public string MaterialUniform_LumaFormat = "LumaFormat";
	public string MaterialUniform_ChromaUTexture = "ChromaUTexture";
	public string MaterialUniform_ChromaUFormat = "ChromaUFormat";
	public string MaterialUniform_ChromaVTexture = "ChromaVTexture";
	public string MaterialUniform_ChromaVFormat = "ChromaVFormat";
	[Header("Turn off to debug by changing the material setting manually")]
	public bool SetMaterialFormat = true;
	List<Texture2D> PlaneTextures;
	List<PixelFormat> PlaneFormats;
	PopH264.Decoder Decoder;

	[Header("Whenever there's a new frame, all these blits will be processed")]
	public List<VideoBlitTexture> RgbTextureBlits;

	public UnityEvent_String OnNewFrameTime;
	public UnityEvent_String OnDebugUpdate;

	void OnEnable()
	{
		//	get the file reader
		var FileReader = GetComponent<FileReaderBase>();
		Mp4BytesRead = 0;
		ReadFileFunction = FileReader.GetReadFileFunction();

		if (AutoTrackTimeOnEnable)
		{
			StartTime = Time.time;
		}

	}

	long GetKnownFileSize()
	{
		var FileReader = GetComponent<FileReaderBase>();
		var Size = FileReader.GetKnownFileSize();
		return Size;
	}

	static T[] CombineTwoArrays<T>(T[] a1, T[] a2)
	{
		T[] arrayCombined = new T[a1.Length + a2.Length];
		System.Buffer.BlockCopy(a1, 0, arrayCombined, 0, a1.Length);
		System.Buffer.BlockCopy(a2, 0, arrayCombined, a1.Length, a2.Length);
		return arrayCombined;
	}

	void PushFrame_AnnexB(byte[] H264Packet,int FrameNumber)
	{
		if (H264Packet == null)
			return;
		if (PendingInputFrames == null)
			PendingInputFrames = new List<PopH264.FrameInput>();

		//	if the last frame has the same frame number, append the bytes
		//	assuming we just have continuation packets (or say, SPS and PPS bundled together)
		if (PendingInputFrames.Count > 0 )
		{
			var LastFrame = PendingInputFrames[PendingInputFrames.Count - 1];
			if (LastFrame.FrameNumber == FrameNumber)
			{
				LastFrame.Bytes = CombineTwoArrays(LastFrame.Bytes, H264Packet);
				PendingInputFrames[PendingInputFrames.Count - 1] = LastFrame;
				return;
			}
		}

		var Frame = new PopH264.FrameInput();
		Frame.Bytes = H264Packet;
		Frame.FrameNumber = FrameNumber;
		PendingInputFrames.Add(Frame);
	}


	void ParseNextJpegHeader()
	{
		//	check if there's more data to be read
		var KnownFileSize = GetKnownFileSize();
		if (Mp4BytesRead >= KnownFileSize)
			return;
		
		//	ideally only once we've verified we have an mp4, but before moov. Maybe just if an ftyp is found
		if (Decoder == null)
			Decoder = new PopH264.Decoder( HardwareDecoding, DecodeOnSeperateThread );
		
		System.Func<long, byte[]> PopData = (long DataSize)=>
		{
			var Data = ReadFileFunction(Mp4BytesRead, DataSize);
			Mp4BytesRead += DataSize;
			return Data;
		};

		try
		{
			DecodeNextSample();
		}
		catch(System.Exception e)
		{
			Debug.LogException(e);
		}
	}

	void OnDisable()
	{
		//	reset everything
		if (Decoder != null)
			Decoder.Dispose();
		Decoder = null;

		PendingInputFrames = null;
		ReadFileFunction = null;
		Mp4BytesRead = 0;

		H264TrackIndex = null;
		H264TrackHeader = null;

		StartTime = null;
	}


	//	gah why cant I lamda generics
	int GetCountString<T>(List<T> Array,int Div=1)
	{
		if (Array == null)
			return -1;
		return Array.Count / Div;
	}
	int GetCountString(long Count, int Div = 1)
	{
		return (int)(Count / Div);
	}

	void Update()
	{
		//	auto track time
		if (StartTime.HasValue)
		{
			DecodeToVideoTime = Time.time - StartTime.Value;
		}

		//	update debug output
		{
			string Debug = "";
			Debug += "Kb: " + GetCountString(this.GetKnownFileSize(),1024);
			Debug += "Packets: " + GetCountString(this.PendingInputFrames);
			this.OnDebugUpdate.Invoke(Debug);
		}

		//	always try and get next image
		{
			for (var i = 0; i < DecodeImagesPerFrame;	i++)
				DecodeNextFrameImage();
		}

		//	if we have frames/packets to decode, do them
		try
		{
			for (var i = 0; i < DecodePacketsPerFrame;	i++ )
				DecodeNextFramePacket();
		}
		catch (System.Exception e)
		{
			Debug.LogException(e, this);
		}

		try
		{
			for (var i = 0; i < DecodeMp4AtomsPerFrame; i++)
				ParseNextJpegHeader();
		}
		catch (System.Exception e)
		{
			//	if there's an error, we probably don't have enough mp4 data
			Debug.LogException(e, this);
		}
	
	}

	void DecodeNextSample()
	{
		if ( PendingInputFrames==null )
			PendingInputFrames = new List<PopH264.FrameInput>();

		//	any data left?
		var Size = GetKnownFileSize();
		if (Mp4BytesRead >= Size)
			return;

		//	read next frame
		var Frame = new PopH264.FrameInput();
		Frame.Bytes = ReadFileFunction(Mp4BytesRead, Size - Mp4BytesRead);
		Frame.FrameNumber = 0;
		PendingInputFrames.Add(Frame);
	}

	void DecodeNextFramePacket()
	{
		if (PendingInputFrames == null || PendingInputFrames.Count == 0)
			return;

		var PendingInputFrame = PendingInputFrames[0];
		if ( VerboseDebug )
			Debug.Log("Push frame data #" + PendingInputFrame.FrameNumber);
		var PushResult = Decoder.PushFrameData(PendingInputFrame);
		if (PushResult != 0)
		{
			//	decoder error
			Debug.LogError("Decoder Push returned: " + PushResult + " (decoder error!)");
		}

		if (PendingOutputFrameTimes == null)
			PendingOutputFrameTimes = new List<int>();
		PendingOutputFrameTimes.Add(PendingInputFrame.FrameNumber);
		PendingInputFrames.RemoveAt(0);
	}

	//	returns if any frames decoded.
	bool DecodeNextFrameImage()
	{
		if (Decoder == null)
			return false;

		//	only pull out next frame, if we WANT the pending frame times
		if (PendingOutputFrameTimes == null )
			return false;

		if (PendingOutputFrameTimes.Count == 0)
		{
			Debug.Log("Not expecting any more frames added extra PendingOutputFrameTimes " + (DecodeToVideoTimeMs + 1) );
			//return false;
			PendingOutputFrameTimes.Add(DecodeToVideoTimeMs + 1);
		}

		//	not reached next frame yet
		if (DecodeToVideoTimeMs < PendingOutputFrameTimes[0])
		{
			Debug.Log("DecodeToVideoTimeMs " + DecodeToVideoTimeMs + " < PendingOutputFrameTimes[0] " + PendingOutputFrameTimes[0]);
			return false;
		}

		var FrameTime = Decoder.GetNextFrame(ref PlaneTextures, ref PlaneFormats);

		//	nothing decoded
		if (!FrameTime.HasValue)
			return false;

		var ExpectedTime = PendingOutputFrameTimes[0];
		//	gr: there's something off here, we don't seem to decode early frames
		//		is the decoder skipping frames? (no!) or are the times offset?
		//	anyway, once we get a frame in the future, we shouldn't expect any older ones, so clear the pending list up to this frame
		if ( FrameTime.Value != ExpectedTime )
		{
			if ( VerboseDebug )
				Debug.Log("Decoded frame " + FrameTime.Value + ", expected " + ExpectedTime + " when seeking to " + DecodeToVideoTimeMs);
		}

		//	remove all the frames we may have skipped over
		while (PendingOutputFrameTimes.Count > 0)
		{
			if (PendingOutputFrameTimes[0] > FrameTime.Value)
				break;
			PendingOutputFrameTimes.RemoveAt(0);
		}

		OnNewFrame();
		OnNewFrameTime.Invoke("" + FrameTime.Value);

		if (LastFrameTime.HasValue && LastFrameTime.Value == FrameTime.Value)
		{
			if (VerboseDebug)
				Debug.Log("Decoded last frame");
			OnFinished.Invoke();
		}

		return true;
	}



	void OnNewFrame()
	{
		//	update params on material
		UpdateMaterial(GetComponent<MeshRenderer>());
		UpdateBlits();
	}

	void UpdateBlit(VideoBlitTexture Blit)
	{
		if (Blit.RgbTexture == null)
			return;
		if (Blit.YuvMaterial == null)
			return;

		UpdateMaterial(Blit.YuvMaterial);
		Graphics.Blit(null, Blit.RgbTexture, Blit.YuvMaterial);
	}

	void UpdateBlits()
	{
		if (RgbTextureBlits == null)
			return;

		foreach (var Blit in RgbTextureBlits)
			UpdateBlit(Blit);
	}

	void UpdateMaterial(MeshRenderer MeshRenderer)
	{
		if (MeshRenderer == null)
			return;
		var mat = MeshRenderer.material;
		UpdateMaterial(mat);
	}



	void UpdateMaterial(Material mat)
	{
		if (mat == null)
			return;
	

		if (PlaneTextures.Count >= 1)
		{
			mat.SetTexture(MaterialUniform_LumaTexture, PlaneTextures[0]);
			if (SetMaterialFormat)
				mat.SetInt(MaterialUniform_LumaFormat, (int)PlaneFormats[0]);
		}

		if (PlaneTextures.Count >= 2)
		{
			mat.SetTexture(MaterialUniform_ChromaUTexture, PlaneTextures[1]);
			if (SetMaterialFormat)
				mat.SetInt(MaterialUniform_ChromaUFormat, (int)PlaneFormats[1]);
		}

		if (PlaneTextures.Count >= 3)
		{
			mat.SetTexture(MaterialUniform_ChromaVTexture, PlaneTextures[2]);
			if (SetMaterialFormat)
				mat.SetInt(MaterialUniform_ChromaVFormat, (int)PlaneFormats[2]);
		}


	}

}
