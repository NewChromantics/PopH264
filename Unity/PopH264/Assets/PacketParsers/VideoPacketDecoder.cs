using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;


public class VideoPacketDecoder : MonoBehaviour
{
	public bool VerboseDebug = false;


	[Header("Make this false to sync DecodeToVideoTime manually")]
	public bool AutoTrackTimeOnEnable = true;
	float? StartTime = null;    //	set this to time.time on enable if AutoTrackTimeOnEnable

	public UnityEngine.Events.UnityEvent	OnFinished;
	bool									OnFinishedCalled = false;	//	call once

	[Range(0, 20)]
	public float DecodeToVideoTime = 0;
	public int DecodeToVideoTimeMs { get { return (int)(DecodeToVideoTime * 1000.0f); } }

	public PopH264.DecoderMode HardwareDecoding = PopH264.DecoderMode.Software;
	public bool DecodeOnSeperateThread = true;

	//	these values dictate how much processing time we give to each step
	[Range(0, 20)]
	public int DecodeImagesPerFrame = 1;
	[Range(0, 1000)]
	public int DecodePacketsPerFrame = 1;
	
	List<PopH264.FrameInput> PendingInputFrames;	//	h264 packets per-frame
	List<int> PendingOutputFrameTimes;              //	frames we've submitted to the h264 decoder, that we're expecting to come out
	int? LastFrameTime = null;                      //	maybe a better way of storing this



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
		if (AutoTrackTimeOnEnable)
		{
			StartTime = Time.time;
		}
		OnFinishedCalled = false;
	}

	public void PushPacket(byte[] Data, long TimeMs)
	{
		if (PendingInputFrames == null)
			PendingInputFrames = new List<PopH264.FrameInput>();

		//	gr: for UDP fragmented packets, we're generating massive amounts of packets which is causing a big strain
		//		the h264 decoder can now take any old bunch of packets, they don't need to be cut up into frames
		//		we still want to split them when we DO have frame numbers
		//	todo: keep adding to previous frame until we spot a NAL, would be good to have this split into packets unity side
		//		for now; keep one big buffer if the timestamp is zero (ie, no frame info)
		//	todo: even better when we have no meta, one giant buffer with minimal reallocs
		bool AppendToPrev = (TimeMs == 0);
		if (PendingInputFrames.Count == 0)
			AppendToPrev = false;

		if ( !AppendToPrev )
		{
			var NewPacket = new PopH264.FrameInput();
			NewPacket.FrameNumber = (int)TimeMs;
			PendingInputFrames.Add(NewPacket);
			if ( VerboseDebug )
				Debug.Log("New packet in buffer for time " + TimeMs + " x" + Data.Length);
		}

		//	replace last packet
		var Packet = PendingInputFrames[PendingInputFrames.Count - 1];
		if (Packet.Bytes == null)
		{
			Packet.Bytes = Data;
		}
		else
		{
			if ( VerboseDebug )
				Debug.Log("Joining packets x" + Packet.Bytes.Length + " + x" + Data.Length);
			Packet.Bytes = Packet.Bytes.JoinArray(Data);
		}

		//	structs, so we're not modifying in-place, have to override prev
		PendingInputFrames[PendingInputFrames.Count - 1] = Packet;
	}

	public void PushPacketWithNoTimestamp(byte[] Data)
	{
		//	todo: guess next timestamp from last
		PushPacket(Data, 0);
	}

	public void PushEndOfStreamPacket()
	{
		//	manually push a endofstream h264 packet
		if (PendingInputFrames == null)
			PendingInputFrames = new List<PopH264.FrameInput>();

		var NewPacket = new PopH264.FrameInput();
		PendingInputFrames.Add(NewPacket);
		if (!NewPacket.EndOfStream)
			throw new System.Exception("New blank PopH264.FrameInput should be marked as EndOfStream");
	}



	static T[] CombineTwoArrays<T>(T[] a1, T[] a2)
	{
		T[] arrayCombined = new T[a1.Length + a2.Length];
		System.Buffer.BlockCopy(a1, 0, arrayCombined, 0, a1.Length);
		System.Buffer.BlockCopy(a2, 0, arrayCombined, a1.Length, a2.Length);
		return arrayCombined;
	}

	

	void OnDisable()
	{
		//	reset everything
		if (Decoder != null)
			Decoder.Dispose();
		Decoder = null;

		PendingInputFrames = null;
		
		StartTime = null;
	}

	public void Reset()
	{
		OnDisable();
		OnEnable();
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
	}


	void DecodeNextFramePacket()
	{
		if (PendingInputFrames == null || PendingInputFrames.Count == 0)
			return;

		if (Decoder == null)
			Decoder = new PopH264.Decoder(HardwareDecoding, DecodeOnSeperateThread);

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
		{
			if (Decoder.HadEndOfStream)
			{
				Debug.Log("EndOfStream detected");
				if (!OnFinishedCalled)
				{
					Debug.Log("OnFinished call");
					OnFinished.Invoke();
					OnFinishedCalled = true;
				}
			}
			return false;
		}

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
			if ( !OnFinishedCalled )
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
