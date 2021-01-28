using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

[System.Serializable]
public class UnityEvent_String : UnityEngine.Events.UnityEvent<string> { }


[System.Serializable]
public struct VideoBlitTexture
{
	public RenderTexture RgbTexture;
	public Material YuvMaterial;
}


public class Mp4 : MonoBehaviour {

	enum PacketFormat
	{
		Avcc,
		AnnexB
	};

	struct TPendingSample
	{
		public PopX.Mpeg4.TSample Sample;
		public PacketFormat Format;
		public uint MdatIndex;
		public long? DataPosition { get { return Sample.DataPosition; } }
		public long? DataFilePosition { get { return Sample.DataFilePosition; } }
		public long DataSize { get { return Sample.DataSize; } }
		public int PresentationTime { get { return Sample.PresentationTimeMs; } }
	}

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

	public PopH264.DecoderParams DecoderParams;
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
	List<TPendingSample> PendingInputSamples;		//	references to all samples that are scheduled, but we may not have data for yet (and may need conversion to a packet)
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
	List<PopH264.PixelFormat> PlaneFormats;
	PopH264.Decoder Decoder;

	[Header("Whenever there's a new frame, all these blits will be processed")]
	public List<VideoBlitTexture> RgbTextureBlits;

	struct MdatBlock
	{
		public TAtom Atom;
		public byte[] Bytes	{ get { return Atom.AtomData; }}
		public long FileOffset;	//	where did this mdat start (sample chunks are 
	}
	Dictionary<uint, MdatBlock> Mdats;
	uint NextMdat = 0;
	bool MDatBeforeTrack = false;	//	this is a bit hacky atm, but we can get mdat before tracks

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

	void CullMdatBlocks(int DiscardBlockIndex)
	{
		//	any before old block (and inclusive) we don't need any more
		var AllIndexes = new List<uint>(Mdats.Keys);
		foreach(var Index in AllIndexes)
		{
			if (Index > DiscardBlockIndex)
				continue;
			Mdats.Remove(Index);
		}
	}

	bool HasMdat(uint MdatIndex)
	{
		if (Mdats == null)
			return false;
		return Mdats.ContainsKey(MdatIndex);
	}

	byte[] GetDataBytes(TPendingSample Sample)
	{
		var MdatIndex = Sample.MdatIndex;
		//Sample.DataPosition, Sample.DataSizeuint MdatIndex, long Position, long Size)
		if (!HasMdat(MdatIndex))
			throw new System.Exception("Not yet recieved mdat #" + MdatIndex);
		var Meta = Mdats[MdatIndex];

		long Position;
		if (Sample.DataFilePosition.HasValue)
		{
			Position = Sample.DataFilePosition.Value - Meta.Atom.AtomDataFilePosition;
		}
		else
		{
			Position = Sample.DataPosition.Value;
		}

		return Meta.Bytes.SubArray(Position, Sample.DataSize);
	}

	void ParseNextMp4Header()
	{
		//	check if there's more data to be read
		var KnownFileSize = GetKnownFileSize();
		if (Mp4BytesRead >= KnownFileSize)
			return;


		int TimeOffset = 0;

		System.Action<PopX.Mpeg4.TTrack> EnumTrack = (Track) =>
		{
			System.Action<byte[],int> PushPacket;

			byte[] Sps_AnnexB;
			byte[] Pps_AnnexB;

			//	track has header (so if we have moov and moof's, this only comes up once, and that's when we need to decode SPS/PPS
			if (Track.SampleDescriptions != null)
			{
				if (Track.SampleDescriptions[0].Fourcc != "avc1")
					throw new System.Exception("Expecting fourcc avc1, got " + Track.SampleDescriptions[0].Fourcc);

				H264.AvccHeader Header;
				Header = PopX.H264.ParseAvccHeader(Track.SampleDescriptions[0].AvccAtomData);

				//	wrong place to assign this! should be when we assign the track index
				H264TrackHeader = Header;

				var Pps = new List<byte>(new byte[] { 0, 0, 0, 1 });
				Pps.AddRange(Header.PPSs[0]);
				Pps_AnnexB = Pps.ToArray();

				var Sps = new List<byte>(new byte[] { 0, 0, 0, 1 });
				Sps.AddRange(Header.SPSs[0]);
				Sps_AnnexB = Sps.ToArray();

				PushPacket = (Packet, FrameNumber) =>
				{
					H264.AvccToAnnexb4(Header, Packet, (Bytes) => { PushFrame_AnnexB(Bytes, FrameNumber); });
				};
			}
			else if (Preconfigured_SPS_Bytes != null && Preconfigured_SPS_Bytes.Length > 0)
			{
				throw new System.Exception("Need to refactor to process preconfigured SPS before handling tracks");
				//	split this header
				Sps_AnnexB = Preconfigured_SPS_Bytes;
				Pps_AnnexB = null;

				//	gr: turns out these are AVCC, not annexb
				//	gr: should be able to auto detect without header
				//PushPacket = PushAnnexB;
				H264.AvccHeader Header = new H264.AvccHeader();
				Header.NaluLength = 2;
				PushPacket = (Packet, FrameNumber) =>
				{
					H264.AvccToAnnexb4(Header, Packet, (Bytes) => { PushFrame_AnnexB(Bytes, FrameNumber); });
				};
			}
			else
			{
				//	track without header, assume it's already come via moov and this is a moof
				Sps_AnnexB = null;
				Pps_AnnexB = null;
				PushPacket = null;
			}


			//	load h264 header
			if (Sps_AnnexB != null)
			{
				H264.Profile Profile;
				float Level;
				PopX.H264.GetProfileLevel(Sps_AnnexB, out Profile, out Level);
				if (Profile != H264.Profile.Baseline)
					Debug.LogWarning("PopH264 currently only supports baseline profile. This is " + Profile + " level=" + Level);

				PushFrame_AnnexB(Sps_AnnexB, 0);
				PushFrame_AnnexB(Pps_AnnexB, 0);
			}

			//	load samples
			if (Track.Samples == null)
			{
				if (VerboseDebug)
					Debug.Log("Mp4 Track with null samples (next mdat="+ NextMdat+")");
			}
			else
			{
				//	gr: is there a better mdat ident? don't think so, it's just the upcoming one in mpeg sequence
				//	gr: sometimes, the mdat is before the tracks...
				//	gr: the sample offset also needs correcting
				var MdatIndex = MDatBeforeTrack ? NextMdat-1 : 0;

				if (VerboseDebug)
					Debug.Log("Found mp4 track " + Track.Samples.Count + " for next mdat: "+ MdatIndex);

				if ( Track.Samples.Count > 0 )
				{
					var LastSampleTime = Track.Samples[Track.Samples.Count - 1].PresentationTimeMs;
					if (!LastFrameTime.HasValue)
						LastFrameTime = LastSampleTime;
					LastFrameTime = Mathf.Max(LastFrameTime.Value, LastSampleTime);
				}

				foreach (var Sample in Track.Samples)
				{
					try
					{
						var NewSample = new TPendingSample();
						if (PendingInputSamples == null)
							PendingInputSamples = new List<TPendingSample>();

						NewSample.Sample = Sample;
						NewSample.MdatIndex = MdatIndex;
						//NewSample.DataPosition = Sample.DataPosition;
						//NewSample.DataFilePosition = Sample.DataFilePosition;
						//NewSample.DataSize = Sample.DataSize;
						var TimeOffsetMs = (int)(TimeOffset / 10000.0f);
						//NewSample.PresentationTime = Sample.PresentationTimeMs + TimeOffsetMs;
						NewSample.Format = PacketFormat.Avcc;
						PendingInputSamples.Add(NewSample);
					}
					catch (System.Exception e)
					{
						Debug.LogException(e);
						break;
					}
				}
			}
		
		};

		System.Action<List<PopX.Mpeg4.TTrack>> EnumTracks = (Tracks)=>
		{
			//	moof headers don't have track headers, so we should have our track by now
			//	this track may be the original though, so hunt down the h264 one
			if (!H264TrackIndex.HasValue)
			{
				for (int t = 0; t < Tracks.Count; t++)
				{
					var Track = Tracks[t];
					if (Track.SampleDescriptions == null)
						continue;

					if (Track.SampleDescriptions[0].Fourcc != "avc1")
					{
						Debug.Log("Skipping track codec: " + Track.SampleDescriptions[0].Fourcc);
						continue;
					}

					H264TrackIndex = t;
				}
			}

			if (!H264TrackIndex.HasValue)
				throw new System.Exception("Couldn't find avc1 track");

			EnumTrack(Tracks[H264TrackIndex.Value]);
		};

		System.Action<PopX.TAtom> EnumMdat = (MdatAtom) =>
		{
			if (VerboseDebug)
				Debug.Log("EnumMdat( NextMdat=" + NextMdat);

			if (!H264TrackIndex.HasValue)
				MDatBeforeTrack = true;

			//	this is the meta for the pending mdat
			var MdatIndex = NextMdat;
					
			if (Mdats == null)
				Mdats = new Dictionary<uint, MdatBlock>();

			var Mdat = new MdatBlock();
			Mdat.Atom = MdatAtom;
			Mdat.FileOffset = Mdat.Atom.AtomDataFilePosition;	//	need a better way to do this

			if ( VerboseDebug )
				Debug.Log("Got MDat " + MdatIndex + " x" + Mdat.Bytes.Length + " bytes");
			Mdats.Add(MdatIndex, Mdat);

			//	increment once everything succeeds
			NextMdat++;
		};

		//	ideally only once we've verified we have an mp4, but before moov. Maybe just if an ftyp is found
		if (Decoder == null)
			Decoder = new PopH264.Decoder(DecoderParams, DecodeOnSeperateThread );


		System.Func<long, byte[]> PopData = (long DataSize)=>
		{
			var Data = ReadFileFunction(Mp4BytesRead, DataSize);
			Mp4BytesRead += DataSize;
			return Data;
		};

		try
		{
			PopX.Mpeg4.ParseNextAtom(PopData, Mp4BytesRead, EnumTracks, EnumMdat);
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
		PendingInputSamples = null;
		ReadFileFunction = null;
		Mp4BytesRead = 0;

		H264TrackIndex = null;
		H264TrackHeader = null;

		Mdats = null;
		NextMdat = 0;   //	could be wise not to reset this to weed out bugs

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
			Debug += "Samples: " + GetCountString(this.PendingInputSamples);
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

		//	convert samples to input frames, if data is availible
		//	gr: maybe process next mp4 IF this fails
		try
		{
			for (var i = 0; i < DecodeSamplesPerFrame; i++)
				DecodeNextSample();
		}
		catch (System.Exception e)
		{
			//	if there's an error, we probably don't have enough mp4 data
			Debug.LogException(e, this);
		}


		//	decode more of the mp4 (may need to do inital header, or a moof header, or next mdat, before we can do a frame again
		try
		{
			for (var i = 0; i < DecodeMp4AtomsPerFrame; i++)
				ParseNextMp4Header();
		}
		catch (System.Exception e)
		{
			Debug.LogException(e, this);
		}


	}

	void DecodeNextSample()
	{
		if (PendingInputSamples == null || PendingInputSamples.Count == 0)
			return;

		var Sample = PendingInputSamples[0];

		//	fail silently if mdat not availible yet
		if (!VerboseDebug)
			if (!HasMdat(Sample.MdatIndex))
				return;

		//	grab data
		var SampleBytes = GetDataBytes(Sample);

		//	need to convert?
		if (Sample.Format == PacketFormat.Avcc)
		{
			System.Action<byte[]> OnAnnexB = (AnnexBBytes) =>
			{
				PushFrame_AnnexB(AnnexBBytes, Sample.PresentationTime);
			};
			PopX.H264.AvccToAnnexb4(H264TrackHeader.Value, SampleBytes, OnAnnexB);
		}
		else if ( Sample.Format == PacketFormat.AnnexB )
		{
			PushFrame_AnnexB(SampleBytes, Sample.PresentationTime);
		}

		PendingInputSamples.RemoveAt(0);

		//	we've processed this sample, we're assuming we won't need any samples in previous mdat blocks now
		//	could play it safe and do -1
		if (Sample.MdatIndex > 0)
			CullMdatBlocks((int)Sample.MdatIndex - 1);
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
