using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

public class Mp4 : MonoBehaviour {

	public string Filename = "Assets/cat.mov";

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

	public bool PushAllData = false;
	[Range(1, 1024)]
	public int PopKbPerFrame = 30;

	[Range(0,20)]
	public int PushPacketsPerFrameMin = 0;

	List<PopH264.FrameInput> PendingInputFrames;
	public string MaterialUniform_LumaTexture = "LumaTexture";
	public string MaterialUniform_LumaFormat = "LumaFormat";
	public string MaterialUniform_ChromaUTexture = "ChromaUTexture";
	public string MaterialUniform_ChromaUFormat = "ChromaUFormat";
	public string MaterialUniform_ChromaVTexture = "ChromaVTexture";
	public string MaterialUniform_ChromaVFormat = "ChromaVFormat";
	[Header("Turn off to debug by changing the material setting manually")]
	public bool SetMaterialFormat = true;
	List<Texture2D> PlaneTextures;
	List<PopH264.SoyPixelsFormat> PlaneFormats;
	PopH264.Decoder Decoder;

	public UnityEvent_String OnNewFrameTime;

	void OnEnable()
	{
		if (string.IsNullOrEmpty(Filename))
			return;

		if (!System.IO.File.Exists(Filename))
			throw new System.Exception("File missing: " + Filename);

		var Mp4Bytes = System.IO.File.ReadAllBytes(Filename);
		LoadMp4(Mp4Bytes);
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

	public void LoadMp4(byte[] Mp4Bytes,int TimeOffset=0)
	{
		if ( Decoder == null)
			Decoder = new PopH264.Decoder();	

		System.Func<long, long,byte[]> ExtractSample = (Position,Size)=>
		{
			var SampleBytes = Mp4Bytes.SubArray(Position, Size);
			return SampleBytes;
		};

		System.Action<PopX.Mpeg4.TTrack> EnumTrack = (Track) =>
		{
			System.Action<byte[],int> PushPacket;

			byte[] Sps_AnnexB;
			byte[] Pps_AnnexB;

			//	track has header
			if (Track.SampleDescriptions!= null )
			{
				if (Track.SampleDescriptions[0].Fourcc != "avc1")
				{
					Debug.Log("Skipping track codec: " + Track.SampleDescriptions[0].Fourcc);
					return;
				}

				H264.AvccHeader Header;
				Header = PopX.H264.ParseAvccHeader(Track.SampleDescriptions[0].AvccAtomData);
				var Pps = new List<byte>(new byte[] { 0, 0, 0, 1 });
				Pps.AddRange(Header.PPSs[0]);
				Pps_AnnexB = Pps.ToArray();

				var Sps = new List<byte>(new byte[] { 0, 0, 0, 1 });
				Sps.AddRange(Header.SPSs[0]);
				Sps_AnnexB = Sps.ToArray();

				PushPacket = (Packet, FrameNumber) => 
				{
					H264.AvccToAnnexb4(Header, Packet, (Bytes) => { PushFrame_AnnexB(Bytes, FrameNumber); } ); 
				};
			}
			else
			{
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

			H264.Profile Profile;
			float Level;
			PopX.H264.GetProfileLevel(Sps_AnnexB, out Profile, out Level);
			if (Profile != H264.Profile.Baseline)
				Debug.LogWarning("PopH264 currently only supports baseline profile. This is " + Profile + " level=" + Level);

			PushFrame_AnnexB(Sps_AnnexB,0);
			PushFrame_AnnexB(Pps_AnnexB,0);

			Debug.Log("Found mp4 track " + Track.Samples.Count);
			foreach ( var Sample in Track.Samples )
			{
				try
				{
					var Packet = ExtractSample(Sample.DataPosition, Sample.DataSize);
					var TimeOffsetMs = (int)(TimeOffset / 10000.0f);
					var FrameNumber = Sample.PresentationTimeMs + TimeOffsetMs;
					PushPacket(Packet, FrameNumber);
				}
				catch(System.Exception e)
				{
					Debug.LogException(e);
					break;
				}
			}
		};

		PopX.Mpeg4.Parse(Mp4Bytes, EnumTrack);
		Debug.Log("Extracted " + PendingInputFrames.Count + " frames/packets of h264");
	}

	void OnDisable()
	{
		if (Decoder != null)
			Decoder.Dispose();
		Decoder = null;
	}


	void StreamInH264Data()
	{
		//	todo:
		//	push in random bytes from file (emulating stream)
		//	split nal and add pending frames
		/*
		 * 
				}
				PendingInputFrames

				var PopBytesSize = PushAllData ? H264PendingData.Count : PopKbPerFrame * 1024;
				PopBytesSize = Mathf.Min(PopBytesSize, H264PendingData.Count);
				if (PopBytesSize == 0)
				{
					return;
				}

				var PopBytes = new byte[PopBytesSize];
				H264PendingData.CopyTo(0, PopBytes, 0, PopBytes.Length);
				H264PendingData.RemoveRange(0, PopBytesSize);
				*/
	}

	void Update()
	{
		if (Decoder != null)
		{
			System.Action PushNewData = () =>
			{
				StreamInH264Data();

				if (PendingInputFrames != null && PendingInputFrames.Count > 0)
				{
					var PushResult = Decoder.PushFrameData(PendingInputFrames[0]);
					PendingInputFrames.RemoveAt(0);
					if (PushResult != 0)
					{
						Debug.Log("Push returned: " + PushResult);
					}
				}
			};

			for (int i = 0; i < PushPacketsPerFrameMin;	i++)
				PushNewData();

			var FrameTime = Decoder.GetNextFrame(ref PlaneTextures, ref PlaneFormats);
			if (FrameTime.HasValue)
			{
				OnNewFrame();
				OnNewFrameTime.Invoke("" + FrameTime.Value);
			}
			else
			{
				PushNewData();
			}
		}
	}


	void OnNewFrame()
	{
		var mr = GetComponent<MeshRenderer>();
		var mat = mr.material;

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
