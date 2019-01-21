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

	List<byte> H264PendingData;
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
		if (!System.IO.File.Exists(Filename))
			throw new System.Exception("File missing: " + Filename);

		var Mp4Bytes = System.IO.File.ReadAllBytes(Filename);
		LoadMp4(Mp4Bytes);
	}

	public void LoadMp4(byte[] Mp4Bytes)
	{
		if ( Decoder == null)
			Decoder = new PopH264.Decoder();
	
		System.Action<byte[]> PushAnnexB = (Bytes) =>
		{
			if (Bytes == null)
				return;
			if (H264PendingData == null)
				H264PendingData = new List<byte>();

			H264PendingData.AddRange(Bytes);
		};

		System.Func<long, long,byte[]> ExtractSample = (Position,Size)=>
		{
			var SampleBytes = Mp4Bytes.SubArray(Position, Size);
			return SampleBytes;
		};

		System.Action<PopX.Mpeg4.TTrack> EnumTrack = (Track) =>
		{
			System.Action<byte[]> PushPacket;

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

				PushPacket = (Packet) => { H264.AvccToAnnexb4(Header, Packet, PushAnnexB); };
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
				PushPacket = (Packet) => { H264.AvccToAnnexb4(Header, Packet, PushAnnexB); };
			}

			H264.Profile Profile;
			float Level;
			PopX.H264.GetProfileLevel(Sps_AnnexB, out Profile, out Level);
			if (Profile != H264.Profile.Baseline)
				Debug.LogWarning("PopH264 currently only supports baseline profile. This is " + Profile + " level=" + Level);

			PushAnnexB(Sps_AnnexB);
			PushAnnexB(Pps_AnnexB);

			Debug.Log("Found mp4 track " + Track.Samples.Count);
			foreach ( var Sample in Track.Samples )
			{
				var Packet = ExtractSample(Sample.DataPosition, Sample.DataSize);
				PushPacket(Packet);
			}
		};

		PopX.Mpeg4.Parse(Mp4Bytes, EnumTrack);
		Debug.Log("Extracted " + H264PendingData.Count + " bytes of h264");
	}

	void OnDisable()
	{
		if (Decoder != null)
			Decoder.Dispose();
		Decoder = null;
	}


	void Update()
	{
		if (H264PendingData == null)
		{
			this.enabled = false;
			throw new System.Exception("Didnt load any h264 data");
		}

		if (Decoder != null)
		{
			System.Action PushNewData = () =>
			{
				var PopBytesSize = PushAllData ? H264PendingData.Count : PopKbPerFrame * 1024;
				PopBytesSize = Mathf.Min(PopBytesSize, H264PendingData.Count);
				if (PopBytesSize == 0)
				{
					return;
				}

				var PopBytes = new byte[PopBytesSize];
				H264PendingData.CopyTo(0, PopBytes, 0, PopBytes.Length);
				H264PendingData.RemoveRange(0, PopBytesSize);

				var PushResult = Decoder.PushFrameData(PopBytes);
				if (PushResult != 0)
				{
					Debug.Log("Push returned: " + PushResult);
				}
			};

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
		var mat = mr.sharedMaterial;

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
