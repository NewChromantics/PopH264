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

	public string Preconfigured_SPS_HexString = "";
	public string Preconfigured_PPS_HexString = "";
	public byte[] Preconfigured_SPS_Bytes	{ get { return HexStringToBytes(Preconfigured_SPS_HexString); } }
	public byte[] Preconfigured_PPS_Bytes { get { return HexStringToBytes(Preconfigured_PPS_HexString); } }

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
		Decoder = new PopH264.Decoder();


		if (!System.IO.File.Exists(Filename))
			throw new System.Exception("File missing: " + Filename);
		var Mp4Bytes = System.IO.File.ReadAllBytes(Filename);
		var Mp4BytesArray = new List<byte>(Mp4Bytes);

		System.Action<byte[]> PushAnnexB = (Bytes) =>
		{
			if (H264PendingData == null)
				H264PendingData = new List<byte>();

			H264PendingData.AddRange(Bytes);
		};

		System.Action<List<byte[]>> PushAnnexBs = (Bytess) =>
		{
			if (H264PendingData == null)
				H264PendingData = new List<byte>();

			Bytess.ForEach(b => H264PendingData.AddRange(b));
		};

		System.Action<byte[]> PushH264Sample = (Bytes) =>
		{
			if (H264PendingData == null)
				H264PendingData = new List<byte>();
			H264PendingData.AddRange(new byte[] { 0, 0, 0, 1 });
			H264PendingData.AddRange(Bytes);
		};

		System.Action<long, long> ExtractH264Sample = (Position,Size)=>
		{
			var SampleBytes = Mp4Bytes.SubArray(Position, Size);
			PushH264Sample(SampleBytes);
		};

		System.Action<PopX.Mpeg4.TTrack> EnumTrack = (Track) =>
		{
			var Pps = Preconfigured_PPS_Bytes;
			var Sps = Preconfigured_SPS_Bytes;

			if (Pps ==null|| Sps==null)
			{
				if (Track.SampleDescriptions != null && Track.SampleDescriptions[0].Fourcc != "avc1")
				{
					Debug.Log("Skipping track codec: " + Track.SampleDescriptions[0].Fourcc);
					return;
				}

				var Header = PopX.H264.ParseAvccHeader(Track.SampleDescriptions[0].AvccAtomData);
				Pps = Header.PPSs[0];
				Sps = Header.SPSs[0];
			}

			PushAnnexB(Pps);
			PushAnnexB(Sps);

			Debug.Log("Found mp4 track " + Track.Samples.Count);
			foreach ( var Sample in Track.Samples )
			{
				//ExtractH264Sample(Sample.DataPosition, Sample.DataSize);
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
