using System.Collections;
using System.Collections.Generic;
using UnityEngine;


//	gr: I had a script in PopUnityCommon somewhere for this...
[System.Serializable]
public class UnityEvent_String : UnityEngine.Events.UnityEvent<string> { }


public class CameraDevice : MonoBehaviour {


	public bool PushAllData = false;
	[Range(1, 1024)]
	public int PopKbPerFrame = 30;
	public TextAsset H264Data;
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
	}

	void OnDisable()
	{
		if (Decoder != null )
			Decoder.Dispose();
		Decoder = null;
	}


	void Update()
	{
		//	gr: be careful with looping in this way, as we'll just queue tons of data to the decoder
		if (H264PendingData == null )//|| H264PendingData.Count == 0 )
		{
			H264PendingData = new List<byte>(H264Data.bytes);
		}

		if (Decoder != null )
		{
			System.Action PushNewData = () =>
			{
				var PopBytesSize = PushAllData ? H264PendingData.Count : PopKbPerFrame * 1024;
				PopBytesSize = Mathf.Min(PopBytesSize, H264PendingData.Count);
				if (PopBytesSize == 0)
				{
					//Debug.Log("Pushed all data");
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

		if ( PlaneTextures.Count >= 1 )
		{
			mat.SetTexture(MaterialUniform_LumaTexture, PlaneTextures[0] );
			if ( SetMaterialFormat )
				mat.SetInt(MaterialUniform_LumaFormat, (int)PlaneFormats[0] );
		}

		if ( PlaneTextures.Count >= 2 )
		{
			mat.SetTexture(MaterialUniform_ChromaUTexture, PlaneTextures[1] );
			if ( SetMaterialFormat )
				mat.SetInt(MaterialUniform_ChromaUFormat, (int)PlaneFormats[1] );
		}

		if ( PlaneTextures.Count >= 3 )
		{
			mat.SetTexture(MaterialUniform_ChromaVTexture, PlaneTextures[2] );
			if ( SetMaterialFormat )
				mat.SetInt(MaterialUniform_ChromaVFormat, (int)PlaneFormats[2] );
		}
	
	}

}
