using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraDevice : MonoBehaviour {

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
		if (H264PendingData==null || H264PendingData.Count == 0 )
			H264PendingData = new List<byte>(H264Data.bytes);

		if (Decoder != null )
		{
			var PopBytesSize = Mathf.Min(PopKbPerFrame, H264PendingData.Count);
			var PopBytes = new byte[PopBytesSize];
			H264PendingData.CopyTo(0, PopBytes, 0, PopBytes.Length);
			H264PendingData.RemoveRange(0, PopBytesSize);

			var PushResult = Decoder.PushFrameData(PopBytes);
			Debug.Log("Push returned: " + PushResult);

			if (Decoder.GetNextFrame( ref PlaneTextures,  ref PlaneFormats ) )
				OnNewFrame();
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
