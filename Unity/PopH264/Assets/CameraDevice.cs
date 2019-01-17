using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraDevice : MonoBehaviour {

	public int DeviceIndex = -1;
	public string DeviceName = "Test";
	public string MaterialUniform_LumaTexture = "LumaTexture";
	public string MaterialUniform_LumaFormat = "LumaFormat";
	public string MaterialUniform_ChromaUTexture = "ChromaUTexture";
	public string MaterialUniform_ChromaUFormat = "ChromaUFormat";
	public string MaterialUniform_ChromaVTexture = "ChromaVTexture";
	public string MaterialUniform_ChromaVFormat = "ChromaVFormat";
	[Header("Turn off to debug by changing the material setting manually")]
	public bool SetMaterialFormat = true;
	List<Texture2D> PlaneTextures;
	List<PopCameraDevice.SoyPixelsFormat> PlaneFormats;
	PopCameraDevice.Device Device;

	void OnEnable()
	{
		if ( DeviceIndex >= 0 )
		{
			var DeviceNames = PopCameraDevice.EnumCameraDevices();
			DeviceName = DeviceNames[DeviceIndex];
		}

		Device = new PopCameraDevice.Device(DeviceName);
	}

	void OnDisable()
	{
		if ( Device != null )
			Device.Dispose();
		Device = null;
	}


	void Update()
	{
		if ( Device != null )
		{
			if ( Device.GetNextFrame( ref PlaneTextures,  ref PlaneFormats ) )
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
