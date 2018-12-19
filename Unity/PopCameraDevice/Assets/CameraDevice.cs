using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraDevice : MonoBehaviour {

	public string DeviceName = "Test";
	public string MaterialUniform_Luma = "_MainTex";
	public string MaterialUniform_ChromaU = "ChromaU";
	public string MaterialUniform_ChromaV = "ChromaV";
	Texture2D Plane0;
	Texture2D Plane1;
	Texture2D Plane2;
	PopCameraDevice.Device Device;

	void OnEnable()
	{
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
			if ( Device.GetNextFrame( ref Plane0,  ref Plane1,  ref Plane2 ) )
				OnNewFrame();
	}


	void OnNewFrame()
	{
		var mr = GetComponent<MeshRenderer>();
		var mat = mr.sharedMaterial;
		mat.SetTexture(MaterialUniform_Luma, Plane0);
		mat.SetTexture(MaterialUniform_ChromaU, Plane1);
		mat.SetTexture(MaterialUniform_ChromaV, Plane2);
	}

}
