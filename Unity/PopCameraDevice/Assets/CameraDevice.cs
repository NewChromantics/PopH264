using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraDevice : MonoBehaviour {

	public string DeviceName = "Test";
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
		if ( Device.GetNextFrame( ref Plane0,  ref Plane1,  ref Plane2 ) )
			OnNewFrame( Plane0);
	}


	void OnNewFrame(Texture texture)
	{
		var mr = GetComponent<MeshRenderer>();
		var mat = mr.sharedMaterial;
		mat.mainTexture = texture;
	}

}
