using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugCameraDevices : MonoBehaviour {

	void Update ()
	{
		var DeviceNames = PopCameraDevice.EnumCameraDevices();
		DeviceNames.ForEach( Debug.Log );
	}
}
