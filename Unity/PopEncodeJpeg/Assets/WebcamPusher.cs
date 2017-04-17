using System.Collections;
using System.Collections.Generic;
using UnityEngine;



[System.Serializable]
public class UnityEvent_Texture : UnityEngine.Events.UnityEvent <Texture> {}

[System.Serializable]
public class UnityEvent_WebCamTexture : UnityEngine.Events.UnityEvent <WebCamTexture> {}


public class WebcamPusher : MonoBehaviour {

	public string			WebcamName;
	public WebCamTexture	Webcam;
	public int				WebcamWidth = 0;
	public int				WebcamHeight = 0;

	public UnityEvent_Texture	OnNewFrameTexture;
	public UnityEvent_WebCamTexture	OnNewFrameWebCamTexture;


	public void ListDevices()
	{
		// Gets the list of devices and prints them to the console.
		WebCamDevice[] devices = WebCamTexture.devices;
		for (int i = 0; i < devices.Length; i++)
			Debug.Log(devices[i].name);
	}

	void Start()
	{
		if ( WebcamName.Length == 0 )
		{
			WebCamDevice[] devices = WebCamTexture.devices;
			if ( devices.Length > 0 )
				WebcamName = devices[0].name;
		}
	}

	void Update () 
	{
		if (Webcam == null )
		{
			try
			{
				ListDevices();

				if ( WebcamWidth>0 && WebcamHeight>0 )
					Webcam = new WebCamTexture( WebcamName, WebcamWidth, WebcamHeight );
				else
					Webcam = new WebCamTexture( WebcamName );
				Webcam.Play();
				if ( !Webcam.isPlaying )
					throw new System.Exception("No webcam");
			}
			catch {
				Webcam = null;
			}
		}

		if ( Webcam!=null && Webcam.didUpdateThisFrame )
		{
			WebcamWidth = Webcam.width;
			WebcamHeight = Webcam.height;
			OnNewFrameTexture.Invoke ( Webcam as Texture );
			OnNewFrameWebCamTexture.Invoke ( Webcam );
		}
	}
}
