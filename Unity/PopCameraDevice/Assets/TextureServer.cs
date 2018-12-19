using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;

#if WINDOWS_UWP
using Windows.Networking.Sockets;
#endif

[System.Serializable]
public class UnityEvent_String : UnityEngine.Events.UnityEvent <string> {}


#if WINDOWS_UWP
public class HttpListener
{
/*
	public HttpListenerPrefixCollection Prefixes { get; }
	
	public IAsyncResult BeginGetContext(AsyncCallback callback, object state)
	{
		return null;
	}
	public void Close()	{}
	//public HttpListenerContext EndGetContext(IAsyncResult asyncResult);
	//public HttpListenerContext GetContext();
	public void Start()	{}
	public void Stop()	{}
*/
};
#endif


public class TextureServer : MonoBehaviour {

	public int			Port = 80;
	HttpListener		Http;

	public Texture2D	texture;
	public bool			TextureDirty = true;
	byte[]				LastJpeg;
	int					LastJpegLength;

	public UnityEvent_String	OnDebug;

	void Start()
	{
		if (Http != null)
			return;

		Http = new HttpListener ();

		try
		{
			/*
			Http.Prefixes.Add ("http://*:" + Port + "/");
			Http.Start ();
			Http.BeginGetContext(ListenerCallback, Http);
			*/
		}
		catch {
			//Stop ();
			throw;
		}
	}

	void Stop()
	{
		/*
		if (Http == null)
			return;
		Http.Stop ();
		Http = null;
		System.GC.Collect ();
		*/
	}
		
	public void ListenerCallback(IAsyncResult result)
	{
		/*
		HttpListener listener = (HttpListener)result.AsyncState;
		// Call EndGetContext to complete the asynchronous operation.
		HttpListenerContext context = listener.EndGetContext(result);
		HttpListenerRequest request = context.Request;
		HttpListenerResponse response = context.Response;
	
		//	Construct a response.
		if (LastJpeg == null) {
			string responseString = "<HTML><BODY>Jpeg not encoded.</BODY></HTML>";
			byte[] buffer = System.Text.Encoding.UTF8.GetBytes (responseString);
			response.ContentLength64 = buffer.Length;
			System.IO.Stream output = response.OutputStream;
			output.Write (buffer, 0, buffer.Length);
			output.Close ();
		} else {
			//lock (LastJpeg) 
			{
				response.ContentLength64 = LastJpegLength;
				System.IO.Stream output = response.OutputStream;
				output.Write (LastJpeg, 0, LastJpegLength);
				output.Close ();
			};
		}

		//	listen for next request
		listener.BeginGetContext(ListenerCallback, listener);
		*/
	}

		

	void Update () {

		if (texture == null)
			return;

		if (TextureDirty) {
			//lock (LastJpeg)
			{
				try
				{
					var EncodeStart = Time.realtimeSinceStartup;
					PopEncodeJpeg.EncodeToJpeg (texture, ref LastJpeg, ref LastJpegLength);
					var EncodeTimeMs = (int)((Time.realtimeSinceStartup - EncodeStart) * 1000.0f);
					Debug.Log("Jpeg encode took " + EncodeTimeMs + "ms");
					OnDebug.Invoke("Jpeg encode took " + EncodeTimeMs + "ms");
					TextureDirty = false;
				}
				catch {
					LastJpegLength = 0;
					throw;
				}					
			}
		}

	}

	public void SetTexture(Texture2D NewTexture)
	{
		texture = NewTexture;
		TextureDirty = true;
	}

	public void SetTexture(WebCamTexture NewTexture)
	{
		if (texture != null) {
			if (texture.width != NewTexture.width || texture.height != NewTexture.height)
				texture = null;
		}

		try
		{
			if (texture == null) {
				texture = new Texture2D (NewTexture.width, NewTexture.height, TextureFormat.ARGB32, false);
			}

			var Pixels = NewTexture.GetPixels32();
			texture.SetPixels32( Pixels );
			TextureDirty = true;
		}
		catch
		{
			texture = null;
			throw;
		}
	}
}
