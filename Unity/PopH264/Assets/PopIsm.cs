using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

//[System.Serializable]
//public class UnityEvent_String : UnityEngine.Events.UnityEvent<string> { }

[System.Serializable]
public class UnityEvent_SmoothStream : UnityEngine.Events.UnityEvent<PopX.Ism.SmoothStream> { }


public class PopIsm : MonoBehaviour
{
	public UnityEvent_String OnError;
	public UnityEvent_SmoothStream OnParsedStream;
	[Header("Example url: http://poph264test-euwe.streaming.media.azure.net/56909db0-0ba4-45bf-b2ae-0497e6e93049/cat_baseline.ism/")]
	public string Url_Host = "poph264test-euwe";
	public string Url_Endpoint = "56909db0-0ba4-45bf-b2ae-0497e6e93049";
	public string Url_Asset = "cat_baseline";

	public string Url_Base
	{
		get
		{
			return "http://" + Url_Host + ".streaming.media.azure.net/" + Url_Endpoint + "/" + Url_Asset + ".ism/";
		}
	}
	public string Url_Manifest
	{
		get
		{
			return Url_Base + "Manifest";
		}
	}

	void OnEnable()
	{
		System.Action<string> HandleError = (Error)=>
		{
			Debug.LogError(Error);
			OnError.Invoke(Error);
		};
		System.Action<PopX.Ism.SmoothStream> HandleStream = (Stream)=>
		{
			Debug.Log("Parsed smooth stream");
			OnParsedStream.Invoke(Stream);
			LoadIntoMp4(Stream);
		};
		StartCoroutine(PopX.Ism.GetManifest(Url_Base, Url_Manifest, HandleStream, HandleError) );
	}

	public void LoadIntoMp4(PopX.Ism.SmoothStream Stream)
	{
		var Mp4 = GetComponent<Mp4>();

		//	download mp4 from stream
		var Track = Stream.GetVideoTrack();
		var ChunkIndex = 0;
		var Source = Track.Sources[0];
		var ChunkUrl = Track.GetUrl(Source.BitRate, ChunkIndex, Stream.BaseUrl);

		//	todo: convert to bytes here and remove from Mp4.cs
		var TrackSpsAndPps = Source.CodecData_Hex;

		System.Action<string> HandleError = (Error) =>
		{
			Debug.LogError(Error);
			OnError.Invoke(Error);
		};

		System.Action<byte[]> HandleMp4Bytes = (Bytes) =>
		{
			Mp4.Preconfigured_SPS_HexString = TrackSpsAndPps;
			Mp4.LoadMp4(Bytes);
			Mp4.enabled = true;
		};

		StartCoroutine(LoadMp4(ChunkUrl, HandleError, HandleMp4Bytes));
	}

	static IEnumerator LoadMp4(string Url,System.Action<string> OnError,System.Action<byte[]> OnBytesDownloaded)
	{
		Debug.Log("Fetching " + Url);
		var www = UnityWebRequest.Get(Url);
		yield return www.SendWebRequest();

		if (www.isNetworkError || www.isHttpError)
		{
			OnError(Url + " error: " + www.error);
			yield break;
		}

		//	Show results as text
		try
		{
			var Bytes = www.downloadHandler.data;
			OnBytesDownloaded(Bytes);
		}
		catch (System.Exception e)
		{
			OnError(e.Message);
		}
	}

}

