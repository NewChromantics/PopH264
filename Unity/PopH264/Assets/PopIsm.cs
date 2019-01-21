using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class PopIsm : MonoBehaviour {

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


	IEnumerator GetManifest()
	{
		var ManifestUrl = Url_Manifest;
		Debug.Log("Fetching " + ManifestUrl);
		var www = UnityWebRequest.Get(ManifestUrl);
		yield return www.SendWebRequest();

		if (www.isNetworkError || www.isHttpError)
		{
			OnError(www.error);
			yield break;
		}

		//	Show results as text
		ParseManifest(www.downloadHandler.text);
	}

	void OnEnable()
	{
		StartCoroutine(GetManifest());
	}

	void OnError(string Error)
	{
		Debug.LogError(Error);
	}

	void ParseManifest(string ManifestContents)
	{
		Debug.Log(ManifestContents);
	}

}
