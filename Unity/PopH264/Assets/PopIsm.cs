using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Xml;


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
		var Url = Url_Manifest;
		System.Action<string> HandleError = (Error)=>
		{
			Debug.LogError(Error);
			OnError.Invoke(Error);
		};
		System.Action<PopX.Ism.SmoothStream> HandleStream = (Stream)=>
		{
			Debug.Log("Parsed smooth stream");
			OnParsedStream.Invoke(Stream);
		};
		StartCoroutine(PopX.Ism.GetManifest( Url, HandleStream, HandleError) );
	}
}


namespace PopX
{
	//	microsoft smooth streaming xml manifest
	public static class Ism
	{
		public static IEnumerator GetManifest(string ManifestUrl,System.Action<SmoothStream> OnSmoothStreamDecoded,System.Action<string> OnError)
		{
			Debug.Log("Fetching " + ManifestUrl);
			var www = UnityWebRequest.Get(ManifestUrl);
			yield return www.SendWebRequest();

			if (www.isNetworkError || www.isHttpError)
			{
				OnError(www.error);
				yield break;
			}

			//	Show results as text
			try
			{
				var ParsedStream = ParseManifest(www.downloadHandler.text);
				OnSmoothStreamDecoded(ParsedStream);
			}
			catch(System.Exception e)
			{
				OnError(e.Message);
			}
		}



		public enum SmoothStreamTrackType
		{
			Video,
			Audio,
		}
		public struct SmoothStreamSource //	"Quality Level"
		{
			public int Index;
			public int Bitrate;
			public string Fourcc;
			public int Width;
			public int Height;
			public string CodecData_Hex;    //	contents in xml are a string of hex bytes
											//byte[] CodecDataBytes{	get}
		}
		public class SmoothStreamTrack
		{
			public SmoothStreamTrackType Type;
			public string UrlTemplate;
			public List<int> ChunkStartTimes = new List<int>();
			public List<int> ChunkDurations = new List<int>();
			public List<SmoothStreamSource> Sources = new List<SmoothStreamSource>();
		}
		public class SmoothStream
		{
			public List<SmoothStreamTrack> Tracks = new List<SmoothStreamTrack>();
		}

		static SmoothStreamSource ParseTrackQualityLevel(XmlElement QualityLevelXml)
		{
			//	< QualityLevel Index = "0" Bitrate = "5492715" FourCC = "H264" MaxWidth = "1920" MaxHeight = "1080" CodecPrivateData = "0000000167640028AC2CA501E0089F97015202020280000003008000001E31300016E360000E4E1FF8C7076850A4580000000168E9093525" />
			var Source = new SmoothStreamSource();
			Source.Bitrate = int.Parse(QualityLevelXml.GetAttribute("Bitrate"));
			Source.Fourcc = QualityLevelXml.GetAttribute("FourCC");
			Source.Width = int.Parse(QualityLevelXml.GetAttribute("MaxWidth"));
			Source.Height = int.Parse(QualityLevelXml.GetAttribute("MaxHeight"));
			Source.CodecData_Hex = QualityLevelXml.GetAttribute("CodecPrivateData");
			return Source;
		}

		static SmoothStreamTrack ParseTrack(XmlElement StreamIndexXml)
		{
			//	<StreamIndex Chunks="4" Type="video" Url="QualityLevels({bitrate})/Fragments(video={start time})" QualityLevels="8">
			var Track = new SmoothStreamTrack();
			Track.UrlTemplate = StreamIndexXml.GetAttribute("Url");

			var Type = StreamIndexXml.GetAttribute("Type");
			if (Type == "video")
				Track.Type = SmoothStreamTrackType.Video;
			else if (Type == "audio")
				Track.Type = SmoothStreamTrackType.Audio;
			else
				throw new System.Exception("Unknown track type " + Type);

			//	< QualityLevel Index = "0" Bitrate = "5492715" FourCC = "H264" MaxWidth = "1920" MaxHeight = "1080" CodecPrivateData = "0000000167640028AC2CA501E0089F97015202020280000003008000001E31300016E360000E4E1FF8C7076850A4580000000168E9093525" />
			var ChunkStartTimes = new List<int?>();
			var ChunkDurations = new List<int>();
			foreach (XmlElement ChildElement in StreamIndexXml.ChildNodes)
			{
				if (ChildElement.Name == "QualityLevel")
				{
					var Source = ParseTrackQualityLevel(ChildElement);
					Track.Sources.Add(Source);
					continue;
				}
				if (ChildElement.Name == "c")
				{
					var StartTimeStr = ChildElement.HasAttribute("t") ? ChildElement.GetAttribute("t") : (string)null;
					var DurationStr = ChildElement.GetAttribute("d");

					var StartTime = !string.IsNullOrEmpty(StartTimeStr) ? int.Parse(StartTimeStr) : (int?)null;
					var Duration = int.Parse(DurationStr);

					ChunkStartTimes.Add(StartTime);
					ChunkDurations.Add(Duration);
					continue;
				}
				Debug.LogWarning("Skipping SmoothStringMedia child " + ChildElement.Name);
			}

			//	evaluate the chunk times
			{
				var StartTime = 0;
				for (var c = 0; c < ChunkStartTimes.Count; c++)
				{
					//	reset time
					var ChunkStartTime = ChunkStartTimes[c];
					var ChunkDuration = ChunkDurations[c];
					if (ChunkStartTime.HasValue)
						StartTime = ChunkStartTime.Value;

					Track.ChunkStartTimes.Add(StartTime);
					Track.ChunkDurations.Add(ChunkDuration);

					StartTime += ChunkDuration;
				}
			}

			return Track;
		}

		static SmoothStream ParseManifest(XmlNode SmoothStreamingMedia)
		{
			//	<SmoothStreamingMedia MajorVersion="2" MinorVersion="0" Duration="63854875" TimeScale="10000000">
			if (SmoothStreamingMedia == null)
				throw new System.Exception("Missing SmoothStreamingMedia element");

			var Stream = new SmoothStream();

			//	each streamindex is a track
			//	<StreamIndex Chunks="4" Type="video" Url="QualityLevels({bitrate})/Fragments(video={start time})" QualityLevels="8">
			//	<StreamIndex Chunks="4" Type="audio" Url="QualityLevels({bitrate})/Fragments(AAC_und_ch2_128kbps={start time})" QualityLevels="1" Name="AAC_und_ch2_128kbps">
			foreach (XmlElement ChildElement in SmoothStreamingMedia.ChildNodes)
			{
				if (ChildElement.Name == "StreamIndex")
				{
					var Track = ParseTrack(ChildElement);
					Stream.Tracks.Add(Track);
					continue;
				}
				Debug.LogWarning("Skipping SmoothStringMedia child " + ChildElement.Name);
			}

			return Stream;
		}

		static SmoothStream ParseManifest(string ManifestContents)
		{
			var xml = new XmlDocument();
			xml.LoadXml(ManifestContents);
			var SmoothStreamingMediaNodes = xml.GetElementsByTagName("SmoothStreamingMedia");
			if (SmoothStreamingMediaNodes.Count != 1)
				throw new System.Exception("Expected one SmoothStreamingMedia xml element but got " + SmoothStreamingMediaNodes.Count);
			var Stream = ParseManifest(SmoothStreamingMediaNodes[0]);
			return Stream;
		}
	}


}

