using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

[System.Serializable]
public class UnityEvent_Packet : UnityEngine.Events.UnityEvent<byte[],long> { }


//	this parses a Pcap packet
[RequireComponent(typeof(FileReaderBase))]
public class PcapParser : MonoBehaviour {

	public UnityEvent_Packet OnPacket;
	
	[Range(0, 20)]
	public int DecodePacketsPerFrame = 1;

	PopX.Pcap.GlobalHeader? Header = null;
	long FileBytesRead = 0;                          //	amount of data we've processed from the start of the asset, so we know correct file offsets
	System.Func<long, long, byte[]> ReadFileFunction;   //	if set, we use this to read data (eg, from memory buffer). Other
	
	void OnEnable()
	{
		//	get the file reader
		var FileReader = GetComponent<FileReaderBase>();
		FileBytesRead = 0;
		ReadFileFunction = FileReader.GetReadFileFunction();
	}

	long GetKnownFileSize()
	{
		var FileReader = GetComponent<FileReaderBase>();
		var Size = FileReader.GetKnownFileSize();
		return Size;
	}


	void ParseNextPacket()
	{
		//	check if there's more data to be read
		var KnownFileSize = GetKnownFileSize();
		if (FileBytesRead >= KnownFileSize)
			return;

		System.Action<byte[],int> EnumPacket = (Packet,Time) =>
		{
			OnPacket.Invoke(Packet, Time);
		};
			

		System.Func<long, byte[]> PopData = (long DataSize)=>
		{
			var Data = ReadFileFunction(FileBytesRead, DataSize);
			FileBytesRead += DataSize;
			return Data;
		};

		try
		{
			if (!Header.HasValue)
				Header = PopX.Pcap.ParseHeader(PopData);

			PopX.Pcap.ParseNextPacket(PopData,Header.Value,  EnumPacket);
		}
		catch(System.Exception e)
		{
			Debug.LogException(e);
		}
	}

	void OnDisable()
	{
		//	reset everything
		ReadFileFunction = null;
		FileBytesRead = 0;
	}


	void Update()
	{
		for (var i = 0; i < DecodePacketsPerFrame;	i++ )
			ParseNextPacket();
	}
	

}
