using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

//[System.Serializable]
//public class UnityEvent_Packet : UnityEngine.Events.UnityEvent<byte[],long> { }


//	gr: this isn't VAAPI, but I don't know what the transport is
//		rename when we find out!
namespace PopX
{
	public static class Vaapi
	{
		public struct PendingPacket
		{
			public int FrameNumber;
			public List<byte> Data;
			public List<int> PartNumbers;	//	keep a list of parts we've processed, should just increment
		}

		static public PendingPacket? ParseNextPacket(System.Func<long,byte[]> ReadData, PendingPacket? PendingPacket,System.Action<byte[],long> EnumPacket)
		{
			//	first byte is incrementing
			var FrameNumber = ReadData(1)[0];

			//	this isn't a length, but it's consistent per frame number
			var FrameLength = PopX.Mpeg4.Get24(ReadData(3));
			
			//	next seems to be a byte that increments (Part X/N?)
			var PartNumber = ReadData(1)[0];

			//	next 3 (or X and 2) is the length of this part
			var PartLength3 = ReadData(3);
			var PartLength = PopX.Mpeg4.Get24(PartLength3[0], PartLength3[2], PartLength3[1]);

			Debug.Log("Frame " + FrameNumber + "x" + FrameLength + " Part " + PartNumber + "x" + PartLength);
			
			//	grab the rest of the data (this should match what's left?)
			var PacketData = ReadData(PartLength);

			//	gr: how do we detect last part?
			bool EndOfFrame = false;
			bool FlushFrame = EndOfFrame;

			//	flush last packet if the frame number has changed
			if (PendingPacket.HasValue && PendingPacket.Value.FrameNumber != FrameNumber)
			{
				var LastPacket = PendingPacket.Value;
				PendingPacket = null;
				EnumPacket(LastPacket.Data.ToArray(), LastPacket.FrameNumber);
			}
			
			//	if we dont have a pending packet, we should be starting with part0
			if (!PendingPacket.HasValue)
			{
				if (PartNumber != 0)
				{
					throw new System.Exception("Got frame " + FrameNumber + " part" + PartNumber + ", but no pending packet so expecting part 0");
					return null;
				}

				//	new packet
				var NewPacket = new PendingPacket();
				NewPacket.FrameNumber = FrameNumber;
				NewPacket.Data = new List<byte>();
				NewPacket.PartNumbers = new List<int>();
				PendingPacket = NewPacket;
			}

			//	check order
			{
				var PartNumbers = PendingPacket.Value.PartNumbers;
				var LastPartNumber = -1;
				if (PartNumbers.Count > 0)
					LastPartNumber = PartNumbers[PartNumbers.Count - 1];
				if (LastPartNumber != PartNumber - 1)
				{
					throw new System.Exception("Packet part numbers are out of order");
					return null;
				}
			}

			//	append data
			PendingPacket.Value.Data.AddRange(PacketData);
			PendingPacket.Value.PartNumbers.Add(PartNumber);

			//	need to flush
			if (FlushFrame)
			{
				var LastPacket = PendingPacket.Value;
				PendingPacket = null;
				EnumPacket(LastPacket.Data.ToArray(), LastPacket.FrameNumber);
			}

			return PendingPacket;
		}
	}
}


		



public class VaapiParser : MonoBehaviour {

	public UnityEvent_Packet OnPacket;
	
	[Range(0, 20)]
	public int DecodePacketsPerFrame = 1;

	List<PopH264.FrameInput> PendingPackets;
	PopX.Vaapi.PendingPacket? PendingPacket = null;
	

	void OnEnable()
	{
	}

	long GetKnownFileSize()
	{
		var FileReader = GetComponent<FileReaderBase>();
		var Size = FileReader.GetKnownFileSize();
		return Size;
	}


	void ParseNextPacket()
	{
		if (PendingPackets == null)
			return;
		if (PendingPackets.Count == 0)
			return;

		var NextPacket = PendingPackets[0];
		PendingPackets.RemoveAt(0);

		long DataRead = 0;
		System.Func<long,byte[]> PopData = (Length)=>
		{
			var Data = NextPacket.Bytes.SubArray(DataRead, Length);
			DataRead += Length;
			return Data;
		};

		System.Action<byte[],long> EnumPacket = (Bytes, Time) =>
		{
			OnPacket.Invoke(Bytes, Time);
		};

		PendingPacket = PopX.Vaapi.ParseNextPacket(PopData, PendingPacket, EnumPacket);
	}

	void OnDisable()
	{
		PendingPackets = null;
	}


	void Update()
	{
		for (var i = 0; i < DecodePacketsPerFrame;	i++ )
			ParseNextPacket();
	}
	
	public void PushPacket(byte[] Data,long TimeStamp)
	{
		if (PendingPackets == null)
			PendingPackets = new List<PopH264.FrameInput>();

		var NewPacket = new PopH264.FrameInput();
		NewPacket.Bytes = Data;
		NewPacket.FrameNumber = (int)TimeStamp;
		PendingPackets.Add(NewPacket);
	}

}
