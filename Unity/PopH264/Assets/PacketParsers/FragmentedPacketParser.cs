using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

//[System.Serializable]
//public class UnityEvent_Packet : UnityEngine.Events.UnityEvent<byte[],long> { }


//	gr: SOMETHING is fragmenting these packets. Seems to be from the transport, but not always the same
namespace PopX
{
	public static class FragmentedPacket
	{
		[System.Serializable]
		public struct FragmentedOptions
		{
			//	sets of packets
			//		Frame8_Checksum24_Part8_Length24
			//		Part8
			public bool HasFrameNumberAndChecksumAndSize;
			public bool HasPartNumber;
			public bool HasPartSize { get { return HasFrameNumberAndChecksumAndSize; } }
			public bool HasFrameNumber { get { return HasFrameNumberAndChecksumAndSize; } }
		}

		public struct PendingPacket
		{
			public int FrameNumber;
			public List<byte> Data;
			public List<int> PartNumbers;	//	keep a list of parts we've processed, should just increment
		}

		
		static public PendingPacket? ParseNextPacket(System.Func<long,byte[]> ReadData, int PacketSize,FragmentedOptions Options, PendingPacket? PendingPacket,System.Action<byte[],long> EnumPacket)
		{
			var FrameNumber = 0;
			int? PartLength = null;
			var PartNumber = 0;
			long BytesRead = 0;
			System.Func<long,byte[]> PopData = (Length)=>
			{
				var Data = ReadData(Length);
				BytesRead += Length;
				return Data;
			};

			if (Options.HasFrameNumberAndChecksumAndSize )
			{
				//	first byte is incrementing
				FrameNumber = PopData(1)[0];

				//	this isn't a length, but it's consistent per frame number (checksum?)
				var Checksum = PopX.Mpeg4.Get24(ReadData(3));
			}

			//	next seems to be a byte that increments (Part X/N?)
			if ( Options.HasPartNumber )
				PartNumber = PopData(1)[0];

			if (Options.HasPartSize)
			{
				//	next 3 (or X and 2) is the length of this part
				var PartLength3 = PopData(3);
				PartLength = PopX.Mpeg4.Get24(PartLength3[0], PartLength3[2], PartLength3[1]);
			}
			
			if ( !PartLength.HasValue )
			{
				PartLength = PacketSize - (int)BytesRead;
			}

			Debug.Log("Frame " + FrameNumber + "x" + PartLength.Value + " Part " + PartNumber + "x" + PartLength);
			
			//	grab the rest of the data (this should match what's left?)
			var PacketData = ReadData(PartLength.Value);

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
					FlushFrame = true;
					//	gr: only throw if we have a frame, otherwise its how we know we're on a new fragment
					if ( Options.HasFrameNumber)
						throw new System.Exception("Packet part numbers are out of order");
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


		



public class FragmentedPacketParser : MonoBehaviour {

	public UnityEvent_Packet OnPacket;
	
	[Range(0, 100)]
	public int DecodePacketsPerFrame = 1;

	[Header("Some fragments have lengths, but no flags in PCAP to indicate it came from there")]
	public PopX.FragmentedPacket.FragmentedOptions FragmentedOptions;

	List<PopH264.FrameInput> PendingPackets;
	PopX.FragmentedPacket.PendingPacket? CurrentPacket = null;
	

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

		CurrentPacket = PopX.FragmentedPacket.ParseNextPacket(PopData, NextPacket.Bytes.Length, FragmentedOptions, CurrentPacket, EnumPacket);
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


	public void PushPacketWithNoTimestamp(byte[] Data)
	{
		PushPacket(Data, 0);
	}

}
