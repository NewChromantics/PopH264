using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class BufferReader : FileReaderBase
{
	struct BufferMeta
	{
		public byte[] Data;
		public long FilePosition;
	};
	List<BufferMeta> Buffers;
	long CurrentFilePos = 0;

	public void PushData(byte[] Data)
	{
		if (Buffers == null)
			Buffers = new List<BufferMeta>();

		//	file pos is end of last buffer
		var Buffer = new BufferMeta();
		Buffer.Data = Data;
		Buffer.FilePosition = CurrentFilePos;
		Buffers.Add(Buffer);

		CurrentFilePos += Buffer.Data.Length;
        Debug.Log("buffer count " + Buffers.Count);
	}

	public void Reset()
	{
		Buffers = null;
		CurrentFilePos = 0;
	}

	public override long GetKnownFileSize()
	{
		//	return the length up to the end of the next chunk
		if (Buffers == null)
			return CurrentFilePos;
		if (Buffers.Count == 0)
			return CurrentFilePos;

		var Buffer0 = Buffers[0];
		return Buffer0.FilePosition + Buffer0.Data.LongLength;
	}

	override public System.Func<long, long, byte[]> GetReadFileFunction()
	{
		Reset();
		return ReadFileBytes;
	}

	byte[] ReadFileBytes(long Position, long Size)
	{
		//	find buffer (should be first)
		if (Buffers == null)
			throw new System.Exception("No data buffered");
		if ( Buffers.Count==0)
			throw new System.Exception("No new data buffered");
	
		var Buffer0 = Buffers[0];
		if (Position < Buffer0.FilePosition)
			throw new System.Exception("Requesting data at " + Position + " already discarded");
		if (Position != Buffer0.FilePosition)
			throw new System.Exception("Requesting data at " + Position + " somewhere other than next buffer " + Buffer0.FilePosition);

		Buffers.RemoveAt(0);
		return Buffer0.Data;
	}

}

