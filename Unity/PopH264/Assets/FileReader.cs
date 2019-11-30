using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;



public abstract class FileReaderBase : MonoBehaviour
{
	public abstract long GetKnownFileSize();	//	gr: this isn't actually file SIZE for streaming, but total known, so we know if there's no pending data
	public abstract System.Func<long, long, byte[]> GetReadFileFunction();
}


public class FileReader : FileReaderBase
{
	public string Filename = "Cat.mp4";

	byte[] FileBytes;

	public override long GetKnownFileSize()
	{
		return FileBytes!=null ? FileBytes.Length : 0;
	}

	override public System.Func<long, long, byte[]> GetReadFileFunction()
	{
		if (!System.IO.File.Exists(Filename))
		{
			var AssetFilename = "Assets/" + Filename;
			if (System.IO.File.Exists(AssetFilename))
				Filename = AssetFilename;

			var StreamingAssetsFilename = Application.streamingAssetsPath + "/" + Filename;
			if (System.IO.File.Exists(StreamingAssetsFilename))
				Filename = StreamingAssetsFilename;
		}

		if (!System.IO.File.Exists(Filename))
			throw new System.Exception("File missing: " + Filename);

		//	current hack, 
		FileBytes = System.IO.File.ReadAllBytes(Filename);

		return ReadFileBytes;
	}

	byte[] ReadFileBytes(long Position, long Size)
	{
		return FileBytes.SubArray(Position, Size);
	}

}

