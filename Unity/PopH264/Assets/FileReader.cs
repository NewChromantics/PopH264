//#if NET_4_6	//	gr: not working?
#if true
#define USE_MEMORY_MAPPED_FILE
#endif
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;


#if USE_MEMORY_MAPPED_FILE
//	.net4 required
using System.IO.MemoryMappedFiles;
#endif

public abstract class FileReaderBase : MonoBehaviour
{
	public abstract long GetKnownFileSize();    //	gr: this isn't actually file SIZE for streaming, but total known, so we know if there's no pending data
	public abstract System.Func<long, long, byte[]> GetReadFileFunction();
}


public class FileReader : FileReaderBase
{
	public string Filename = "Cat.mp4";

#if USE_MEMORY_MAPPED_FILE
	MemoryMappedFile File;
	MemoryMappedViewAccessor FileView;
	long FileSize;			//	the file view capcacity is bigger than the file size (page aligned) so we need to know the proper size
#else
	byte[] FileBytes;
#endif

	public override long GetKnownFileSize()
	{
#if USE_MEMORY_MAPPED_FILE
		//return FileView.Capacity;
		return FileSize;
#else
		return FileBytes != null ? FileBytes.Length : 0;
#endif
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

#if USE_MEMORY_MAPPED_FILE
		File = MemoryMappedFile.CreateFromFile(Filename,System.IO.FileMode.Open);
		FileView = File.CreateViewAccessor();
		FileSize = new System.IO.FileInfo(Filename).Length;
#else
		FileBytes = System.IO.File.ReadAllBytes(Filename);
#endif

		return ReadFileBytes;
	}

	byte[] ReadFileBytes(long Position, long Size)
	{
#if USE_MEMORY_MAPPED_FILE
		var Data = new byte[Size];
		//	gr: [on OSX at least] you can read past the file size, (but within capacity)
		//		this doesn't error, but does fill the bytes with zeros.
		var BytesRead = FileView.ReadArray(Position, Data, 0, (int)Size);
		if (BytesRead != Size)
			throw new System.Exception("Memory mapped file only read " + BytesRead + "/" + Size + " bytes");
		return Data;
#else
		return FileBytes.SubArray(Position, Size);
#endif
	}

}

