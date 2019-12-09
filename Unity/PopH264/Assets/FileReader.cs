//#if NET_4_6	//	gr: not working?
#if true
//#define USE_MEMORY_MAPPED_FILE
#define USE_FILE_HANDLE
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
#elif USE_FILE_HANDLE
	System.IO.FileStream File;
#else
	byte[] FileBytes;
#endif

	public void Reset()
	{
#if USE_MEMORY_MAPPED_FILE
		File = null;
		FileView = null;
		long FileSize = 0;
#elif USE_FILE_HANDLE
		File = null;
#else
		FileBytes = null;
#endif
	}

	public override long GetKnownFileSize()
	{
#if USE_MEMORY_MAPPED_FILE
		//return FileView.Capacity;
		return FileSize;
#elif USE_FILE_HANDLE
		return File.Length;
#else
		return FileBytes != null ? FileBytes.Length : 0;
#endif
	}

	override public System.Func<long, long, byte[]> GetReadFileFunction()
	{
		Reset();
		var FullFilename = Filename;

		if (!System.IO.File.Exists(FullFilename))
		{
			var AssetFilename = "Assets/" + FullFilename;
			if (System.IO.File.Exists(AssetFilename))
				FullFilename = AssetFilename;

			var StreamingAssetsFilename = Application.streamingAssetsPath + "/" + FullFilename;
			if (System.IO.File.Exists(StreamingAssetsFilename))
				FullFilename = StreamingAssetsFilename;
		}

		if (!System.IO.File.Exists(FullFilename))
			throw new System.Exception("File missing: " + FullFilename);

		Debug.Log("FileReader opening file " + FullFilename + " (length = " + FullFilename.Length + ")");

#if USE_MEMORY_MAPPED_FILE
		Debug.Log("Creating Memory mapped file");
		File = MemoryMappedFile.CreateFromFile(Filename,System.IO.FileMode.Open);
		Debug.Log("Memory mapped file = "+ File);
		FileView = File.CreateViewAccessor();
		Debug.Log("Memory mapped FileView = " + FileView);
		FileSize = new System.IO.FileInfo(FullFilename).Length;
		Debug.Log("Memory mapped FileSize = " + FileSize);
#elif USE_FILE_HANDLE
		File = System.IO.File.OpenRead(FullFilename);
#else
		FileBytes = System.IO.File.ReadAllBytes(FullFilename);
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
#elif USE_FILE_HANDLE
		var Data = new byte[Size];
		var NewPos = File.Seek(Position, System.IO.SeekOrigin.Begin);
		if (NewPos != Position)
			throw new System.Exception("Seeked to " + Position + " but stream is at " + NewPos);
		var BytesRead = File.Read( Data, 0, (int)Size);
		if (BytesRead != Size)
			throw new System.Exception("FileStream only read " + BytesRead + "/" + Size + " bytes");
		return Data;
#else
		return FileBytes.SubArray(Position, Size);
#endif
	}

}

