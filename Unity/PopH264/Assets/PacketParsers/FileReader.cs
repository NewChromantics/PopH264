//#if NET_4_6	//	gr: not working?
#if UNITY_ANDROID
#define USE_JAVA_FILEHANDLE
#else
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

	public UnityEvent_Packet OnPacketRecieved;

	protected void OnDataChanged()
	{
		//	currently sends everything, need to change this to fit streaming formats
		var ReadFunc = GetReadFileFunction();
		var FileSize = GetKnownFileSize();
		if (FileSize > 0)
		{
			var FrameNumber = 0;
			var Data = ReadFunc(0, FileSize);
			OnPacketRecieved.Invoke(Data, FrameNumber);
		}
	}
}



public class FileReader : FileReaderBase
{
	public string Filename = "Cat.mp4";

	[Header("On first update, send all bytes. Best disabled on very large files")]
	public bool EnableOnPacket = true;

#if USE_MEMORY_MAPPED_FILE
	MemoryMappedFile File;
	MemoryMappedViewAccessor FileView;
	long FileSize;			//	the file view capcacity is bigger than the file size (page aligned) so we need to know the proper size
#elif USE_FILE_HANDLE
	System.IO.FileStream File;
#elif USE_JAVA_FILEHANDLE
	PopAndroidFileReader File;
#endif
	byte[] FileBytes;		//	"other" mode reads all bytes. We also use this for dumping from Resources.Load

	bool FirstUpdate = true;

	void OnEnable()
	{
		//	reset if re-enabled
		FirstUpdate = true;
	}

	//	gr: move this to Base and trigger when we get more data in
	void Update()
	{
		if (!FirstUpdate)
			return;

		if (EnableOnPacket)
			OnDataChanged();

		FirstUpdate = false;
	}

	public override long GetKnownFileSize()
	{
		if (FileBytes != null)
			return FileBytes.Length;

#if USE_MEMORY_MAPPED_FILE
		//return FileView.Capacity;
		return FileSize;
#elif USE_FILE_HANDLE
		return File.Length;
#elif USE_JAVA_FILEHANDLE
		return File != null ? File.GetFileSize() : 0;
#else
		return FileBytes != null ? FileBytes.Length : 0;
#endif
	}

	//	gr: this is currently specific to files (assume other readers can't), but maybe wants a generic interface anyway
	//		OnDataChanged shouldn't really re-set/end everything anyway
	public void ResetStream()
	{
		FirstUpdate = true;
	}

	override public System.Func<long, long, byte[]> GetReadFileFunction()
	{
		//	catch files in resources
		if (Filename.StartsWith("Resources/") || Filename.StartsWith("Resources\\"))
		{
			var ResourceFilename = Filename;
			//	remove resources path
			ResourceFilename = ResourceFilename.Remove(0, "Resources/".Length);
			//	remove .bytes extension
			//	gr: should also remove .text extension and... any last extension?
			var RemoveExtension = ".bytes";
			ResourceFilename = ResourceFilename.Remove(ResourceFilename.Length - RemoveExtension.Length, RemoveExtension.Length);

			var Asset = Resources.Load<TextAsset>(ResourceFilename);
			if (Asset != null)
			{
				FileBytes = Asset.bytes;
				Debug.Log("Loaded " + ResourceFilename + "(" + Filename + ") from resources as bytes x" + FileBytes.Length);
				return ReadFileBytes;
			}
			else
			{
				Debug.LogWarning("Tried to open file (" + ResourceFilename + ") as resource but failed");
			}
		}

		if (!System.IO.File.Exists(Filename))
		{
			var AssetFilename = "Assets/" + Filename;
			if (System.IO.File.Exists(AssetFilename))
			{
				Debug.Log("Assets/ filename exists:" + AssetFilename);
				Filename = AssetFilename;
			}

			var StreamingAssetsFilename = Application.streamingAssetsPath + "/" + Filename;
			if (System.IO.File.Exists(StreamingAssetsFilename))
			{
				Debug.Log("Streaming assets filename exists:" + StreamingAssetsFilename);
				Filename = StreamingAssetsFilename;
			}

		}

#if USE_JAVA_FILEHANDLE
		//	dont check file exists on android
#else
		if (!System.IO.File.Exists(Filename))
			throw new System.Exception("File missing: " + Filename);
#endif

		Debug.Log("FileReader opening file " + Filename + " (length = " + Filename.Length + ")");

#if USE_MEMORY_MAPPED_FILE
		Debug.Log("Creating Memory mapped file");
		File = MemoryMappedFile.CreateFromFile(Filename,System.IO.FileMode.Open);
		Debug.Log("Memory mapped file = "+ File);
		FileView = File.CreateViewAccessor();
		Debug.Log("Memory mapped FileView = " + FileView);
		FileSize = new System.IO.FileInfo(Filename).Length;
		Debug.Log("Memory mapped FileSize = " + FileSize);
#elif USE_FILE_HANDLE
		File = System.IO.File.OpenRead(Filename);
#elif USE_JAVA_FILEHANDLE
		File = new PopAndroidFileReader(Filename);
#else
		FileBytes = System.IO.File.ReadAllBytes(Filename);
#endif

		return ReadFileBytes;
	}

	byte[] ReadFileBytes(long Position, long Size)
	{
		if ( FileBytes != null )
			return FileBytes.SubArray(Position, Size);

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
#elif USE_JAVA_FILEHANDLE
		return File.ReadBytes(Position, Size);
#else
		return FileBytes.SubArray(Position, Size);
#endif
	}

}

