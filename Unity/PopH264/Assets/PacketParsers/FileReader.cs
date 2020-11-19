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

public class JavaFileReader
{
	public JavaFileReader(string InternalFilename, string JarOrZipOrApkFilename=null)
	{
		//	default to app's apk
		//	 "jar: file://" + Application.dataPath + "!/assets/" + Filename;
		if (JarOrZipOrApkFilename == null || JarOrZipOrApkFilename.Length==0)
			JarOrZipOrApkFilename = Application.dataPath;

		//InternalFilename = "!/assets/" + InternalFilename;
		InternalFilename = "assets/" + InternalFilename;

		AndroidJavaObject ZipFile;
		try
		{
			//	"com.android.vending.expansion.zipfile.ZipResourceFile"
			//	"com.google.android.vending.expansion.zipfile"
			var PackageName = "com.google.android.vending.expansion.zipfile";   //	package name inside the .java
			var ClassName = "ZipResourceFile";  //	class inside .java
			ZipFile = new AndroidJavaObject(PackageName + "." + ClassName, JarOrZipOrApkFilename);
		}
		catch (AndroidJavaException e)
		{
			throw new System.Exception("Failed to open zip file (" + JarOrZipOrApkFilename + "): " + e.Message);
		}
		/*
		var StringSig = "Ljava/lang/String;";
		var VoidSig = "V";

		var ZipClass = AndroidJNI.FindClass("com.android.vending.expansion.zipfile.ZipResourceFile");
		var ConstructorName = "<init>";
		var ConstructorSignature = "("+StringSig+")" + VoidSig +"";    //	1 string arg, return void
		var ConstructorMethod = AndroidJNI.GetMethodID(ZipClass, ConstructorName, ConstructorSignature);
		var JarOrZipOrApkFilename_j = AndroidJNI.NewString(JarOrZipOrApkFilename);

		var ConstructorArguments = new jvalue[1];
		ConstructorArguments[0].l = JarOrZipOrApkFilename_j;	//	.l = L in signature
		ZipFile = AndroidJNI.NewObject(ZipClass, ConstructorMethod, ConstructorArguments);
		if (ZipFile==System.IntPtr.Zero)
		*/
		if ( ZipFile == null)
			throw new System.Exception("Failed to open zip/jar/apk " + JarOrZipOrApkFilename);

		var GetAssetFileDescriptorName = "getAssetFileDescriptor";
		AndroidJavaObject FileDescriptor;
		try
		{
			FileDescriptor = ZipFile.Call<AndroidJavaObject>(GetAssetFileDescriptorName, InternalFilename);
			/*
			var AssetFileDescriptorClassName = "android/content/res/AssetFileDescriptor";
			var AssetFileDescriptorSig = "L" + AssetFileDescriptorClassName + ";";
			var getAssetFileDescriptorSignature = "(" + StringSig + ")" + AssetFileDescriptorSig + "";
			var getAssetFileDescriptorMethod = AndroidJNI.GetMethodID(ZipClass, GetAssetFileDescriptorName, getAssetFileDescriptorSignature);
			var InternalFilename_j = AndroidJNI.NewString(InternalFilename);
			var GetAssetFileDescriptorArguments = new jvalue[1];
			GetAssetFileDescriptorArguments[0].l = InternalFilename_j;
			var FileDescriptor = AndroidJNI.CallObjectMethod(ZipFile, getAssetFileDescriptorMethod, GetAssetFileDescriptorArguments);
			//	gr: returns a null object when filenot found (no exception set in java vm!)
			if (FileDescriptor == System.IntPtr.Zero)
			*/
			if (FileDescriptor == null)
				throw new System.Exception("Opened zip but failed to open " + InternalFilename);
		}
		catch (AndroidJavaException e)
		{
			throw new System.Exception("Failed to open file inside zip file (" + InternalFilename + "): " + e.Message);
		}

		throw new System.Exception("Got file descriptor");
	}

	public int GetFileSize()
	{
		return 0;
	}

	public byte[] ReadBytes(long Position, long Size)
	{
		return null;
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
	JavaFileReader File;
#else
	byte[] FileBytes;
#endif
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
		File = new JavaFileReader(Filename);
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

