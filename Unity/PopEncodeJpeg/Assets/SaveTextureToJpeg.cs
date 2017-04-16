using UnityEngine;
using System.Collections;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif
public class SaveTextureToJpeg : MonoBehaviour {

	#if UNITY_EDITOR
	[MenuItem("Assets/Texture/Save Texture to Jpeg")]
	static void _SaveTextureToJpeg()
	{
		//	get selected textures
		string[] AssetGuids = Selection.assetGUIDs;
		for (int i=0; i<AssetGuids.Length; i++) {
			string Guid = AssetGuids[i];
			string Path = AssetDatabase.GUIDToAssetPath (Guid);
			Texture Tex = AssetDatabase.LoadAssetAtPath( Path, typeof(Texture) ) as Texture;
			if ( !Tex )
				continue;

			DoSaveTextureToJpeg( Tex, Path, Guid );
		}
	}
	#endif

	#if UNITY_EDITOR
	static public bool DoSaveTextureToJpeg(Texture Tex,string AssetPath,string AssetGuid)
	{
		string Filename = EditorUtility.SaveFilePanel("save " + AssetPath, "", AssetGuid, "Jpeg");
		if ( Filename.Length == 0 )
			return false;

		return DoSaveTextureToJpeg( Tex, Filename );
	}
#endif

	static public Texture2D GetTexture2D(Texture Tex)
	{ 
		return GetTexture2D( Tex, TextureFormat.RGB24 );
	}

	static public Texture2D GetTexture2D(Texture Tex,TextureFormat Format)
	{ 
		if ( Tex is Texture2D )
			return Tex as Texture2D;
	
		//	copy to render texture and read
		RenderTexture rt = RenderTexture.GetTemporary( Tex.width, Tex.height, 0, RenderTextureFormat.ARGBFloat );
		Graphics.Blit( Tex, rt );
		Texture2D Temp = new Texture2D( rt.width, rt.height, Format, false );
		RenderTexture.active = rt;
		Temp.ReadPixels( new Rect(0,0,rt.width,rt.height), 0, 0 );
		Temp.Apply();
		RenderTexture.active = null;
		RenderTexture.ReleaseTemporary( rt );

		return Temp;
	}

	static public bool DoSaveTextureToJpeg(Texture Tex,string Filename)
	{
		var Temp = GetTexture2D( Tex );

		byte[] Bytes = PopEncodeJpeg.EncodeToJpeg( Temp );
		File.WriteAllBytes( Filename, Bytes );
		return true;
	}
}
