#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using PopX;

using UnityEditor;
using UnityEditor.IMGUI.Controls;


[ExecuteInEditMode]
public class Mp4Inspector : MonoBehaviour
{
	public List<string> Parts;
	public bool Changed = false;


	void EnumTracks(List<PopX.Mpeg4.TTrack> Tracks)
	{
		foreach (var Track in Tracks)
		{
			Parts.Add("Track");
		}
	}

	void EnumMdat(PopX.TAtom MdatAtom)
	{
		Parts.Add("Mdat");
	}

	void LoadMp4()
	{
		Parts = new List<string>();
		var FileReader = GetComponent<FileReaderBase>();
		long Mp4BytesRead = 0;
		var ReadFileFunction = FileReader.GetReadFileFunction();

		System.Func<long, byte[]> PopData = (long DataSize) =>
		{
			var Data = ReadFileFunction(Mp4BytesRead, DataSize);
			Mp4BytesRead += DataSize;
			return Data;
		};

		//	run until expception
		while (true)
		{
			try
			{
				PopX.Mpeg4.ParseNextAtom(PopData, Mp4BytesRead, EnumTracks, EnumMdat);
			}
			catch (System.Exception e)
			{
				Parts.Add("Exception: " + e.Message);
				break;
			}
		}
		Changed = true;
	}

	void OnEnable()
	{
		LoadMp4();
	}


}

class Mp4TreeView : TreeView
{
	Mp4Inspector Mp4;

	public Mp4TreeView(Mp4Inspector Mp4,TreeViewState treeViewState)
		: base(treeViewState)
	{
		this.Mp4 = Mp4;
		Reload();
	}

	protected override TreeViewItem BuildRoot()
	{
		var root = new TreeViewItem { id = 0, depth = -1, displayName = "Mp4" };
		var allItems = new List<TreeViewItem>();

		foreach (var Part in Mp4.Parts)
		{
			var Item = new TreeViewItem { id = 1, depth = 0, displayName = Part };
			allItems.Add(Item);
		}

		// Utility method that initializes the TreeViewItem.children and .parent for all items.
		SetupParentsAndChildrenFromDepths(root, allItems);

		// Return root of the tree
		return root;
	}
}


[CustomEditor(typeof(Mp4Inspector))]
public class Mp4InspectorEditor : Editor
{
	Mp4TreeView Mp4Tree;
	[SerializeField] 
	TreeViewState Mp4TreeState;

	public override void OnInspectorGUI()
	{
		if (Mp4TreeState == null)
			Mp4TreeState = new TreeViewState();

		var Mp4 = (target as Mp4Inspector);

		if (Mp4 == null)
		{
			EditorGUILayout.HelpBox("Mp4 is null", MessageType.Error);
			return;
		}

		if (Mp4.Changed)
			Mp4Tree = null;

		if ( Mp4Tree==null )
		{
			Mp4Tree = new Mp4TreeView(Mp4,Mp4TreeState);
		}
		Mp4.Changed = false;

		GUILayoutOption[] Options = { GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true), GUILayout.MinHeight(300) };

		var Rect = EditorGUILayout.GetControlRect(Options);
		Mp4Tree.OnGUI(Rect);
	

	}

}
#endif