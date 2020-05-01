using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using System.Reflection;
#endif

[System.AttributeUsage(System.AttributeTargets.Field)]
public class ShowIfAttribute : PropertyAttribute
{
	public readonly string	FunctionName;

	public ShowIfAttribute(string _FunctionName)
	{
		this.FunctionName = _FunctionName;
	}


}


#if UNITY_EDITOR
[CustomPropertyDrawer(typeof(ShowIfAttribute))]
public class PopPlatformAttributePropertyDrawer : PropertyDrawer
{
	private MethodInfo CachedEventMethodInfo = null;
	
	
	bool IsVisible(Object TargetObject,ShowIfAttribute Attrib)
	{
		var TargetObjectType = TargetObject.GetType();

		if (CachedEventMethodInfo == null)
			CachedEventMethodInfo = TargetObjectType.GetMethod(Attrib.FunctionName, BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);

		if (CachedEventMethodInfo != null) {
			var Result = CachedEventMethodInfo.Invoke (TargetObject, null);
			var ResultType = (Result == null) ? "null" : Result.GetType ().Name;
			try
			{
				var ResultBool = (bool)Result;
				return ResultBool;
			}
			catch(System.Exception e) {
				Debug.LogWarning ("Failed to get event " + Attrib.FunctionName + " in " + TargetObjectType + " result as bool (is " + ResultType + "); " + e.Message );
			}
		}

		Debug.LogWarning("ShowIfAttribute: Unable to find method "+ Attrib.FunctionName + " in " + TargetObjectType);
		return false;
	}

	
	public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
	{
		var Attrib = (ShowIfAttribute)attribute;
		var TargetObject = property.serializedObject.targetObject;

		if ( IsVisible(TargetObject,Attrib)) {
			//base.OnGUI (position, prop, label);
			EditorGUI.PropertyField (position, property, label, true);
		}
	}

	public override float GetPropertyHeight (SerializedProperty property, GUIContent label)
	{
		var Attrib = (ShowIfAttribute)attribute;
		var TargetObject = property.serializedObject.targetObject;

		if ( IsVisible(TargetObject,Attrib)) {
			//base.OnGUI (position, prop, label);
			return base.GetPropertyHeight ( property,  label);
		}
		return 0;
	}

}
#endif
