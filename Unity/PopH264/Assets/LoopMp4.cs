using System.Collections;
using System.Collections.Generic;
using UnityEngine;


[RequireComponent(typeof(Mp4))]
public class LoopMp4 : MonoBehaviour
{
	void OnEnable()
	{
		var Mp4 = GetComponent<Mp4>();
		Mp4.OnFinished.AddListener(OnMp4Finished);
	}

	void OnDisable()
	{
		var Mp4 = GetComponent<Mp4>();
		Mp4.OnFinished.RemoveListener(OnMp4Finished);
	}

	void OnMp4Finished()
	{
		if (!this.enabled)
			return;

		var Mp4 = GetComponent<Mp4>();
		Mp4.enabled = false;
		Mp4.enabled = true;
	}


}
