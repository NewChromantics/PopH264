using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RawH264Decoder : MonoBehaviour
{
    PopH264.Decoder Decoder;
    List<Texture2D> FramePlanes;
    List<PopH264.PixelFormat> FramePlaneFormats;
    public bool ThreadedDecoding = true;
    public PopH264.DecoderParams DecoderParams;

    public void PushData(byte[] Data,long TimeStamp)
    {
        if ( Decoder == null )
            Decoder = new PopH264.Decoder(DecoderParams,ThreadedDecoding);

        Debug.Log("pushing x" + Data.Length);
        Decoder.PushFrameData(Data,(int)TimeStamp);
    }

    void Update()
    {
        if ( Decoder == null )
            return;

        var NewFrame = Decoder.GetNextFrame( ref FramePlanes, ref FramePlaneFormats );
        if ( NewFrame.HasValue )
            Debug.Log("Decoded frame " + NewFrame.Value );
        
    }
}
