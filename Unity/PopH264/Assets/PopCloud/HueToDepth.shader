Shader "NSCCreative/HueToDepth"
{
	Properties
	{
		HueTexture("HueTexture", 2D) = "white" {}
		_Crop("_Crop", VECTOR) = (0,0,1,1)
		_ImageDimensions("_ImageDimensions", VECTOR) = (512,512,0,0)
		_FocalLength("_FocalLength", VECTOR) = (366,366,0,0)
		_PrincipalPoint("_PrincipalPoint", VECTOR) = (256,256,0,0)
		_NearClip("_NearClip", float) = 0.5
		_FarClip("_FarClip", float) = 1.25961
		[Toggle]Flip("Flip",Range(0,1) ) = 0
		[Toggle]Rotate90("Rotate90",Range(0,1)) = 1
		[Toggle]EnableLinearToGamma("EnableLinearToGamma",Range(0,1))=1
		[Toggle]_ShowDepthOutOfRange("_ShowDepthOutOfRange",Range(0,1)) = 0
		[Toggle]_ShowNegativeInput("_ShowNegativeInput",Range(0,1)) = 0
		[Toggle]_ShowOverOneInput("_ShowOverOneInput",Range(0,1)) = 0
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct v2f
			{
				float2 Sampleuv : TEXCOORD0;
				float2 Outuv : TEXCOORD1;
				float4 vertex : SV_POSITION;
				UNITY_VERTEX_OUTPUT_STEREO
			};

			sampler2D HueTexture;
			float4 HueTexture_TexelSize;
			float Flip;
			#define FLIP (Flip>0.5)
			float Rotate90;
			#define ROTATE90	(Rotate90>0.5)
			float EnableLinearToGamma;
			#define ENABLELINEARTOGAMMA	(EnableLinearToGamma>0.5)
			
			float _ShowDepthOutOfRange;
			float _ShowNegativeInput;
			float _ShowOverOneInput;
			#define ShowDepthOutOfRange	(_ShowDepthOutOfRange>0.5)
			#define ShowNegativeInput	(_ShowNegativeInput>0.5)
			#define ShowOverOneInput	(_ShowOverOneInput>0.5)

			
			v2f vert (appdata v)
			{
				v2f o;

				UNITY_SETUP_INSTANCE_ID(v); //Insert
				UNITY_INITIALIZE_OUTPUT(v2f, o); //Insert
				UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o); //Insert

				o.vertex = UnityObjectToClipPos(v.vertex);
				o.Sampleuv = v.uv;
				o.Outuv = v.uv;

				if ( FLIP )
					o.Sampleuv.y = 1 - o.Sampleuv.y;
				if ( ROTATE90 )
				{
					//	turn to -05..05 rotate and turn back
					float x = (-(o.Sampleuv.y-0.5)) + 0.5;
					float y = ((o.Sampleuv.x-0.5)) + 0.5;
					o.Sampleuv = float2(x,y);
				}
					
				return o;
			}
			
			// Hue value calculation
			float RGB2Hue(float3 c)
			{
			#if !defined(UNITY_COLORSPACE_GAMMA)
				if ( ENABLELINEARTOGAMMA )
			    	c = LinearToGammaSpace(c);
			#endif
				float minc = min(min(c.r, c.g), c.b);
			    float maxc = max(max(c.r, c.g), c.b);
			    
			    
			    float div = 1 / (6 * max(maxc - minc, 0.01));
			    float r = (c.g - c.b) * div;
			    float g = 1.0 / 3 + (c.b - c.r) * div;
			    float b = 2.0 / 3 + (c.r - c.g) * div;
			    float h = lerp(r, lerp(g, b, c.g < c.b), c.r < max(c.g, c.b));
			    
			    //	wrap around
			    h += (h<0) ? 1 : 0;
		    	return h;
			}

			// Depthkit metadata
			float4 _Crop;
			float2 _ImageDimensions;
			float2 _FocalLength;
			float2 _PrincipalPoint;
			float _NearClip;
			float _FarClip;
			float4x4 _Extrinsics;

			// Check if a depth sample is valid or not.
			bool ValidateDepth(float3 depthSample)
			{
			    return dot(depthSample, 1) > 0.3;
			}
			
			//	hue rgb to linear depth
			float RgbToDepth(float3 Rgb)
			{
				float Depth = RGB2Hue(Rgb);
				if ( !ValidateDepth(Rgb) )
				{
					Depth = 1;
				}
				return Depth;
			}

			// Object space position from depth sample
			float3 DepthToPosition(float2 coord, float DepthSample)
			{
			    coord = (coord * _Crop.zw + _Crop.xy) * _ImageDimensions - _PrincipalPoint;
			    
			    //	gr: depth is -1 to 1?
			    float d = DepthSample;
			    float z = lerp(_NearClip, _FarClip, d);
			    return mul(_Extrinsics, float4(coord * z / _FocalLength, z, 1)).xyz;
			}
			
			float Range(float Min,float Max,float Value)
			{
				return (Value-Min) / (Max-Min);
			}
			
			
			fixed4 frag (v2f i) : SV_Target
			{
				float3 Rgb = tex2D( HueTexture, i.Sampleuv );
				
				if ( ShowNegativeInput )
				{
					if ( Rgb.r < 0.0 )	return float4(1,0,0,1);
					if ( Rgb.g < 0.0 )	return float4(0,1,0,1);
					if ( Rgb.b < 0.0 )	return float4(0,0,1,1);
				}
				if ( ShowOverOneInput )
				{
					if ( Rgb.r > 1.0 )	return float4(1,0,0,1);
					if ( Rgb.g > 1.0 )	return float4(0,1,0,1);
					if ( Rgb.b > 1.0 )	return float4(0,0,1,1);
				}
				float Depth = RgbToDepth(Rgb);
				if ( ShowDepthOutOfRange )
				{
					if ( Depth < 0.0 )
						return float4(0,0,0,1);
					if ( Depth > 1.0 )
						return float4(1,1,1,1);
				}
						
				return float4(DepthToPosition(i.Outuv, Depth), 1);
			}
			ENDCG
		}
	}
}
