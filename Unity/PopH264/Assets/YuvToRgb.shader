Shader "NSCCreative/YuvToRgb"
{
	Properties
	{
		Source_Top("Source_Top", Range(0,1) ) = 0
		Source_Bottom("Source_Bottom", Range(0,1) ) = 1
		Source_Left("Source_Left", Range(0,1) ) = 0
		Source_Right("Source_Right", Range(0,1) ) = 1
	
		[Space]LumaTexture ("LumaTexture", 2D) = "white" {}
		[Enum(None,0,Greyscale,1,YYuv_8888_Full,18,YYuv_8888_Ntsc,19)]LumaFormat("LumaFormat",int) = 0
		ChromaUTexture ("ChromaUTexture", 2D) = "black" {}
		[Enum(Debug,999,None,0,ChromaUV_88,25,ChromaVU_88,998,Chroma_U,26,Chroma_V,27)]ChromaUFormat("ChromaUFormat",int) = 0
		ChromaVTexture ("ChromaVTexture", 2D) = "black" {}
		[Enum(Debug,999,None,0,Chroma_U,26,Chroma_V,27)]ChromaVFormat("ChromaVFormat",int) = 0
		
		[Header(NTSC etc colour settings)]LumaMin("LumaMin", Range(0,255) ) = 16
		LumaMax("LumaMax", Range(0,255) ) = 253
		ChromaVRed("ChromaVRed", Range(-2,2) ) = 1.5958
		ChromaUGreen("ChromaUGreen", Range(-2,2) ) = -0.39173
		ChromaVGreen("ChromaVGreen", Range(-2,2) ) = -0.81290
		ChromaUBlue("ChromaUBlue", Range(-2,2) ) = 2.017
		[Toggle]Flip("Flip", Range(0,1)) = 1
		[Toggle]EnableChroma("EnableChroma", Range(0,1)) = 1
		
		
		
		//	depth kit
		[Toggle]InputIsHueDepth("InputIsHueDepth",Range(0,1) ) = 0
		_Crop("_Crop", VECTOR) = (0,0,1,1)
		_ImageDimensions("_ImageDimensions", VECTOR) = (512,512,0,0)
		_FocalLength("_FocalLength", VECTOR) = (366,366,0,0)
		_PrincipalPoint("_PrincipalPoint", VECTOR) = (256,256,0,0)
		_NearClip("_NearClip", float) = 0.5
		_FarClip("_FarClip", float) = 1.25961
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

			sampler2D LumaTexture;
			sampler2D ChromaUTexture;
			sampler2D ChromaVTexture;
			float4 LumaTexture_TexelSize;
			int LumaFormat;
			int ChromaUFormat;
			int ChromaVFormat;

			float LumaMin;
			float LumaMax;
			float ChromaVRed;
			float ChromaUGreen;
			float ChromaVGreen;
			float ChromaUBlue;

			//	SoyPixelsFormat's 
			//	see https://github.com/SoylentGraham/SoyLib/blob/master/src/SoyPixels.h#L16
		#define Debug		999
		#define None		0
		#define Greyscale	1
		#define RGB			3
		#define RGBA		4
		#define YYuv_8888_Full	18
		#define YYuv_8888_Ntsc	19
		#define ChromaUV_88	25
		#define Chroma_U	27
		#define Chroma_V	28
		#define ChromaVU_88	998

			float Flip;
			float EnableChroma;
			#define FLIP	( Flip > 0.5f )	
			#define ENABLE_CHROMA	( EnableChroma > 0.5f )
			
			float Source_Top;
			float Source_Bottom;
			float Source_Left;
			float Source_Right;
			
			float InputIsHueDepth;
			
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

				o.Sampleuv.x = lerp( Source_Left, Source_Right, o.Sampleuv.x );				
				o.Sampleuv.y = lerp( Source_Top, Source_Bottom, o.Sampleuv.y );				
		
					
				return o;
			}

			float2 GetChromaUv_88(float2 uv)
			{
				//	uv in one plane but organised as 2-component texture
				float2 ChromaUV = tex2D(ChromaUTexture, uv).xy;
				return ChromaUV;
			}

			float2 GetChromaVu_88(float2 uv)
			{
				//	uv in one plane but organised as 2-component texture
				float2 ChromaUV = tex2D(ChromaUTexture, uv).yx;
				return ChromaUV;
			}

			float2 GetChromaUv_Debug(float2 uv)
			{
				return uv;
			}

			float2 GetChromaUv_8_8(float2 uv)
			{
				//	seperate planes
				float ChromaU = tex2D(ChromaUTexture, uv);
				float ChromaV = tex2D(ChromaVTexture, uv);
				return float2(ChromaU, ChromaV);
			}

			void GetLumaChromaUv_8888(float2 uv,out float Luma,out float2 ChromaUV)
			{
				//	data is 
				//	LumaX+0, ChromaU+0, LumaX+1, ChromaV+0
				//	2 lumas for each chroma 
				float2 x = fmod(uv.x * LumaTexture_TexelSize.z, 2.0);
				float uRemainder = x * LumaTexture_TexelSize.x;
				
				//	uv0 = left pixel of pair
				float2 uv0 = uv;
				uv0.x -= uRemainder;
				//	uv1 = right pixel of pair
				float2 uv1 = uv0;
				uv1.x += LumaTexture_TexelSize.x;

				//	just in case, sample from middle of texel!
				uv0.x += LumaTexture_TexelSize.x * 0.5;
				uv1.x += LumaTexture_TexelSize.x * 0.5;

				float ChromaU = tex2D(LumaTexture, uv0).y;
				float ChromaV = tex2D(LumaTexture, uv1).y;
				Luma = tex2D(LumaTexture, uv).x;
				ChromaUV = float2(ChromaU, ChromaV);
			}
			
			
			//	from https://github.com/keijiro/Dkvfx/blob/master/Assets/Dkvfx/Shader/Common.hlsl
			//	gr: todo: convert to a simple matrix
			// Depthkit UV space
			float2 DepthUV(float2 coord)
			{
			    return float2(coord.x, 1 - coord.y / 2);
			}

			float2 ColorUV(float2 coord)
			{
			    return float2(coord.x, 0.5 - coord.y / 2);
			}

			// Hue value calculation
			fixed RGB2Hue(fixed3 c)
			{
			#if !defined(UNITY_COLORSPACE_GAMMA)
			    c = LinearToGammaSpace(c);
			#endif
			    fixed minc = min(min(c.r, c.g), c.b);
			    fixed maxc = max(max(c.r, c.g), c.b);
			    half div = 1 / (6 * max(maxc - minc, 1e-5));
			    half r = (c.g - c.b) * div;
			    half g = 1.0 / 3 + (c.b - c.r) * div;
			    half b = 2.0 / 3 + (c.r - c.g) * div;
			    return lerp(r, lerp(g, b, c.g < c.b), c.r < max(c.g, c.b));
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

			// Object space position from depth sample
			float3 DepthToPosition(float2 coord, float3 depthSample)
			{
			    coord = (coord * _Crop.zw + _Crop.xy) * _ImageDimensions - _PrincipalPoint;
			    float d = ValidateDepth(depthSample) ? RGB2Hue(depthSample) : 1;
			    float z = lerp(_NearClip, _FarClip, d);
			    return mul(_Extrinsics, float4(coord * z / _FocalLength, z, 1)).xyz;
			}
			
			float Range(float Min,float Max,float Value)
			{
				return (Value-Min) / (Max-Min);
			}
			
			float3 YuvToRgb(float2 uv)
			{
				// sample the texture
				float3 Luma3 = tex2D(LumaTexture, uv);
				float Luma = Luma3.x;
			
				float2 ChromaUV = float2(0, 0);
				if ( LumaFormat == YYuv_8888_Full || LumaFormat == YYuv_8888_Ntsc )
				{
					GetLumaChromaUv_8888(uv, Luma, ChromaUV);
				}
				else if ( ChromaUFormat == Debug )
				{
					ChromaUV = GetChromaUv_Debug(uv);
				}
				else if ( ChromaUFormat == ChromaUV_88 )
				{
					ChromaUV = GetChromaUv_88(uv);
				}
				else if ( ChromaUFormat == Chroma_U && ChromaVFormat == Chroma_V  )
				{
					ChromaUV = GetChromaUv_8_8(uv);
				}
				else
				{
					//	assume direct format (greyscale, rgba etc)
					return Luma3;
				}
				
				if ( LumaFormat == RGB || LumaFormat == RGBA )
				{
					return Luma3;
				}

				//	0..1 to -0.5..0.5
				ChromaUV -= 0.5;
				
				//	override for quick debug
				if ( !ENABLE_CHROMA )
				{
					ChromaUV = float2(0, 0);
				}

				//	set luma range
				Luma = Range( LumaMin/255, LumaMax/255, Luma);
				float3 Rgb;
				Rgb.x = Luma + (ChromaVRed * ChromaUV.y);
				Rgb.y = Luma + (ChromaUGreen * ChromaUV.x) + (ChromaVGreen * ChromaUV.y);
				Rgb.z = Luma + (ChromaUBlue * ChromaUV.x);

				return Rgb;
			}
			

			fixed4 frag (v2f i) : SV_Target
			{
				float3 rgb = YuvToRgb( i.Sampleuv );
				if ( InputIsHueDepth > 0.5 )
					return float4(DepthToPosition(i.Outuv, rgb), 1);

				return float4(rgb, 1);
			}
			ENDCG
		}
	}
}
