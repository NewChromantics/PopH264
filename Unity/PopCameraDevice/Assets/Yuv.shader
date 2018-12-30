Shader "Unlit/Yuv"
{
	Properties
	{
		LumaTexture ("LumaTexture", 2D) = "white" {}
		[Enum(None,0,Greyscale,1,YYuv_8888_Ntsc,19)]LumaFormat("LumaFormat",int) = 0
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
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			sampler2D LumaTexture;
			sampler2D ChromaUTexture;
			sampler2D ChromaVTexture;
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
		#define ChromaUV_88	25
		#define Chroma_U	26
		#define Chroma_V	27
		#define ChromaVU_88	998

			float Flip;
			float EnableChroma;
			#define FLIP	( Flip > 0.5f )	
			#define ENABLE_CHROMA	( EnableChroma > 0.5f )
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;

				if ( FLIP )
					o.uv.y = 1 - o.uv.y;

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


			fixed4 frag (v2f i) : SV_Target
			{
				// sample the texture
				float Luma = tex2D(LumaTexture, i.uv);
				float2 ChromaUV = float2(0, 0);
				if ( ChromaUFormat == Debug )
				{
					ChromaUV = GetChromaUv_Debug(i.uv);
				}
				else if ( ChromaUFormat == ChromaUV_88 )
				{
					ChromaUV = GetChromaUv_88(i.uv);
				}
				else if ( ChromaUFormat == Chroma_U && ChromaVFormat == Chroma_V  )
				{
					ChromaUV = GetChromaUv_8_8(i.uv);
				}

				//	0..1 to -0.5..0.5
				ChromaUV -= 0.5;
				
				//	override for quick debug
				if ( !ENABLE_CHROMA )
				{
					ChromaUV = float2(0, 0);
				}

				//	set luma range
				Luma = lerp(LumaMin/255, LumaMax/255, Luma);
				float3 Rgb;
				Rgb.x = Luma + (ChromaVRed * ChromaUV.y);
				Rgb.y = Luma + (ChromaUGreen * ChromaUV.x) + (ChromaVGreen * ChromaUV.y);
				Rgb.z = Luma + (ChromaUBlue * ChromaUV.x);

				return float4( Rgb.xyz, 1);
			}
			ENDCG
		}
	}
}
