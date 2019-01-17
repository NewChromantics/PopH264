Shader "Unlit/Yuv"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		[Toggle]Flip("Flip", Range(0,1)) = 1
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

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float Flip;
			#define FLIP	( Flip > 0.5f )	
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);

				if ( FLIP )
					o.uv.y = 1 - o.uv.y;

				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				// sample the texture
				float3 Rgb = tex2D(_MainTex, i.uv);

				Rgb.xyz = Rgb.xxx;

				return float4(Rgb, 1);
			}
			ENDCG
		}
	}
}
