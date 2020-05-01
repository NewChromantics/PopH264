Shader "PointCloudHalfAndHalf"
{
	Properties
	{
		ColourAndDepth("ColourAndDepth", 2D) = "white" {}
		ColourTop("ColourTop", Range(0,1) ) = 0
		ColourBottom("ColourBottom", Range(0,1) ) = 0.5
		DepthTop("DepthTop", Range(0,1) ) = 0.5
		DepthBottom("DepthBottom", Range(0,1) ) = 1
		ProjectionMatrixRow0("ProjectionMatrixRow0", VECTOR) = (1,0,0,0)
		ProjectionMatrixRow1("ProjectionMatrixRow1", VECTOR) = (0,1,0,0)
		ProjectionMatrixRow2("ProjectionMatrixRow2", VECTOR) = (0,0,1,0)
		ProjectionMatrixRow3("ProjectionMatrixRow3", VECTOR) = (0,0,0,1)
		LocalScale("LocalScale", Range(0.001,0.1) ) = 1
		WorldScale("WorldScale", Range(0.001,10.0) ) = 1.0
		DepthCap("DepthCap", Range(0.001,20)) = 1
		[Toggle]UseAlpha("UseAlpha",Range(0,1)) = 0
		[Toggle]JoinQuads("JoinQuads",Range(0,1)) = 1
		[Toggle]DebugQuadPositions("DebugQuadPositions",Range(0,1))=0
		[IntRange]TriangleCap("TriangleCap",Range(0,1000)) = 0
		[Toggle]DegenerateFarNeighbours("DegenerateFarNeighbours", Range(0,1) ) =1
		MaxNeighbourDistance("MaxNeighbourDistance", Range(0,5))=0.5
		[Toggle]DegenerateZeroDepth("DegenerateZeroDepth", Range(0,1) ) = 1
		MinimumDepth("MinimumDepth",Range(0,0.1)) = 0.0001
		[Toggle]DebugDepth("DebugDepth", Range(0,1) ) = 0
		[Toggle]DebugDepthx("DebugDepthx", Range(0,1) ) = 0
		[Toggle]DebugDepthy("DebugDepthy", Range(0,1) ) = 0
		[Toggle]DebugDepthz("DebugDepthz", Range(0,1) ) = 0
		DepthRangeMetresMin("DepthRangeMetresMin",Range(0,10) ) = 0
		DepthRangeMetresMax("DepthRangeMetresMax",Range(0,20) ) = 10
	}
	SubShader
	{
		Tags { "RenderType"="Geometry" "Queue"="Geometry"}
		LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
			//#include "PopUnityCommon/PopCommon.cginc"

			struct appdata
			{
				float4 LocalPos : POSITION;
				float2 TriangleIndex_CornerIndex : TEXCOORD0;
			};


			struct v2f
			{
				float4 ClipPos : SV_POSITION;
				float3 Bary : TEXCOORD0;
				float4 Colour : COLOR;
				float3 LocalPos : TEXCOORD1;
			};

			sampler2D ColourAndDepth;
			float4 ColourAndDepth_ST;
			float4 ColourAndDepth_TexelSize;

			float ColourTop;
			float ColourBottom;
			float DepthTop;
			float DepthBottom;			
			
			//	to debug
			float LocalScale;
			float WorldScale;
			float UseAlpha;
			float JoinQuads;
			float DebugQuadPositions;
			#define ImageWidth	ColourAndDepth_TexelSize.z
			int TriangleCap;
			float DegenerateFarNeighbours;
			float DegenerateZeroDepth;
			float MinimumDepth;
			float DepthCap;
			float MaxNeighbourDistance;
			float DebugDepth;
			float DebugDepthx;
			float DebugDepthy;
			float DebugDepthz;
			float DepthRangeMetresMin;
			float DepthRangeMetresMax;
			
			#define USE_ALPHA		(UseAlpha>0.5f)
			#define JOIN_QUADS		(JoinQuads>0.5f)
			#define DEBUG_QUADPOSITIONS	(DebugQuadPositions>0.5f)
			#define DEGENERATE_FAR_NEIGHBOURS	( DegenerateFarNeighbours > 0.5f)
			#define DEGENERATE_ZERO_DEPTH		(DegenerateZeroDepth>0.5f)
			#define DEBUG_DEPTH					(DebugDepth>0.5f)
			#define DEBUG_DEPTH_X				(DebugDepthx>0.5f)
			#define DEBUG_DEPTH_Y				(DebugDepthy>0.5f)
			#define DEBUG_DEPTH_Z				(DebugDepthz>0.5f)
			
			float4 ProjectionMatrixRow0;
			float4 ProjectionMatrixRow1;
			float4 ProjectionMatrixRow2;
			float4 ProjectionMatrixRow3;
			#define ViewToProjectionMatrix	float4x4( ProjectionMatrixRow0, ProjectionMatrixRow1, ProjectionMatrixRow2, ProjectionMatrixRow3 )
			#define ProjectionToViewMatrix	matrix_inverse( ViewToProjectionMatrix )
			
			//	gr: precalc inverse!
			float4x4 matrix_inverse(float4x4 m) 
			{
				float
				a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3],
				a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3],
				a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3],
				a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3],

				b00 = a00 * a11 - a01 * a10,
				b01 = a00 * a12 - a02 * a10,
				b02 = a00 * a13 - a03 * a10,
				b03 = a01 * a12 - a02 * a11,
				b04 = a01 * a13 - a03 * a11,
				b05 = a02 * a13 - a03 * a12,
				b06 = a20 * a31 - a21 * a30,
				b07 = a20 * a32 - a22 * a30,
				b08 = a20 * a33 - a23 * a30,
				b09 = a21 * a32 - a22 * a31,
				b10 = a21 * a33 - a23 * a31,
				b11 = a22 * a33 - a23 * a32,

				det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

			return float4x4(
				a11 * b11 - a12 * b10 + a13 * b09,
				a02 * b10 - a01 * b11 - a03 * b09,
				a31 * b05 - a32 * b04 + a33 * b03,
				a22 * b04 - a21 * b05 - a23 * b03,
				a12 * b08 - a10 * b11 - a13 * b07,
				a00 * b11 - a02 * b08 + a03 * b07,
				a32 * b02 - a30 * b05 - a33 * b01,
				a20 * b05 - a22 * b02 + a23 * b01,
				a10 * b10 - a11 * b08 + a13 * b06,
				a01 * b08 - a00 * b10 - a03 * b06,
				a30 * b04 - a31 * b02 + a33 * b00,
				a21 * b02 - a20 * b04 - a23 * b00,
				a11 * b07 - a10 * b09 - a12 * b06,
				a00 * b09 - a01 * b07 + a02 * b06,
				a31 * b01 - a30 * b03 - a32 * b00,
				a20 * b03 - a21 * b01 + a22 * b00) / det;
			}
			
			float3 NormalToRedGreen(float Normal)
			{
				if ( Normal < 0.5f )
				{
					Normal /= 0.5f;
					return float3( 1, Normal, 0 );
				}
				
				if ( Normal <= 1.0f )
				{
					Normal -= 0.5f;
					Normal /= 0.5f;
					return float3( 1-Normal, 1, 0 );
				}
				
				return float3(0,0,1);	
			}
			
			int GetTriangleIndex(int TriangleIndex,int xoffset,int yoffset)
			{
				TriangleIndex += xoffset;
				TriangleIndex += yoffset * ImageWidth;
				return TriangleIndex;
			}
			
			float2 GetColourUv(float2 uv)
			{
				return lerp( float2(0,ColourTop), float2(1,ColourBottom), uv );
			}

			float2 GetDepthUv(float2 uv)
			{
				return lerp( float2(0,DepthTop), float2(1,DepthBottom), uv );
			}
			
			float3 GetCameraProjectedPosition(float2 uv,float Depth)
			{
				float2 ImageSize = ColourAndDepth_TexelSize.zw;
				ImageSize.y /= 2.0;

				//	make projection matrix
				float4 ScreenPos = float4(uv, 1, 1);
				ScreenPos.xy = lerp( 0, ImageSize, ScreenPos.xy );
				float4 WorldPos4 = mul( ProjectionToViewMatrix, ScreenPos );
				float3 WorldPos = WorldPos4.xyz * Depth;
				return WorldPos;
			}
			
			#define DEPTH_Z_SCALAR	1.0
			
			//	position.w is validity
			void GetWorldPos(int TriangleIndex,out float4 Position,out float2 ColourUv,out float4 RawDebug)
			{
				//	turn into uv in sampler
				int Width = ColourAndDepth_TexelSize.z;
				int x = TriangleIndex % Width;
				int y = TriangleIndex / Width;
				
				float u = float(x) * ColourAndDepth_TexelSize.x;
				float v = float(y) * ColourAndDepth_TexelSize.y;
				float2 Sampleuv = float2(u,v);
				ColourUv = GetColourUv(Sampleuv);
				float2 DepthUv = GetDepthUv(Sampleuv);
				
				float4 Depth4 = tex2Dlod( ColourAndDepth, float4(DepthUv, 0,0));
				RawDebug = Depth4;
				
				
				int3 DepthInt;
				DepthInt = Depth4.xyz * float3(255,255,255);
				DepthInt.z = (Depth4.z / DEPTH_Z_SCALAR) * 255;
				int Depth = 0;
				Depth |= DepthInt.z << 16;
				Depth |= DepthInt.y << 8;
				Depth |= DepthInt.x << 0;
				
				//	depth is out of 10,000 mm (10m)
				float Int24Max = (256*256*256)-1;
				float Depthf = (Depth/Int24Max);
				Depthf = lerp( DepthRangeMetresMin, DepthRangeMetresMax, Depthf);
				Depthf -= DepthRangeMetresMin;
				/*
				Depth += lerp(0, 255, Depth4.x);		//	0x0000ff (blue-z)
				Depth += lerp( 256, 65535-255, Depth4.y );		//	0x00ff00 (green-y)
				Depth += lerp( 65536, 16777215-65535, Depth4.z );	//	0xff0000 (red-x)
				Depth /= 256 * 256 * 256;
				Depth *= 10; // in meters
				*/

				Position.xyz = GetCameraProjectedPosition( Sampleuv, Depthf );
				if ( DEGENERATE_ZERO_DEPTH )
					Position.w = (Depthf <= MinimumDepth ) ? 0 : 1;
				
				if (Position.z > DepthCap) 
					Position.w = 0;
				
				if ( DEBUG_QUADPOSITIONS )
				{
					Position = float4( x, y, 0, 1 );
				}
				
				#define TEST_POSITION	false
				if ( TEST_POSITION )
					Position = float4(0,0,0,1);
			}
			
			//	w of position is validity
			void GetWorldPosAndColour(int TriangleIndex,int VertexIndex,out float4 Position,out float4 Colour)
			{
				if ( TriangleCap != 0 && TriangleIndex >= TriangleCap )
				{
					Position = float4(0,0,0,0);
					Colour = float4(1,0,0,1);
					return;
				}
			
				//	if we're gonna sample neighbours to join up the quads
				//	then we need to change which triangle we're sampling, using the real stride (which we don't pad)
				
				if ( JOIN_QUADS )
				{
					//	sample neighbours for different corners
					if ( VertexIndex == 1 )	//	top right
						TriangleIndex = GetTriangleIndex( TriangleIndex, 1, 0 );
					if ( VertexIndex == 2 )	//	bottom right
						TriangleIndex = GetTriangleIndex( TriangleIndex, 1, 1 );
					if ( VertexIndex == 3 )	//	bottom left
						TriangleIndex = GetTriangleIndex( TriangleIndex, 0, 1 );
				}
				
				float2 Sampleuv;
				float4 RawDepth;
				GetWorldPos( TriangleIndex, Position, Sampleuv, RawDepth );
				Colour = tex2Dlod( ColourAndDepth, float4(Sampleuv,0,0) );
				Colour.w = 1;
				
				if ( DEBUG_DEPTH )
				{
					Colour.xyz = RawDepth.xyz;
				}
				else if ( DEBUG_DEPTH_X )
				{
					Colour.xyz = NormalToRedGreen( RawDepth.x );
				}
				else if ( DEBUG_DEPTH_Y )
				{
					Colour.xyz = NormalToRedGreen( RawDepth.y );
				}
				else if ( DEBUG_DEPTH_Z )
				{
					Colour.xyz = NormalToRedGreen( RawDepth.z / DEPTH_Z_SCALAR );
				}

				
				#define TEST_COLOUR		false
				if ( TEST_COLOUR )
					Colour = float4(0,1,0,1);
			}
			
			bool IsBigNeighbour(int TriangleIndex)
			{
				//	get size of neighbours to see if we're stretching
				float2 Temp2;
				float4 Temp4;
				float4 TopLeftPos;
				float4 TopRightPos;
				float4 BottomRightPos;
				float4 BottomLeftPos;
				GetWorldPos( GetTriangleIndex(TriangleIndex,0,0), TopLeftPos, Temp2, Temp4 );
				GetWorldPos( GetTriangleIndex(TriangleIndex,1,0), TopRightPos, Temp2, Temp4 );
				GetWorldPos( GetTriangleIndex(TriangleIndex,1,1), BottomRightPos, Temp2, Temp4 );
				GetWorldPos( GetTriangleIndex(TriangleIndex,0,1), BottomLeftPos, Temp2, Temp4 );
				float TopRightDistance = distance(TopLeftPos.xyz,TopRightPos.xyz);
				float BottomRightDistance = distance(TopLeftPos.xyz,BottomRightPos.xyz);
				float BottomLeftDistance = distance(TopLeftPos.xyz,BottomLeftPos.xyz);
				float BigDistance = max( TopRightDistance, max( BottomRightDistance,BottomLeftDistance));
				float Valid = TopLeftPos.w * TopRightPos.w * BottomLeftPos.w * BottomRightPos.w;
				if ( !Valid)
					return false;
				return BigDistance > MaxNeighbourDistance;
			}

		
			v2f vert (appdata v)
			{
				v2f o;
				
				//	z is not zero to solve overdraw issue
				float3 LocalPos = v.LocalPos;
				LocalPos.z = 0;
				LocalPos.x = lerp( -0.5, 0.5, LocalPos.x );
				LocalPos.y = lerp( -0.5, 0.5, LocalPos.y );
				
				int TriangleIndex = v.TriangleIndex_CornerIndex.x;
				int VertexIndex = v.TriangleIndex_CornerIndex.y;
				float4 WorldPos4;
				float4 Colour;
				GetWorldPosAndColour( TriangleIndex, VertexIndex, WorldPos4, Colour );
				float3 WorldPos = WorldPos4.xyz;
				WorldPos *= WorldScale;
		
				if ( !JOIN_QUADS )
					WorldPos += LocalPos * LocalScale;
				float4 ProjectionPos = UnityObjectToClipPos(WorldPos);
				/*
				
				//	move to camera space
				//float4 WorldPos = LocalToWorldTransform * float4(TriangleWorldPos,1);
				float3 CameraPos = UnityWorldToViewPos( float4(WorldPos,1) );
				
				//	note: no good for vr! need to rotate toward camera in world space
				LocalPos *= ParticleSize;
				CameraPos.xy += LocalPos.xy;

				float4 ProjectionPos = UnityViewToClipPos(CameraPos);
				*/
				o.ClipPos = ProjectionPos;
				
				//	invalidate bad positions
				o.ClipPos *= WorldPos4.w;
				
				if ( DEGENERATE_FAR_NEIGHBOURS && JOIN_QUADS )
				{
					if ( IsBigNeighbour(TriangleIndex) )
						o.ClipPos = float4(0,0,0,0);
				}
				
				/*
				//	degenerate
				bool ValidTriangle = true;
				if ( !ValidTriangle )
					o.ClipPos = float4(0,0,0,0);
				*/	
				o.LocalPos = v.LocalPos;
				o.Colour = Colour;
				o.Colour.w = USE_ALPHA ? o.Colour.w : 1;
				return o;
			}
			
			fixed4 frag (v2f Input) : SV_Target
			{
				//return float4(1,0,0,1);
				return Input.Colour;
			}
			ENDCG
		}
	}
}
