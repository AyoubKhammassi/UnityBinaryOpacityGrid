Shader "Unlit/BinaryOpacityGrid"
{
    Properties
    {
        _Triplane ("Triplane Texture", 2DArray) = "" {}
        _SceneScaleFactor ("Scene Scale Factor", Float) = 1.0
        _DisplayMode ("Display Mode", Int) = 0
        _TriplaneVoxelSize ("Triplane Voxel Size", Float) = 0
        _TriplaneResolution ("Triplane Resolution", Float) = 0
        
        _RangeDiffuseRgbMin ("Range Diffuse RGB Min", Float) = 0
        _RangeDiffuseRgbMax ("Range Diffuse RGB Max", Float) = 0
        _RangeColorMin ("Range Color Min", Float) = 0
        _RangeColorMax ("Range Color Max", Float) = 0
        _RangeMeanMin ("Range Mean Min", Float) = 0
        _RangeMeanMax ("Range Mean Max", Float) = 0
        _RangeScaleMin ("Range Scale Min", Float) = 0
        _RangeScaleMax ("Range Scale Max", Float) = 0
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
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv0 : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                float2 uv2 : TEXCOORD2;
                float2 uv3 : TEXCOORD3;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                half4 sparseGridFeature0: TEXCOORD0;
                half4 sparseGridFeature1: TEXCOORD1;
                half4 sparseGridFeature2: TEXCOORD2;
                half4 sparseGridFeature3: TEXCOORD3;
                half4 sparseGridFeature4: TEXCOORD4;
                half4 sparseGridFeature5: TEXCOORD5;
                float3 vDirection : TEXCOORD6;
                float3 vWorldPosition: TEXCOORD7;

            };

            void unpack(in float2 uv, out half3 first, out half3 second)
            {
                uint uvx = asuint(uv.x);
                uint uvy = asuint(uv.y);
                first.x = half((uvx) & 0xFF);
                first.y = half((uvx >> 8) & 0xFF);
                first.z = half((uvx >> 16) & 0xFF);
                second.x = half((uvy) & 0xFF);
                second.y = half((uvy >> 8) & 0xFF);
                second.z = half((uvy >> 16) & 0xFF);
                first /= 255.0;
                second /= 255.0;
            }
            
            #define SIGMOID(DTYPE) DTYPE sigmoid(DTYPE x) { return 1.0 / (1.0 + exp(-x)); }
            SIGMOID(float3)
            SIGMOID(float4)
            
            #define DENORMALIZE(DTYPE)\
            DTYPE denormalize(DTYPE x, float min, float max) {\
                return min + x * (max - min);\
            }
            DENORMALIZE(float)
            DENORMALIZE(float3)
            DENORMALIZE(float4)

            #define GRID_MIN float3(-2.0, -2.0, -2.0)

            float3 contract(float3 x) {
                float3 xAbs = abs(x);
                float xMax = max(max(xAbs.x, xAbs.y), xAbs.z);
                if (xMax <= 1.0) {
                    return x;
                }
                float scale = 1.0 / xMax;
                float3 z = scale * x;
                if (xAbs.x >= xAbs.y && xAbs.x >= xAbs.z) {
                    z.x *= (2.0 - scale); // argmax = 0
                } else if (xAbs.y >= xAbs.x && xAbs.y >= xAbs.z) {
                    z.y *= (2.0 - scale); // argmax = 1
                } else {
                    z.z *= (2.0 - scale); // argmax = 2
                }
                return z;
            }

            half3 evalSphericalGaussian(half3 direction, half3 mean, float scale, half3 color) {
                color = sigmoid(color);
                mean = normalize(mean);
                scale = abs(scale);
                return color * exp(scale * (dot(direction, mean) - 1.0));
            }


            UNITY_DECLARE_TEX2DARRAY(_Triplane);
            float _SceneScaleFactor;
            int _DisplayMode;
            float _TriplaneVoxelSize;
            float _TriplaneResolution;
            float _RangeDiffuseRgbMin;
            float _RangeDiffuseRgbMax;
            float _RangeColorMin;
            float _RangeColorMax;
            float _RangeMeanMin;
            float _RangeMeanMax;
            float _RangeScaleMin;
            float _RangeScaleMax;

            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);

                //Unpack feature maps from UV coords
                unpack(v.uv0, o.sparseGridFeature0.rgb, o.sparseGridFeature1.rgb);
                unpack(v.uv1, o.sparseGridFeature2.rgb, o.sparseGridFeature3.rgb);
                unpack(v.uv2, o.sparseGridFeature4.rgb, o.sparseGridFeature5.rgb);
                half3 alpha012, alpha345;
                unpack(v.uv3, alpha012, alpha345);
                o.sparseGridFeature0.a = alpha012.r;
                o.sparseGridFeature1.a = alpha012.g;
                o.sparseGridFeature2.a = alpha012.b;
                o.sparseGridFeature3.a = alpha345.r;
                o.sparseGridFeature4.a = alpha345.g;
                o.sparseGridFeature5.a = alpha345.b;

                //World Position (Not really, it's just local position)
                o.vWorldPosition = v.vertex;
                o.vWorldPosition.x = -o.vWorldPosition.x;
                
                //View direction (in local space of the model)
                o.vDirection = -WorldSpaceViewDir(v.vertex);
                o.vDirection = normalize(mul(unity_WorldToObject, o.vDirection));
                o.vDirection.yz = o.vDirection.zy;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {

                // Query triplanes
                float3 z = contract(i.vWorldPosition * _SceneScaleFactor);
                float3 posTriplaneGrid = (z - GRID_MIN) / _TriplaneVoxelSize;
                float3 planeUv[3];
                planeUv[0] = float3(posTriplaneGrid.yz / _TriplaneResolution, 0.0);
                planeUv[1] = float3(posTriplaneGrid.xz / _TriplaneResolution, 6.0);
                planeUv[2] = float3(posTriplaneGrid.xy / _TriplaneResolution, 12.0);

                float3 diffuse = float3(0.0, 0.0, 0.0);
                half4 v_00[4];

                v_00[0] = i.sparseGridFeature0;
                v_00[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0]);
                v_00[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1]);
                v_00[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2]);

                for (int k = 0; k < 4; k++) {
                  diffuse += denormalize(v_00[k].rgb, _RangeDiffuseRgbMin,
                    _RangeDiffuseRgbMax);
                 }
                diffuse = sigmoid(diffuse);

                float3 viewDependence;
                if (1) {
                  half4 v_01[4];
                  half4 v_02[4];
                  half4 v_03[4];
                  half4 v_04[4];
                  half4 v_05[4];

                  // Read sparse grid features from unpacked UVs
                  v_01[0] = i.sparseGridFeature1;
                  v_02[0] = i.sparseGridFeature2;
                  v_03[0] = i.sparseGridFeature3;
                  v_04[0] = i.sparseGridFeature4;
                  v_05[0] = i.sparseGridFeature5;

                  // Read from triplanes.
                  v_01[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0] + half3(0.0, 0.0, 1.0));
                  v_02[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0] + half3(0.0, 0.0, 2.0));
                  v_03[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0] + half3(0.0, 0.0, 3.0));
                  v_04[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0] + half3(0.0, 0.0, 4.0));
                  v_05[1] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[0] + half3(0.0, 0.0, 5.0));

                  v_01[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1] + half3(0.0, 0.0, 1.0));
                  v_02[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1] + half3(0.0, 0.0, 2.0));
                  v_03[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1] + half3(0.0, 0.0, 3.0));
                  v_04[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1] + half3(0.0, 0.0, 4.0));
                  v_05[2] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[1] + half3(0.0, 0.0, 5.0));

                  v_01[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2] + half3(0.0, 0.0, 1.0));
                  v_02[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2] + half3(0.0, 0.0, 2.0));
                  v_03[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2] + half3(0.0, 0.0, 3.0));
                  v_04[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2] + half3(0.0, 0.0, 4.0));
                  v_05[3] = UNITY_SAMPLE_TEX2DARRAY(_Triplane, planeUv[2] + half3(0.0, 0.0, 5.0));

                  half3 c0 = half3(0.0, 0.0, 0.0);
                  half3 m0 = half3(0.0, 0.0, 0.0);
                  float s0 = 0.0;

                  half3 c1 = half3(0.0, 0.0, 0.0);
                  half3 m1 = half3(0.0, 0.0, 0.0);
                  float s1 = 0.0;

                  half3 c2 = half3(0.0, 0.0, 0.0);
                  half3 m2 = half3(0.0, 0.0, 0.0);
                  float s2 = 0.0;

                  for (int k = 0; k < 4; k++) {
                    c0 += denormalize(half3(v_00[k].a, v_01[k].rg), _RangeColorMin, _RangeColorMax);
                    m0 += denormalize(half3(v_01[k].ba, v_02[k].r), _RangeColorMin, _RangeColorMax);
                    s0 += denormalize(v_02[k].g, _RangeScaleMin, _RangeScaleMax);

                    c1 += denormalize(half3(v_02[k].ba, v_03[k].r), _RangeColorMin, _RangeColorMax);
                    m1 += denormalize(v_03[k].gba, _RangeMeanMin, _RangeMeanMax);
                    s1 += denormalize(v_04[k].r, _RangeScaleMin, _RangeScaleMax);

                    c2 += denormalize(v_04[k].gba, _RangeColorMin, _RangeColorMax);
                    m2 += denormalize(v_05[k].rgb, _RangeMeanMin, _RangeMeanMax);
                    s2 += denormalize(v_05[k].a, _RangeScaleMin, _RangeScaleMax);
                  }

                  float3 directionWorld = normalize(i.vDirection);
                  viewDependence = evalSphericalGaussian( directionWorld, m0, s0, c0);
                  viewDependence += evalSphericalGaussian(directionWorld, m1, s1, c1);
                  viewDependence += evalSphericalGaussian(directionWorld, m2, s2, c2);
                }
                fixed4 col;
                col.rgb = diffuse + viewDependence;
                col.a = 1.0;
                return col;
            }
            ENDCG
        }
    }
}
