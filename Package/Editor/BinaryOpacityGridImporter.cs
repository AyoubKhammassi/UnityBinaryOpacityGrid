using UnityEditor;
using UnityEngine;
using System.IO;
using System;
using Unity.Collections;
using GLTFast;
using File = System.IO.File;
using Directory = System.IO.Directory;
using Material = UnityEngine.Material;
using System.Threading.Tasks;

namespace UnityBinaryOpacityGrids
{
    public class BinaryOpacityGridImporter : EditorWindow
    {
        const UInt16 GNumChannelChunks = 6;
        const string GOriginalGLBName = "viewer_mesh_post_gltfpack.glb";
        static async Task<(bool, string)> TryLoadAssets(string path)
        {
            string message = "";
            string sceneName = Path.GetFileName(path);
            Debug.Log("Loading Assets for " + sceneName);
            BOGSceneParameters sceneParams = new BOGSceneParameters();
            string[] foundFiles = Directory.GetFiles(path, "*.json");
            if (foundFiles.Length == 0)
            {
                message = "No Json file found in the folder: " + path;
                return (false, message);
            } 
                
            //use the first json file we found
            sceneParams = JsonUtility.FromJson<BOGSceneParameters>(File.ReadAllText(foundFiles[0]));


           
            //load glb file
            foundFiles = Directory.GetFiles(path, "*.glb");
            if (foundFiles.Length == 0)
            {
                message = "No glb file found in the folder: " + path;
                return (false, message);
            }

            bool foundBakedGlb = false;
            var gltf = new GltfImport(deferAgent: new UninterruptedDeferAgent());

            foreach (var glbpath in foundFiles)
            {
                //Skip the original glb file as we're only interested in the glbs with the baked feature maps
                if (Path.GetFileName(glbpath) == GOriginalGLBName)
                    continue;


                bool success = await gltf.LoadFile(glbpath);
                if (success) 
                {
                    var meshes = gltf.GetMeshes();
                    if(meshes.Length > 0 && meshes[0].uv.Length > 0 && meshes[0].uv2.Length > 0 && meshes[0].uv3.Length > 0 && meshes[0].uv4.Length > 0)
                    {
                        foundBakedGlb = true;
                        break;
                    }
                }
            }

            //Create a texture array with the correct settings from scene params
            Texture2DArray triplaneArray = new Texture2DArray(sceneParams.triplane_resolution, sceneParams.triplane_resolution,
                3 * GNumChannelChunks, TextureFormat.RGBA32, false, true);

            triplaneArray.name = sceneName + "Triplane";
            //Load data from .raw files
            for (UInt16 i = 0; i < 3u; ++i)
            {
                for (UInt16 j = 0; j < GNumChannelChunks; ++j)
                {
                    string filePath = String.Format("{2}/plane_features_{0}_{1:00}.raw", i, j, path);
                    if (!File.Exists(filePath))
                    {
                        message = String.Format("Can't find file plane_features_{0}_{1:00}.raw in path {2}! " +
                            "Make sure that all plane feature .raw files are in the path you selected.",
                            i, j, path);
                        return (false, message);
                    }
                    var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                    NativeArray<byte> byteData = new NativeArray<byte>(4 * sceneParams.triplane_resolution * sceneParams.triplane_resolution, Allocator.Temp);
                    fs.Read(byteData);
                    triplaneArray.SetPixelData(byteData, 0, i * GNumChannelChunks + j);
                }
            }

            triplaneArray.filterMode = FilterMode.Bilinear;
            triplaneArray.wrapMode = TextureWrapMode.Clamp;
            triplaneArray.Apply(false);


            if (foundBakedGlb)
            {
                //Create new folder for the assets
                AssetDatabase.CreateFolder("Assets", sceneName);

                //root asset
                var assetRoot = ObjectFactory.CreateInstance<BinaryOpacityGridSettings>();
                assetRoot.name = sceneName;
                AssetDatabase.CreateAsset(assetRoot, String.Format("Assets/{0}/{0}Assets.asset", sceneName));
                var assetRootPath = AssetDatabase.GetAssetPath(assetRoot);


                //Add all assets to the root asset
                //Texture Array
                AssetDatabase.AddObjectToAsset(triplaneArray, assetRootPath);

                //Material
                Material material = new Material(Shader.Find("Unlit/BinaryOpacityGrid"));
                material.name = sceneName + "Mat";
                //Set properties
                material.SetTexture("_Triplane", triplaneArray);
                material.SetFloat("_SceneScaleFactor", sceneParams.scene_scale_factor); 
                material.SetInt("_DisplayMode", 0); 
                material.SetFloat("_TriplaneVoxelSize", sceneParams.triplane_voxel_size);
                material.SetFloat("_TriplaneResolution", sceneParams.triplane_resolution);
                material.SetFloat("_RangeDiffuseRgbMin", sceneParams.ranges.diffuse_rgb.min);
                material.SetFloat("_RangeDiffuseRgbMax", sceneParams.ranges.diffuse_rgb.max);
                material.SetFloat("_RangeColorMin", sceneParams.ranges.color.min);
                material.SetFloat("_RangeColorMax", sceneParams.ranges.color.max);
                material.SetFloat("_RangeMeanMin", sceneParams.ranges.mean.min);
                material.SetFloat("_RangeMeanMax", sceneParams.ranges.mean.max);
                material.SetFloat("_RangeScaleMin", sceneParams.ranges.scale.min);
                material.SetFloat("_RangeScaleMax", sceneParams.ranges.scale.max);

                AssetDatabase.AddObjectToAsset(material, assetRootPath);

                assetRoot.SceneParameters = sceneParams;
                assetRoot.Meshes = gltf.GetMeshes();
                assetRoot.Material = material;
                assetRoot.TriplaneTexture = triplaneArray;

                GameObject RootGO = new GameObject(sceneName);

                foreach (var mesh in assetRoot.Meshes)
                {
                    //First save the mesh asset
                    AssetDatabase.AddObjectToAsset(mesh, assetRootPath);

                    //Create a child game object for this mesh in the prefab
                    GameObject childObject = new GameObject(mesh.name);
                    var meshFilter = childObject.AddComponent<MeshFilter>();
                    var meshRenderer = childObject.AddComponent<MeshRenderer>();
                    meshFilter.mesh = mesh;
                    meshRenderer.sharedMaterial = material;
                    meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    meshRenderer.receiveShadows = false;
                    meshRenderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
                    meshRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
                    childObject.transform.parent = RootGO.transform;
                }
                AssetDatabase.SaveAssets();
                PrefabUtility.SaveAsPrefabAsset(RootGO, String.Format("Assets/{0}/{0}.prefab", sceneName));
                //success = await gltf.InstantiateMainSceneAsync(transform);
            }
            else
            {
                message = "No valid glb file with baked feature maps found in the folder: " + path;
                return (false, message);
            }

            //Sucess
            return (true, message);
        }


        [MenuItem("Binary Opacity Grid/Load Assets from Folder")]
        static async void LoadAssets()
        {
            string path = EditorUtility.OpenFolderPanel("Load Assets", "", "");
            bool result;
            string message;
            (result, message) = await TryLoadAssets(path);
            if (result)
            {
                Debug.Log("Successfully loaded Binary Opacity Grid Assets from " + path);
            }
            else
            {
                Debug.LogError(String.Format("Loading Binary Opacity Grid Assets from {0} failed!", path));
                Debug.LogError(message);
            }
        }
    }

}
