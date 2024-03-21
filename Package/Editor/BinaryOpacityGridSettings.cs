using UnityEngine;

namespace UnityBinaryOpacityGrids
{
    [System.Serializable]
    public struct BOGSceneParameters
    {
        public int sparse_grid_resolution;
        public float sparse_grid_voxel_size;
        public int data_block_size;
        public int atlas_width;
        public int atlas_height;
        public int atlas_depth;
        public int num_slices;
        public int slice_depth;
        public float scene_scale_factor;
        public int triplane_resolution;
        public float triplane_voxel_size;
        public Ranges ranges;

        [System.Serializable]
        public struct Ranges
        {
            public Range diffuse_rgb;
            public Range color;
            public Range mean;
            public Range scale;
        }


        [System.Serializable]

        public struct Range
        {
            public float min;
            public float max;
        }
    }
    public class BinaryOpacityGridSettings : ScriptableObject
    {
        public Mesh[] Meshes;
        public Texture2DArray TriplaneTexture;
        public Material Material;
        public BOGSceneParameters SceneParameters; //Scene parameters deserialized from Json
    }

}