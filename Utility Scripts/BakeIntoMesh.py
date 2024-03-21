import argparse
import os
from pathlib import Path
import trimesh
import numpy as np
import json
from numba import jit, prange, jit_module
import struct
from ModifiedTrimeshFunctions import _append_multi_uv_mesh
trimesh.exchange.gltf._append_mesh = _append_multi_uv_mesh

jit_module(error_model="struct")

def LoadSparseGrid(path, width, height, depth, featureIndex):
    grid = np.zeros((depth, height, width, 4))
    for j in range(depth):
        file_path = "{0}sparse_grid_features_{1:02d}_{2:03d}.raw".format(path, featureIndex, j)
        print("Loading RAW file: " + file_path)
        npimg = np.fromfile(file_path, dtype=np.uint8)
        npimg = npimg.reshape((height, width, 4))
        grid[j] = npimg
    return grid


def LoadSparseGridBlockIndices(path, size):
    file_path = path + "sparse_grid_block_indices.raw"
    print("Loading RAW file: " + file_path)
    npimg = np.fromfile(file_path, dtype=np.uint8)
    npimg = npimg.reshape((size, size, size, 3))
    return npimg

def LoadSceneParams(path):
    file_path = path + "scene_params.json"
    print("Loading JSON file: " + file_path)
    with open(file_path) as fd:
        d = json.load(fd)
        return d

@jit
def TriInterpolate(texture, uvw):
    # Get the integer and fractional parts of the UVW coordinates
    uvw_int = np.floor(uvw)
    uvw_frac = uvw - uvw_int
    u_int = min(int(uvw_int[2]), texture.shape[0] - 2)
    v_int = min(int(uvw_int[1]), texture.shape[1] - 2)
    w_int = min(int(uvw_int[0]), texture.shape[2] - 2)
    u_frac = uvw_frac[2]
    v_frac = uvw_frac[1]
    w_frac = uvw_frac[0]
    
    c000 = texture[u_int, v_int, w_int]
    c100 = texture[u_int + 1, v_int, w_int]
    c010 = texture[u_int, v_int + 1, w_int]
    c110 = texture[u_int + 1, v_int + 1, w_int]
    c001 = texture[u_int, v_int, w_int + 1]
    c101 = texture[u_int + 1, v_int, w_int + 1]
    c011 = texture[u_int, v_int + 1, w_int + 1]
    c111 = texture[u_int + 1, v_int + 1, w_int + 1]
    
    # Perform trilinear interpolation
    c00 = c000 * (1.0 - u_frac) + c100 * u_frac
    c01 = c001 * (1.0 - u_frac) + c101 * u_frac
    c10 = c010 * (1.0 - u_frac) + c110 * u_frac
    c11 = c011 * (1.0 - u_frac) + c111 * u_frac
    
    c0 = c00 * (1.0 - v_frac) + c10 * v_frac
    c1 = c01 * (1.0 - v_frac) + c11 * v_frac
    
    c = c0 * (1.0 - w_frac) + c1 * w_frac
    return c

@jit
def Nearest(texture, uvw):
    return texture[int(uvw[2]), int(uvw[1]), int(uvw[0])]

@jit
def contract(x):
    xAbs = np.abs(x)
    xMax = np.max(xAbs)
    if xMax <= 1.0:
        return x
    scale = 1.0 / xMax
    z = scale * x
    argmax = np.argmax(xAbs)
    z[argmax] *= (2.0 - scale)
    return z

def PackFloat32(x,y,z,w):
    packed = struct.pack('4B', int(x),int(y),int(z),int(w))  # pack 4 int8 into a binary string
    float_val = struct.unpack('f', packed)[0]
    return float_val


#Packs one int8 in the two exponents of the UV floats
@jit
def PackInExponents(x):
    if hasattr(x, "__len__"):
        x=x[0]
    e1 = int((x & 0xF) | 96) 
    e2 = int(((x >> 4) & 0xF) | 96)
    return e1,e2

# Bake feature maps in UV coords per vertex
@jit(parallel=True)
def bake_features(vertices, v_00, v_01, v_02, v_03, v_04, v_05):
    exp = PackInExponents(1)
    for i in prange(vertices.shape[0]):
        pos = vertices[i]
        z = contract(pos * scene_scale_factor)
        posSparseGrid = (z - GRID_MIN) / sparseGridVoxelSize - 0.5
        atlasBlockMin = np.floor(posSparseGrid / dataBlockSize) 
        atlasBlockIndex = Nearest(SparseGridBlockIndices, atlasBlockMin)
        posAtlas = np.clip(posSparseGrid - (atlasBlockMin * dataBlockSize), 0.0, dataBlockSize)
        posAtlas = posAtlas + atlasBlockIndex * (dataBlockSize + 1.0)

        v_00[i] = TriInterpolate(sparseGridFeature0, posAtlas)
        v_01[i] = TriInterpolate(sparseGridFeature1, posAtlas)
        v_02[i] = TriInterpolate(sparseGridFeature2, posAtlas)
        v_03[i] = TriInterpolate(sparseGridFeature3, posAtlas)
        v_04[i] = TriInterpolate(sparseGridFeature4, posAtlas)
        v_05[i] = TriInterpolate(sparseGridFeature5, posAtlas)
        #Can't Numba JIT struct pack and unpack so I'm doing it outside        
        #uv0[i][0] = PackFloat32(v_00[i][0], v_00[i][1], v_00[i][2], exp[0])
        #uv0[i][1] = PackFloat32(v_01[i][0], v_01[i][1], v_01[i][2], exp[0])
        #uv1[i][0] = PackFloat32(v_02[i][0], v_02[i][1], v_02[i][2], exp[0])
        #uv1[i][1] = PackFloat32(v_03[i][0], v_03[i][1], v_03[i][2], exp[0])
        #uv2[i][0] = PackFloat32(v_04[i][0], v_04[i][1], v_04[i][2], exp[0])
        #uv2[i][1] = PackFloat32(v_05[i][0], v_05[i][1], v_05[i][2], exp[0])
        #uv3[i][0] = PackFloat32(v_00[i][3], v_01[i][3], v_02[i][3], exp[0])
        #uv3[i][1] = PackFloat32(v_03[i][3], v_04[i][3], v_05[i][3], exp[0])

parser = argparse.ArgumentParser(
                    prog = 'BakeIntoMesh',
                    description = 'A utility script that bakes the Binary Opacity Grid feature maps in the UV coordinates of the mesh.',
                    )

parser.add_argument('path', help="The path for a Binary Opacity Grid GLB file to be baked. The parent folder must also contain all the .raw files required!") 

args = parser.parse_args()
glb_path = args.path
if(not os.path.exists(glb_path)):
    print(glb_path + " doesn't exist!")
    exit()

if os.path.isfile(glb_path) and Path(glb_path).suffix == ".glb":
    if Path(glb_path).suffix == ".glb":
        print("Loading GLB file: " + glb_path)
        scene = trimesh.load(glb_path)
        folder_path = os.path.dirname(glb_path) + '/'
        

        sp = LoadSceneParams(folder_path)

        print("Loading feature maps...")

        sparseGridFeature0 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 0)
        sparseGridFeature1 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 1)
        sparseGridFeature2 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 2)
        sparseGridFeature3 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 3)
        sparseGridFeature4 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 4)
        sparseGridFeature5 = LoadSparseGrid(folder_path, sp["atlas_width"], sp["atlas_height"], sp["atlas_depth"], 5)

        SparseGridBlockIndices = LoadSparseGridBlockIndices(folder_path, sp["sparse_grid_resolution"] // sp["data_block_size"])

        sparseGridGridSize =  np.array([sp["sparse_grid_resolution"], sp["sparse_grid_resolution"], sp["sparse_grid_resolution"]])
        dataBlockSize =  sp["data_block_size"]

        iBlockGridBlocks = (sparseGridGridSize + dataBlockSize - 1) // dataBlockSize
        iBlockGridSize = iBlockGridBlocks * dataBlockSize
        blockGridSize = np.array(iBlockGridSize, dtype=float)

        scene_scale_factor = sp["scene_scale_factor"]
        sparseGridVoxelSize = sp["sparse_grid_voxel_size"]
        atlasSize = np.array([ sp['atlas_width'], sp['atlas_height'],sp['atlas_depth']])
        triplaneVoxelSize = sp['triplane_voxel_size']
        triplaneSize = sp['triplane_resolution']
        rangeDiffuseRgbMin = sp['ranges']['diffuse_rgb']['min']
        rangeDiffuseRgbMax = sp['ranges']['diffuse_rgb']['max']
        rangeColorMin = sp['ranges']['color']['min']
        rangeColorMax = sp['ranges']['color']['max']
        rangeMeanMin = sp['ranges']['mean']['min']
        rangeMeanMax = sp['ranges']['mean']['max']
        rangeScaleMin = sp['ranges']['scale']['min']
        rangeScaleMax = sp['ranges']['scale']['max']
        GRID_MIN = np.array([-2.0, -2.0, -2.0])
        exp = PackInExponents(1)

        for meshName, meshIndex in zip(scene.geometry, range(len(scene.geometry))):
            print("Baking chunk {0}/{1} : {2}".format(meshIndex, len(scene.geometry), meshName))
            mesh = scene.geometry[meshName]
            v_00 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            v_01 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            v_02 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            v_03 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            v_04 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            v_05 = np.zeros((mesh.vertices.shape[0], 4), dtype=np.int_)
            #UV numpy arrays
            uv0 = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
            uv1 = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
            uv2 = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
            uv3 = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
            bake_features(mesh.vertices, v_00, v_01, v_02, v_03, v_04, v_05)
            for i in range(mesh.vertices.shape[0]):
                uv0[i][0] = PackFloat32(v_00[i][0], v_00[i][1], v_00[i][2], exp[0])
                uv0[i][1] = PackFloat32(v_01[i][0], v_01[i][1], v_01[i][2], exp[0])
                uv1[i][0] = PackFloat32(v_02[i][0], v_02[i][1], v_02[i][2], exp[0])
                uv1[i][1] = PackFloat32(v_03[i][0], v_03[i][1], v_03[i][2], exp[0])
                uv2[i][0] = PackFloat32(v_04[i][0], v_04[i][1], v_04[i][2], exp[0])
                uv2[i][1] = PackFloat32(v_05[i][0], v_05[i][1], v_05[i][2], exp[0])
                uv3[i][0] = PackFloat32(v_00[i][3], v_01[i][3], v_02[i][3], exp[0])
                uv3[i][1] = PackFloat32(v_03[i][3], v_04[i][3], v_05[i][3], exp[0])

            texvisual = trimesh.visual.TextureVisuals(uv=uv0)
            texvisual1 = trimesh.visual.TextureVisuals(uv=uv1)
            texvisual2 = trimesh.visual.TextureVisuals(uv=uv2)
            texvisual3 = trimesh.visual.TextureVisuals(uv=uv3)
            texvisual.vertex_attributes = {'color': mesh.visual.vertex_colors, 'uv' : texvisual.uv, 'uv1' : texvisual1.uv, 'uv2' : texvisual2.uv, 'uv3' : texvisual3.uv}
            mesh.visual = texvisual

        exportPath = folder_path + "FullMeshWithFeatures.glb"
        print("Exporting " + exportPath)
        scene.export(exportPath)
else:
    print(glb_path + " is not a GLB file!")
        




