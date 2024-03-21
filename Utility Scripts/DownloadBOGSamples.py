import os
import wget
import json
import argparse

scene_list = ["bicycle", "flowerbed", "gardenvase", "stump", "treehill", "kitchenlego", "fulllivingroom", "kitchencounter", "kitchenlego", "officebonsai"] 
baseurl = "https://storage.googleapis.com/realtime-nerf-360/binary_opacity_grid/"

def get_num_slices(json_path):
    file = open(json_path)
    data = json.load(file)
    num_slices = data["num_slices"]
    file.close()
    return num_slices

def download_scenes(scene_names, base_dir):
    for scene in scene_names:
        print("\n\n")
        print("*********************************************")
        print("Downloading Binary Opacity Grid sample scene: " + scene)
        scene_dir = os.path.abspath("{0}/{1}/".format(base_dir, scene))
        if(not os.path.exists(scene_dir)):
            os.makedirs(scene_dir)
        else:
            print("A folder of the scene {0} already exists! Skipping...".format(scene))
            continue

        scene_url = baseurl + scene + '/'

        print("Downloading files...")
        print("From URL: " + scene_url)
        print("To directory: "+ scene_dir)

        response = wget.download(scene_url + "scene_params.json", scene_dir)
        num_slices = get_num_slices(os.path.join(scene_dir, "scene_params.json"))
        #download sparse grid feature raw files
        for i in range(6):
            for j in range(num_slices):
                url = "{0}sparse_grid_features_{1:02d}_{2:03d}.raw".format(scene_url, i, j)
                response = wget.download(url, scene_dir)
        
        for i in range(3):
            #download the plane features raw files
            for j in range(6):
                url = "{0}plane_features_{1}_{2:02d}.raw".format(scene_url, i, j)
                response = wget.download(url, scene_dir)
        

        #download sparse grid indices 
        url = scene_url + "sparse_grid_block_indices.raw"
        response = wget.download(url, scene_dir)
        print(response)

        #download the glb file 
        url = scene_url + "viewer_mesh_post_gltfpack.glb"
        response = wget.download(url, scene_dir)
        print(response)



parser = argparse.ArgumentParser(
                    prog = 'DownloadBOGSamples',
                    description = 'A helper script to download Binary Opacity Grid sample scenes',
                    )

parser.add_argument('downloadPath', help="The path where the sample scenes will be downloaded.") 
parser.add_argument('-n', '--name', required=False, help="The name of a specific sample scene.", choices= scene_list)
parser.add_argument('-a', '--all', required=False, help="Download all the sample scenes. Ignores --name if this is used",
                    action='store_true')  # on/off flag

args = parser.parse_args()

if args.all or args.name is None:
    print("Downloading all Binary Opacity Grid sample scenes.")
    download_scenes(scene_list, os.path.join(args.downloadPath, "BinaryOpacityGridSamples"))
else:
    print("Downloading {0} Opacity Grid sample scene.".format(args.name))
    download_scenes(([args.name]), args.downloadPath)




        





