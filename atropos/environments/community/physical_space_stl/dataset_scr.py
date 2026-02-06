import os
import shutil

# Path to your rendered_images directory
base_dir = "rendered_images"
# Path to your rendered_images directory
stl_dir = "selected_stls"

ds_stl_path = "dataset/stls/"
ds_img_path = "dataset/images/"

# List and loop through all subdirectories
for name in os.listdir(base_dir):
    path = os.path.join(base_dir, name)
    if os.path.isdir(path):
        # print(f"Found directory: {name}")
        stl_file_fpath = os.path.join(stl_dir, name)
        stl_file_fpath += ".stl"
        # print(stl_file_path)
        ds_stl_fpath = os.path.join(ds_stl_path, name)
        ds_stl_fpath += ".stl"
        shutil.copy(stl_file_fpath, ds_stl_path)
        base_img_fpath = path + "/render_0.png"
        ds_img_fpath = os.path.join(ds_img_path, name)
        ds_img_fpath += "_0001.png"
        shutil.copy(base_img_fpath, ds_img_fpath)
        # print(base_img_fpath, ds_img_fpath)
