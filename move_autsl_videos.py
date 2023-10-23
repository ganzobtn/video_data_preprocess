import os
import shutil


data_path = '/l/users/ganzorig.batnasan/data/autsl/chalearn_processed_full_head_hands_merged_resized'
dest_path = '/l/users/ganzorig.batnasan/data/autsl/full_head_hands_merged_resized/WLASL2000'
os.makedirs(dest_path,exist_ok=True)

for (root,dirs,files) in os.walk(data_path, topdown=True):
    for file in files:
        if file.endswith('.mp4'):
            shutil.copy(os.path.join(root,file), os.path.join(dest_path,file))
