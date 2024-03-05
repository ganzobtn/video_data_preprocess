import os
import pandas as pd
import argparse
import csv
import pickle as pkl
parser = argparse.ArgumentParser()

parser.add_argument('--my_path', type=str, default='/projects/data/kinetics_dataset/k400/annotations')
parser.add_argument('--csv_path',type=str,default='/projects/videomaev2/datas/dgx/finetune/k400/')
parser.add_argument('--data_path',type=str,default='/projects/data/kinetics_dataset/k400/')
parser.add_argument('--print_paths',action = 'store_true')
opt = parser.parse_args()


wlasl_videos_path ='/projects/data/wlasl_2000/wlasl_2000_sorted/'
kinetics_path = '../datas/dgx/finetune/revised/wlasl_2000/'
os.makedirs(kinetics_path,exist_ok=True)


my_path = opt.my_path
csv_path = opt.csv_path
os.makedirs(csv_path,exist_ok=True)
data_path = opt.data_path
print_paths = opt.print_paths


classes  = os.listdir(os.path.join(wlasl_videos_path,'train'))
classes.sort()


# with open('../misc/label_map_wlasl2000.txt', 'w') as f:
#     for line in classes[:-1]:
#         f.write(f"{line}\n")
#     f.write(f"{classes[-1]}")


for i in ['dev','test','train']:

    with open(os.path.join(csv_path,i+'.csv'),'w',newline ='') as file_write:
        writer = csv.writer(file_write)

        for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
            for video in os.listdir(os.path.join(wlasl_videos_path,i,clas)):
                

                path = os.path.join(wlasl_videos_path,i, clas,video)
                label = classes.index(clas)

                #print(path,label)
                if os.path.exists(path):
                    writer.writerow([path+' '+str(label)])