
import os
import cv2
import numpy as np
import argparse
import time
import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str,
                    default='')
parser.add_argument('--dest_path', type=str,
                    default ='')


opt = parser.parse_args()


data_path = opt.data_path#'/home/ganzorig/Datas/chalearn_processed_full/color/'
dest_path = opt.dest_path#'/home/ganzorig/Datas/chalearn_processed_full_skeleton/color/'

def resize_video(data_path, dest_path):
    #print(data_path, dest_path)
    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (frame_width, frame_height))
    #out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
    #    *'mp4v'), fps, (224, 224))
    while cap.isOpened():
        success, image = cap.read()
        if not success:

            #print("Ignoring empty camera frame.")
            print(data_path)
            # If loading a video, use 'break' instead of 'continue'.
            break


        img_resized = cv2.resize(image,(224,224))

        out.write(img_resized)
    cap.release()
    out.release()

# Define your video processing function
def process_video(file_path):
    root =file_path[0]
    file= file_path[1]
    dest_path_new = file_path[2] 
    if not os.path.exists(dest_path_new):      
        os.makedirs(dest_path_new,exist_ok=True)
    # You can return some result if needed
    video_path = os.path.join(root,file)

    resize_video(data_path= video_path, dest_path = os.path.join(dest_path_new,file))
    return f"Processed {video_path}"

# def main():
#    video_dir = opt.data_path # Replace with the path to your video directory
#    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(".mp4")]#

#    # Create a ThreadPoolExecutor with the number of CPU cores you want to use
#    num_cpus = os.cpu_count()  # This gets the number of available CPU cores
#    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
#        # Process videos in parallel
#        results = list(executor.map(process_video, video_files))#

#    # Print the results
#    for result in results:
#        print(result)


def main():
    file_list = []
    for (root,dirs,files) in os.walk(data_path, topdown=True):
        for file in files:
            if file.endswith('.mp4'):
                #print(os.path.join(root,file))
                dest_path_new = root.replace(data_path,dest_path)
                #if not os.path.exists(dest_path_new):      
                #    os.makedirs(dest_path_new)
                #createFolder()
                file_list.append([root,file,dest_path_new])
                #get_features(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))

    num_cpus = os.cpu_count()  # This gets the number of available CPU cores
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Process videos in parallel
        results = list(executor.map(process_video, file_list))

    # Print the results
    for result in results:
        print(result)


        print ('--------------------------------')


if __name__ == "__main__":
    main()
