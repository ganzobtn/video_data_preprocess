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

def pad_video_into_square(data_path, dest_path):
    # Read the video
    #print(data_path)
    cap = cv2.VideoCapture(data_path)
    # Get the original video dimensions
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = int(cap.get(3)), int(cap.get(4))
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #out = cv2.VideoWriter(os.path.join(dest_path, person, category, video), cv2.VideoWriter_fourcc(
    #    *'MP4V'), fps, (frame_width, frame_height))


    # Determine the size of the square frame
    size = max(width, height)

    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (size,size))
    



    # Create a black square frame
    #square_frame = np.zeros((size, size, 3), dtype=np.uint8)

    # Calculate the padding values
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left

    # Process each frame in the video
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            #print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Pad the frame with black values
        padded_frame = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #print(padded_frame.shape)
        out.write(padded_frame)
        # # Display the padded frame
        # cv2.imshow('Padded Frame', padded_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

def process_video(file_path):
    root =file_path[0]
    file= file_path[1]
    dest_path_new = file_path[2] 
    if not os.path.exists(dest_path_new):      
        os.makedirs(dest_path_new,exist_ok=True)
    # You can return some result if needed
    video_path = os.path.join(root,file)
    #print(video_path,dest_path_new, file)
    pad_video_into_square(data_path= video_path, dest_path = os.path.join(dest_path_new,file))
    print(f"Processed {video_path}")
    return f"Processed {video_path}"

def main():
    file_list = []
    for (root,dirs,files) in os.walk(data_path, topdown=True):
        for file in files:
            if file.endswith('.mp4'):
                #print(os.path.join(root,file))
                dest_path_new = root.replace(data_path,dest_path)
                file_list.append([root,file,dest_path_new])
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
