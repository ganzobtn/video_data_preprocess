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

def pad_video_into_square(video_path):
    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Get the original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the size of the square frame
    size = max(width, height)

    # Create a black square frame
    square_frame = np.zeros((size, size, 3), dtype=np.uint8)

    # Calculate the padding values
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left

    # Process each frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pad the frame with black values
        padded_frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # # Display the padded frame
        # cv2.imshow('Padded Frame', padded_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()



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
    # Example usage
    #video_path = '/path/to/your/video.mp4'
    #pad_video_into_square(video_path)

    num_cpus = os.cpu_count()  # This gets the number of available CPU cores
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Process videos in parallel
        results = list(executor.map(pad_video_into_square, file_list))

    # Print the results
    for result in results:
        print(result)


        print ('--------------------------------')


if __name__ == "__main__":
    main()
