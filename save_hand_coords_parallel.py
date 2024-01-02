
import cv2
import os
import multiprocessing
import pickle
import mediapipe as mp
import numpy as np
mp_holistic = mp.solutions.holistic

#with open('/home/ganzorig/docker_workspace/data/wlasl_2000/keypoints_hrnet_dark_coco_wholebody.pkl', 'rb') as f:
#    data = pickle.load(f)

data_path = '/media/ganzorig/53D11DCE629A37AA/Datas/WLASL2000/'

def get_keypoints(data_path):
    keypoint_data= []
    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame:',length)
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #out = cv2.VideoWriter(os.path.join(dest_path, person, category, video), cv2.VideoWriter_fourcc(
    #    *'MP4V'), fps, (frame_width, frame_height))

    with mp_holistic.Holistic(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)
            #print('ff:', results.right_hand_landmarks)
            data = []


            landmarks = results.right_hand_landmarks
            if landmarks is not None:
                keypoints  = landmarks.landmark
                for i in keypoints:
                    data.append([i.x,i.y,0.5])
                #print(len(data))
            else:
                for i in range(21):
                    data.append([0,0,0.5])

            landmarks = results.left_hand_landmarks
            if landmarks is not None:
                keypoints  = landmarks.landmark
                for i in keypoints:
                    data.append([i.x,i.y,0.5])
                #print(len(data))
            else:
                for i in range(21):
                    data.append([0,0,0.5])




            keypoint_data.append(data)

        cap.release()
    return np.array(keypoint_data[1:-1])


def process_video_wrapper(video_file):
    # Define the output path for each processed video
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_file).replace('mp4','npy'))

    #process_video(video_file, output_path)
    hand_keypoint_data= get_keypoints(video_file)
    np.save(output_path,hand_keypoint_data)
if __name__ == '__main__':
    # List of video files to process
    video_files = [
    ]
    for (root,dirs,files) in os.walk(data_path, topdown=True):
    #print (root)
    #print (dirs)
    #print (files)
        for file in files:
            if file.endswith('.mp4'):
                #print(os.path.join(root,file))

                #get_features(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))
                video_files.append(os.path.join(root,file))

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()  # Use the available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Process the videos in parallel
    pool.map(process_video_wrapper, video_files)

    # Close the pool of processes and wait for them to complete
    pool.close()
    pool.join()

    print("Video processing completed.")