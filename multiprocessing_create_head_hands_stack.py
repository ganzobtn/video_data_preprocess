
import mediapipe as mp
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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green



def get_features(data_path, dest_path):
    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #out = cv2.VideoWriter(os.path.join(dest_path, person, category, video), cv2.VideoWriter_fourcc(
    #    *'MP4V'), fps, (frame_width, frame_height))
    
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (frame_width, frame_height))
    
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
            #blank = image.copy()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape
            # print(shape)
            blank = np.zeros_like(image)
            results = holistic.process(image)
            #print('ff:', results.right_hand_landmarks)
            #print(results.__dir__())
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Right Hand
            mp_drawing.draw_landmarks(
                blank, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Left Hand
            mp_drawing.draw_landmarks(
                blank, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
                blank,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                blank,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            out.write(blank)
            #cv2.imshow('MediaPipe Holistic', cv2.flip(blank, 1))
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
    #time.sleep(1)

    #cv2.destroyWindow('MediaPipe Holistic')
    cap.release()
    out.release()

def get_crop(image, landmarks):


    height, width, _ = image.shape

    x_coordinates = [landmark.x for landmark in landmarks]
    y_coordinates = [landmark.y for landmark in landmarks]
    min_x = max(int(min(x_coordinates)* width)-MARGIN,0)
    min_y = max(int(min(y_coordinates)*height)-MARGIN,0)
    max_x = min(int(max(x_coordinates)*width)+MARGIN,width)
    max_y = min(int(max(y_coordinates)*height)+MARGIN,height)

    
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    return min_x,min_y,max_x,max_y, text_x,text_y

def get_head_hands(data_path, dest_path):

    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (frame_width, frame_height))
    #out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
    #    *'mp4v'), fps, (224, 224))
    
    with mp_holistic.Holistic(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            static_image_mode= False) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            blank = image.copy()
            half_height = int(frame_height/2)
            half_width = int(frame_width/2)
            print('blankK',blank.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape
            #blank = np.zeros_like(image)
            results = holistic.process(image)
            #print('ff:', results.right_hand_landmarks)
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Right Hand
            mp_drawing.draw_landmarks(
               blank, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            #  Left Hand
            mp_drawing.draw_landmarks(
               blank, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
               blank,
               results.face_landmarks,
               mp_holistic.FACEMESH_CONTOURS,
               landmark_drawing_spec=None,
               connection_drawing_spec=mp_drawing_styles
               .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
               blank,
               results.pose_landmarks,
               mp_holistic.POSE_CONNECTIONS,
               landmark_drawing_spec=mp_drawing_styles
               .get_default_pose_landmarks_style())
            #print('ff:',type(results.face_landmarks))
            #print('ddd:',results.face_landmarks.landmark[0])
            #print('--------------------')

            #new_img = np.zeros((224,224,3),dtype=int)
            new_img= np.zeros_like(image)

            img_whole = cv2.resize(blank, (int(frame_height/2),int(frame_width/2)))

            new_img[:half_height,:half_width]= img_whole
            


            # Face
            landmarks = results.face_landmarks
            if landmarks is not None:
                min_x,min_y,max_x,max_y, text_x,text_y = get_crop(image,landmarks.landmark)
                color = (255, 0, 0)
                cv2.putText(blank, "Face",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                blank = cv2.rectangle(blank, (min_x,min_y), (max_x,max_y), color, thickness=2)
                #img_whole = cv2.resize(blank[min_y:max_y,min_x:max_x],(256,256))
                
                img_whole = cv2.resize(image[min_y:max_y,min_x:max_x],(256,256))
                #new_img[112:,:112]= img_whole
                new_img[half_height:,:half_width]= img_whole

            # Left Hand
            landmarks = results.left_hand_landmarks
            if landmarks is not None:

                min_x,min_y,max_x,max_y, text_x,text_y = get_crop(image,landmarks.landmark)
                color = (255, 0, 0)
                blank = cv2.rectangle(blank, (min_x,min_y), (max_x,max_y), color, thickness=2)
                cv2.putText(blank, "Left Hand",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                #img_whole = cv2.resize(blank[min_y:max_y,min_x:max_x],(256,256))

                img_whole = cv2.resize(image[min_y:max_y,min_x:max_x],(256,256))
                #new_img[112:,112:]= img_whole
                new_img[half_height:,half_width:]= img_whole
            #Right Hand
            landmarks = results.right_hand_landmarks
            if landmarks is not None:

                min_x,min_y,max_x,max_y, text_x,text_y = get_crop(image,landmarks.landmark)
                color = (255, 0, 0)
                blank = cv2.rectangle(blank, (min_x,min_y), (max_x,max_y), color, thickness=2)
                cv2.putText(blank, "Right Hand",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                img_whole = cv2.resize(image[min_y:max_y,min_x:max_x],(256,256))
                #img_whole = cv2.resize(blank[min_y:max_y,min_x:max_x],(256,256))

                #new_img[:112,112:]= img_whole
                new_img[:half_height:,half_width:]= img_whole
            # Flip the image horizontally for a selfie-view display.
            new_img_resized = cv2.resize(new_img,(frame_height,frame_width))
            print(new_img_resized.shape)
            print(blank.shape)
            out.write(new_img_resized)
            #cv2.imshow('MediaPipe Holistic', cv2.flip(blank, 1))
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
    #time.sleep(1)

    #cv2.destroyWindow('MediaPipe Holistic')
    cap.release()
    out.release()

# Define your video processing function
def process_video(file_path):
    root =file_path[0],file= file_path[1],dest_path_new = file_path[2] 
    if not os.path.exists(dest_path_new):      
        os.makedirs(dest_path_new)
    # You can return some result if needed
    video_path = os.path.join(root,file)

    get_head_hands(data_path= video_path, dest_path = os.path.join(dest_path_new,file))

    return f"Processed {video_path}"

def main():
    video_dir = "path_to_video_directory"  # Replace with the path to your video directory
    video_files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(".mp4")]

    # Create a ThreadPoolExecutor with the number of CPU cores you want to use
    num_cpus = os.cpu_count()  # This gets the number of available CPU cores
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Process videos in parallel
        results = list(executor.map(process_video, video_files))

    # Print the results
    for result in results:
        print(result)


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
                file.append([root,file,dest_path_new])
                #get_features(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))

    num_cpus = os.cpu_count()  # This gets the number of available CPU cores
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Process videos in parallel
        results = list(executor.map(process_video, files))

    # Print the results
    for result in results:
        print(result)


        print ('--------------------------------')


if __name__ == "__main__":
    main()
