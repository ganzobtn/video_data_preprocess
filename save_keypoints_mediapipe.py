import mediapipe as mp
import os
import cv2
import numpy as np
import argparse

import time
import pickle
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str,
                    default='')
parser.add_argument('--dest_path', type=str,
                    default ='')


opt = parser.parse_args()


data_path = '/home/ganzorig/Datas/chalearn_processed_full/color/'
dest_path = '/home/ganzorig/Datas/chalearn_processed_full_skeleton/color/'

MARGIN =10

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def get_keypoints(data_path):
    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
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

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #shape = image.shape
            # print(shape)
            results = holistic.process(image)
            print('ff:', results.right_hand_landmarks)
            #print(results.__dir__())
            landmarks = results.face_landmarks
            if landmarks is not None:
                keypoints  = landmarks.landmark
                print(keypoints.x,keypoints.y)
            
        cap.release()

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

def main():
    for (root,dirs,files) in os.walk(data_path, topdown=True):
    #print (root)
    #print (dirs)
    #print (files)
        for file in files:
            if file.endswith('.mp4'):
                #print(os.path.join(root,file))
                dest_path_new = root.replace(data_path,dest_path)
                if not os.path.exists(dest_path_new):      
                    os.makedirs(dest_path_new)

                #get_features(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))
                get_keypoints(data_path=os.path.join(root,file))
        print ('--------------------------------')

if __name__=='__main__':

    #data_path = '/tmp/data/wlasl_2000/WLASL2000'
    #dest_path = '/tmp/data/wlasl_2000/wlasl_2000_head_hands_stack/WLASL2000'

    main()