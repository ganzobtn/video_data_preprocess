import mediapipe as mp
import os
import cv2
import numpy as np
import argparse

import time
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

            img_whole = cv2.resize(blank, (256,256))

            new_img[:256,:256]= img_whole
            


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
                new_img[256:,:256]= img_whole

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
                new_img[256:,256:]= img_whole
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
                new_img[:256:,256:]= img_whole
            # Flip the image horizontally for a selfie-view display.
            new_img_resized = cv2.resize(new_img,(256,256))
            print(new_img_resized.shape)
            print(blank.shape)
            out.write(new_img)
            #cv2.imshow('MediaPipe Holistic', cv2.flip(blank, 1))
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
    #time.sleep(1)

    #cv2.destroyWindow('MediaPipe Holistic')
    cap.release()
    out.release()

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
                #createFolder()

                #get_features(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))
                get_head_hands(data_path=os.path.join(root,file), dest_path = os.path.join(dest_path_new,file))
        print ('--------------------------------')



if __name__=='__main__':

    data_path = '/tmp/data/wlasl_2000/WLASL2000'
    dest_path = '/tmp/data/wlasl_2000/wlasl_2000_head_hands_stack/WLASL2000'
    print("Hi")
    main()
    #pass


    #get_head_hands(data_path=data_path,dest_path= dest_path)
