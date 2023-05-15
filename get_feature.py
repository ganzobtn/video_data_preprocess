import mediapipe as mp
import os
import cv2

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str,
                    default='/projects/data/arabic_sign_sampled_videos_resized_augmented_3/60')
parser.add_argument('--model_save_cfg', type=str,
                    default='arabic_sign_sampled_cfg_60_resized_augmented_3rd')



opt = parser.parse_args()



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def get_features(data_path, dest_path):
    cap = cv2.VideoCapture(data_path)

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    #(1920, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #out = cv2.VideoWriter(os.path.join(dest_path, person, category, video), cv2.VideoWriter_fourcc(
    #    *'MP4V'), fps, (frame_width, frame_height))
    
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(
        *'MP4V'), fps, (frame_width, frame_height))

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
            shape = image.shape
            # print(shape)
            blank = np.zeros_like(image)
            results = holistic.process(image)
            print('ff:', results.right_hand_landmarks)
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
    #cv2.destroyWindow('MediaPipe Holistic')
    cap.release()
    out.release()
