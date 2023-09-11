import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import os

def main():

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hand_landmarks_data = []

    # For webcam input:
    cap = cv2.VideoCapture(0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # Get the actual frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Data Collected", "output1.mp4")
    out = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))
    
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        frame_id = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = [landmark for landmark in hand_landmarks.landmark]
                    hand_landmarks_data.append([frame_id, i, landmarks])

            out.write(cv2.flip(image, 1))

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

        cap.release()
        out.release()

    print(f"Size of hand landmarks data: {len(hand_landmarks_data)}")

    df = pd.DataFrame(hand_landmarks_data, columns=['Frame ID', 'Hand ID', 'Hand Landmarks'])
    csv_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Data Collected", "hand_landmarks.csv")
    df.to_csv(csv_filepath)

if __name__ == "__main__":
    main()
