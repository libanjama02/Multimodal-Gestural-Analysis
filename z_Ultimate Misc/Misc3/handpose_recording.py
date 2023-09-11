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

    timestamp = time.strftime("%Y%m%d-%H%M%S") # get the current timestamp

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Data Collected", f"output_{timestamp}.mp4")
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
                hand_landmarks = [None, None]
                for i, hand_landmark in enumerate(results.multi_hand_landmarks):
                    if i > 1: # only consider the first two hands
                        break
                    if i == 0:  # one hand is white colour
                        mp_drawing.draw_landmarks(
                            image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                    else:  # one hand is green colour
                        mp_drawing.draw_landmarks(
                            image, hand_landmark, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
                    
                    landmarks = [landmark for landmark in hand_landmark.landmark]
                    hand_landmarks[i] = landmarks
                
                hand_landmarks_data.append([frame_id] + hand_landmarks)

            out.write(cv2.flip(image, 1))

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processing frame: {frame_id}")  # print frame count
            frame_id += 1

        cap.release()
        out.release()

    print(f"Size of hand landmarks data: {len(hand_landmarks_data)}")

    df = pd.DataFrame(hand_landmarks_data, columns=['Frame ID', 'Hand 1 Landmarks', 'Hand 2 Landmarks'])
    csv_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Data Collected", f"hand_landmarks_{timestamp}.csv")
    df.to_csv(csv_filepath)

if __name__ == "__main__":
    main()
