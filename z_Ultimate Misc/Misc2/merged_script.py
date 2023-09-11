from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event
from optparse import OptionParser
import platform
import sys
import os
import time
from datetime import datetime

class State:
    def __init__(self, device="", filename=""):
        self.device = device
        self.filename = filename
        self.samples = 0
        self.callback_all = FnVoid_VoidP_DataP(self.data_handler_all)
        self.data_quat = None
        self.data_acc = None
        self.qw = self.qx = self.qy = self.qz = None
        self.ax = self.ay = self.az = None
        with open(self.filename, "w") as file_stream:
            file_stream.write("Time,w,x,y,z,ax,ay,az\n")
        self.now = time.time()

    def data_handler_all(self, ctx, data):
        values = parse_value(data)
        try:
            values.w
            self.data_quat = values
            self.qw, self.qx, self.qy, self.qz = values.w, values.x, values.y, values.z
        except AttributeError:
            self.data_acc = values
            self.ax, self.ay, self.az = values.x, values.y, values.z
        time_again = time.time()
        elapsed_ms = round((time_again-self.now)*1000)
        self.samples+= 1
        if self.data_quat is not None and self.data_acc is not None:
            with open(self.filename, "a") as file_stream:
                file_stream.write(f"{datetime.utcnow().isoformat()},{self.qw},{self.qx},{self.qy},{self.qz},{self.ax},{self.ay},{self.az}\n")

# Hard-coded MAC address and filename for simplicity
mac_address = "D3:99:34:4D:01:CC"
filename = "testforvisualization4.csv"

# device instead of dl it seems.
device = MetaWear(mac_address)
device.connect()
state = State(device=device, filename=filename)

libmetawear.mbl_mw_settings_set_connection_parameters(state.device.board, 7.5, 7.5, 0, 6000)
sleep(1.5)

# ============= SETUP ============= 
libmetawear.mbl_mw_sensor_fusion_set_mode(state.device.board, SensorFusionMode.NDOF)
libmetawear.mbl_mw_sensor_fusion_set_acc_range(state.device.board, SensorFusionAccRange._8G)
libmetawear.mbl_mw_sensor_fusion_set_gyro_range(state.device.board, SensorFusionGyroRange._2000DPS)
libmetawear.mbl_mw_sensor_fusion_write_config(state.device.board)
libmetawear.mbl_mw_acc_set_odr(state.device.board, 100.0)
libmetawear.mbl_mw_acc_set_range(state.device.board, 16.0)
libmetawear.mbl_mw_acc_write_acceleration_config(state.device.board)

# get quat signal and subscribe
signal_lq = libmetawear.mbl_mw_sensor_fusion_get_data_signal(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_datasignal_subscribe(signal_lq, None, state.callback_all)

# Get accelerometer data
signal_la = libmetawear.mbl_mw_acc_get_acceleration_data_signal(state.device.board)
libmetawear.mbl_mw_datasignal_subscribe(signal_la, None, state.callback_all)

# ============= START =============
libmetawear.mbl_mw_sensor_fusion_enable_data(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_sensor_fusion_start(state.device.board)
libmetawear.mbl_mw_acc_enable_acceleration_sampling(state.device.board)
libmetawear.mbl_mw_acc_start(state.device.board)

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

def hand_pose_initialization():
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

    return cap, mp_drawing, mp_hands, hand_landmarks_data, out


def data_collection(state, cap, mp_drawing, mp_hands, hand_landmarks_data, out):
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        frame_id = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # IMU data collection (from imu3_recording.py)
            if state.data_quat and state.data_acc:
                count += 1
                current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("============>Count", count, end="\r")
                # Data already written in the data handler

            # Hand pose data collection (from handpose_recording.py)
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

    # Convert hand landmarks data to DataFrame and save as CSV
    df = pd.DataFrame(hand_landmarks_data, columns=['Frame ID', 'Hand 1 Landmarks', 'Hand 2 Landmarks'])
    csv_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\Data Collected", f"hand_landmarks_{timestamp}.csv")
    df.to_csv(csv_filepath)

# Main function to integrate all steps
def main():
    # IMU Initialization
    # (Same code as in Step 1)

    # Hand Pose Initialization
    cap, mp_drawing, mp_hands, hand_landmarks_data, out = hand_pose_initialization()

    # Data Collection and Recording
    data_collection(state, cap, mp_drawing, mp_hands, hand_landmarks_data, out)

# Call the main function
if __name__ == "__main__":
    main()

    