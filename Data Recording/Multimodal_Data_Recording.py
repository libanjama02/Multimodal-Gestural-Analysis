from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep
import os
import time
from datetime import datetime
import cv2
import mediapipe as mp
import pandas as pd


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
            file_stream.write("Timestamp,w,x,y,z,ax,ay,az\n")
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

#hardcoding MAC address // subject to change
mac_address = "D3:99:34:4D:01:CC"
timestamp = time.strftime("%Y%m%d-%H%M%S") # gets the current timestamp
filename = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug", "imu_data_" + timestamp + ".csv")

device = MetaWear(mac_address)
device.connect()
print("Connected to device")
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug", f"output_{timestamp}.mp4")
    out = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))

    return cap, mp_drawing, mp_hands, hand_landmarks_data, out


def data_collection( state, cap, mp_drawing, mp_hands, hand_landmarks_data, out):
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        frame_id = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # IMU data collection
            if state.data_quat and state.data_acc:
                count = 0
                count += 1
                current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("============>Count", count, end="\r")
                

            # Hand pose data collection 
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_landmarks = [None, None]
                for i, hand_landmark in enumerate(results.multi_hand_landmarks):
                    if i > 1: # only considers the first two hands
                        break
                    if i == 0:  # one hand is white and red
                        mp_drawing.draw_landmarks(
                            image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                    else:  # one hand is green and blue
                        mp_drawing.draw_landmarks(
                            image, hand_landmark, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
                    
                    landmarks = [landmark for landmark in hand_landmark.landmark]
                    hand_landmarks[i] = landmarks
                
                current_time = datetime.utcnow().isoformat()
                hand_landmarks_data.append([frame_id, current_time] + hand_landmarks)

                font = cv2.FONT_HERSHEY_SIMPLEX
                flipped_image = cv2.flip(image, 1)
                cv2.putText(flipped_image, current_time, (10,50), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA) #adding the timestamp to the video for ground truth
                image = cv2.flip(flipped_image,1)

            out.write(cv2.flip(image, 1))

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processing frame: {frame_id}")  # printing frame count
            frame_id += 1

        cap.release()
        out.release()

    print(f"Size of hand landmarks data: {len(hand_landmarks_data)}")

    # Converts hand landmarks data to DataFrame and save as .csv
    df = pd.DataFrame(hand_landmarks_data, columns=['Frame ID', 'Timestamp', 'Hand 1 Landmarks', 'Hand 2 Landmarks'])
    csv_filepath = os.path.join(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug", f"hand_landmarks_{timestamp}.csv")
    df.to_csv(csv_filepath)


# Main function to integrate all steps
def main():
    
    # Hand Pose Initialization
    cap, mp_drawing, mp_hands, hand_landmarks_data, out = hand_pose_initialization()

    # Data Collection and Recording
    data_collection( state, cap, mp_drawing, mp_hands, hand_landmarks_data, out)

# Calls the main function
if __name__ == "__main__":
    main()

'''
Room for improvement = call IMU using main function. That way the IMU recording doesn't start a few seconds before handpose recording. (although I struggled to do this because IMU device kept disconnecting)
Also limit the hands detected to one if you need to. I should've done so in the experiment but didn't ngl.
'''    