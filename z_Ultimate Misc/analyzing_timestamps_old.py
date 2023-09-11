import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to read and analyze CSV data
def analyze_csv_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    # Extract timestamp column
    timestamps = pd.to_datetime(df['Timestamp'])
    # Calculate differences between consecutive timestamps (sampling rate)
    time_diffs = timestamps.diff().dropna().dt.total_seconds() * 1000 # milliseconds
    return timestamps, time_diffs

# File paths
imu_file_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_imu_data_20230801-165125.csv"
hand_pose_file_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_hand_landmarks_20230801-165125.csv"

imu_timestamps, imu_time_diffs = analyze_csv_data(imu_file_path)
hand_pose_timestamps, hand_pose_time_diffs = analyze_csv_data(hand_pose_file_path)

# Plotting for IMU data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(imu_timestamps, label="IMU Timestamps")
plt.title("IMU Timestamps")
plt.xlabel("Samples")
plt.ylabel("Time")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(imu_time_diffs, label="IMU Time Differences")
plt.title("IMU Time Differences (ms)")
plt.xlabel("Samples")
plt.ylabel("Time Difference (ms)")
plt.legend()

plt.show()

# Plotting for Hand Pose data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hand_pose_timestamps, label="Hand Pose Timestamps")
plt.title("Hand Pose Timestamps")
plt.xlabel("Samples")
plt.ylabel("Time")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hand_pose_time_diffs, label="Hand Pose Time Differences")
plt.title("Hand Pose Time Differences (ms)")
plt.xlabel("Samples")
plt.ylabel("Time Difference (ms)")
plt.legend()

plt.show()


# Statistical summary
print("IMU Data:")
print("Mean:", imu_time_diffs.mean(), "ms")
print("Median:", imu_time_diffs.median(), "ms")
print("Range:", imu_time_diffs.min(), "ms to", imu_time_diffs.max(), "ms")
print("\nHand Pose Data:")
print("Mean:", hand_pose_time_diffs.mean(), "ms")
print("Median:", hand_pose_time_diffs.median(), "ms")
print("Range:", hand_pose_time_diffs.min(), "ms to", hand_pose_time_diffs.max(), "ms")