import pandas as pd
import matplotlib.pyplot as plt

def analyze_csv_data(file_path):
    df = pd.read_csv(file_path)
    timestamps = pd.to_datetime(df['Timestamp'])
    #Calculate differences between consecutive timestamps (aka the sampling rate)
    time_diffs = timestamps.diff().dropna().dt.total_seconds() * 1000 # milliseconds
    return timestamps, time_diffs

combined_file_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_imu_data_20230801-165125.csv"

timestamps, time_diffs = analyze_csv_data(combined_file_path)

#Plot for Timestamps against samples (linear)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(timestamps, label="Timestamps")
plt.title("Timestamps")
plt.xlabel("Samples")
plt.ylabel("Time")
plt.legend()

#Plot for Time differences against samples (helps to identif outliers)
plt.subplot(1, 2, 2)
plt.plot(time_diffs, label="Time Differences")
plt.title("Time Differences (ms)")
plt.xlabel("Samples")
plt.ylabel("Time Difference (ms)")
plt.legend()

plt.show()

#Statistical summary
print("Data:")
print("Mean:", time_diffs.mean(), "ms")
print("Median:", time_diffs.median(), "ms")
print("Range:", time_diffs.min(), "ms to", time_diffs.max(), "ms")
