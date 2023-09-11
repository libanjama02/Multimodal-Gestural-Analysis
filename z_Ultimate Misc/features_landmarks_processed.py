import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def extract_landmark_features_statistics(df):
    # Placeholder for results
    results = {}

    # For each column in the DataFrame
    for column in df.columns:
        if column not in ["Timestamp", "Index", "Frame ID"]:
            data = df[column].values
            results[f'{column}_mean'] = np.mean(data)
            results[f'{column}_std'] = np.std(data)
            results[f'{column}_max'] = np.max(data)
            results[f'{column}_min'] = np.min(data)
            results[f'{column}_median'] = np.median(data)
            
            # Check if the feature represents distances or angles
            if ("distance" in column) or ("angle" in column):
                peaks, _ = find_peaks(data)
                results[f'{column}_num_peaks'] = len(peaks)
            
            # Check if the feature represents speed, direction, acceleration, jerk
            if any(term in column for term in ["speed", "direction", "accel", "jerk"]):
                direction_changes = np.where(np.diff(np.sign(data)))[0]
                results[f'{column}_num_dir_changes'] = len(direction_changes)

    return results

# Load the data
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features Filled\twisthand_landmarks_f_gesture10.csv")

# Extract statistical features
result = extract_landmark_features_statistics(df)

# Convert dictionary to DataFrame
result_df = pd.DataFrame([result])

# Save to CSV
result_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features Filled Processed\twisthand_landmarks_fP_gesture10.csv", index=False)

