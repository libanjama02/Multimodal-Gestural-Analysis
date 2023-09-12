import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_time_domain_features(dataframe):
    
    def mean_absolute_difference(series):
        return np.mean(np.abs(np.diff(series)))

    def mean_difference(series):
        return np.mean(np.diff(series))

    def root_mean_square(series):
        return np.sqrt(np.mean(series**2))

    def peak_count(series):
        peaks, _ = find_peaks(series)
        return len(peaks)

    def time_between_peaks(series):
        peaks, _ = find_peaks(series)
        return np.mean(np.diff(peaks))

    #Columns to apply extraction
    landmark_columns = [f"{coord}_{i}" for i in range(21) for coord in ['x', 'y', 'z']]
    quaternion_columns = ['w', 'x', 'y', 'z']
    acceleration_columns = ['ax', 'ay', 'az']
    columns_to_consider = landmark_columns + quaternion_columns + acceleration_columns
    
    # Extracts features
    features = {}
    for col in columns_to_consider:
        features[f"td_{col}_mean_abs_diff"] = mean_absolute_difference(dataframe[col])
        features[f"td_{col}_mean_diff"] = mean_difference(dataframe[col])
        features[f"td_{col}_rms"] = root_mean_square(dataframe[col])
        features[f"td_{col}_peak_count"] = peak_count(dataframe[col])
        features[f"td_{col}_mean_time_between_peaks"] = time_between_peaks(dataframe[col])

    return pd.DataFrame([features])

# place dataframe below
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv")
features_df = extract_time_domain_features(df)

# saves the extracted features to chosen path 
features_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Time Domain features\twisthand_timedomain_intermission2.csv", index=False)