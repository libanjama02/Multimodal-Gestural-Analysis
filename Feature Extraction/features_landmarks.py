import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Defining Euclidean distance 
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def extract_distance_features(df):
    # Distances between thumb and other finger tips
    for i in [8, 12, 16, 20]:
        col_name = f'lm_distance_4_{i}'
        df[col_name] = df.apply(lambda row: euclidean_distance(row[f'x_4'], row[f'y_4'], row[f'z_4'],
                                                               row[f'x_{i}'], row[f'y_{i}'], row[f'z_{i}']), axis=1)

    # Distances between wrist and finger tips
    for i in [4, 8, 12, 16, 20]:
        col_name = f'lm_distance_0_{i}'
        df[col_name] = df.apply(lambda row: euclidean_distance(row[f'x_0'], row[f'y_0'], row[f'z_0'],
                                                               row[f'x_{i}'], row[f'y_{i}'], row[f'z_{i}']), axis=1)

    # Distances between adjacent finger tips
    adjacent_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    for pair in adjacent_pairs:
        col_name = f'lm_distance_{pair[0]}_{pair[1]}'
        df[col_name] = df.apply(lambda row: euclidean_distance(row[f'x_{pair[0]}'], row[f'y_{pair[0]}'], row[f'z_{pair[0]}'],
                                                               row[f'x_{pair[1]}'], row[f'y_{pair[1]}'], row[f'z_{pair[1]}']), axis=1)
    return df

def compute_angle(A, B, C):
    AB = euclidean_distance(A[0], A[1], A[2], B[0], B[1], B[2])
    BC = euclidean_distance(B[0], B[1], B[2], C[0], C[1], C[2])
    AC = euclidean_distance(A[0], A[1], A[2], C[0], C[1], C[2])
    angle_rad = np.arccos((AB**2 + BC**2 - AC**2) / (2 * AB * BC))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def extract_angle_features(df):
    df['lm_angle_thumb_wrist_index'] = df.apply(lambda row: compute_angle((row['x_4'], row['y_4'], row['z_4']),
                                                                       (row['x_0'], row['y_0'], row['z_0']),
                                                                       (row['x_8'], row['y_8'], row['z_8'])), axis=1)
    df['lm_angle_wrist_middleMCP_middleTip'] = df.apply(lambda row: compute_angle((row['x_0'], row['y_0'], row['z_0']),
                                                                              (row['x_9'], row['y_9'], row['z_9']),
                                                                              (row['x_12'], row['y_12'], row['z_12'])), axis=1)
    return df

def compute_curvature(A, B, C):
    AB = np.array(A) - np.array(B)
    BC = np.array(B) - np.array(C)
    AB_distance = euclidean_distance(A[0], A[1], A[2], B[0], B[1], B[2])
    BC_distance = euclidean_distance(B[0], B[1], B[2], C[0], C[1], C[2])
    AC_distance = euclidean_distance(A[0], A[1], A[2], C[0], C[1], C[2])
    curvature = 2 * np.linalg.norm(np.cross(AB, BC))
    curvature /= (AB_distance * BC_distance * AC_distance)
    return curvature

def extract_curvature_features(df):
    # Curvature for thumb 
    df['lm_thumb_curvature'] = df.apply(lambda row: compute_curvature((row['x_2'], row['y_2'], row['z_2']),
                                                                  (row['x_3'], row['y_3'], row['z_3']),
                                                                  (row['x_4'], row['y_4'], row['z_4'])), axis=1)
     # Curvature for index finger
    df['lm_index_curvature'] = df.apply(lambda row: compute_curvature((row['x_5'], row['y_5'], row['z_5']),
                                                                  (row['x_6'], row['y_6'], row['z_6']),
                                                                  (row['x_8'], row['y_8'], row['z_8'])), axis=1)

    # Curvature for middle finger
    df['lm_middle_curvature'] = df.apply(lambda row: compute_curvature((row['x_9'], row['y_9'], row['z_9']),
                                                                   (row['x_10'], row['y_10'], row['z_10']),
                                                                   (row['x_12'], row['y_12'], row['z_12'])), axis=1)

    # Curvature for ring finger
    df['lm_ring_curvature'] = df.apply(lambda row: compute_curvature((row['x_13'], row['y_13'], row['z_13']),
                                                                 (row['x_14'], row['y_14'], row['z_14']),
                                                                 (row['x_16'], row['y_16'], row['z_16'])), axis=1)

    # Curvature for pinky
    df['lm_pinky_curvature'] = df.apply(lambda row: compute_curvature((row['x_17'], row['y_17'], row['z_17']),
                                                                  (row['x_18'], row['y_18'], row['z_18']),
                                                                  (row['x_20'], row['y_20'], row['z_20'])), axis=1)

    # General curvature for the hand (using wrist, thumb tip, and pinky tip)
    df['lm_hand_curvature'] = df.apply(lambda row: compute_curvature((row['x_0'], row['y_0'], row['z_0']),
                                                                 (row['x_4'], row['y_4'], row['z_4']),
                                                                 (row['x_20'], row['y_20'], row['z_20'])), axis=1)
    return df

def extract_speed_features(df):
    for i in range(21):
        df[f'lm_speed_x_{i}'] = df[f'x_{i}'].diff()
        df[f'lm_speed_y_{i}'] = df[f'y_{i}'].diff()
        df[f'lm_speed_z_{i}'] = df[f'z_{i}'].diff()
    return df

def extract_direction_features(df):
    for i in range(21):
        df[f'lm_direction_x_{i}'] = np.arctan2(df[f'lm_speed_y_{i}'], df[f'lm_speed_x_{i}'])
        df[f'lm_direction_y_{i}'] = np.arctan2(df[f'lm_speed_z_{i}'], df[f'lm_speed_y_{i}'])
        df[f'lm_direction_z_{i}'] = np.arctan2(df[f'lm_speed_x_{i}'], df[f'lm_speed_z_{i}'])
    return df

def extract_acceleration_features(df):
    for i in range(21):
        df[f'lm_accel_x_{i}'] = df[f'lm_speed_x_{i}'].diff()
        df[f'lm_accel_y_{i}'] = df[f'lm_speed_y_{i}'].diff()
        df[f'lm_accel_z_{i}'] = df[f'lm_speed_z_{i}'].diff()
    return df

def extract_jerk_features(df):
    for i in range(21):
        df[f'lm_jerk_x_{i}'] = df[f'lm_accel_x_{i}'].diff()
        df[f'lm_jerk_y_{i}'] = df[f'lm_accel_y_{i}'].diff()
        df[f'lm_jerk_z_{i}'] = df[f'lm_accel_z_{i}'].diff()
    return df

def extract_all_features(df):
    df = extract_distance_features(df)
    df = extract_angle_features(df)
    df = extract_curvature_features(df)
    df = extract_speed_features(df)
    df = extract_direction_features(df)
    df = extract_acceleration_features(df)
    df = extract_jerk_features(df)

    # Removing the original x, y, z columns for each landmark
    columns_to_drop = [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)] + [f"z_{i}" for i in range(21)] + ["w", "x", "y", "z", "ax", "ay", "az"]
    df = df.drop(columns=columns_to_drop)

    return df

# Fill any possible NaNs (usually in first few frame for speed, direction, acceleration and jerk)
def fill_nans(df):
    return df.fillna(0)

# Extracting statistical features from results to condense data ready for modelling later 
def extract_landmark_features_statistics(df):
    # Keeping dictionary in place to store results later
    results = {}

    for column in df.columns:
        if column not in ["Timestamp", "Index", "Frame ID"]:
            data = df[column].values
            results[f'{column}_mean'] = np.mean(data)
            results[f'{column}_std'] = np.std(data)
            results[f'{column}_max'] = np.max(data)
            results[f'{column}_min'] = np.min(data)
            results[f'{column}_median'] = np.median(data)

            # Checks if the feature represents distances or angles
            if ("distance" in column) or ("angle" in column):
                peaks, _ = find_peaks(data)
                results[f'{column}_num_peaks'] = len(peaks)

            # Checks if the feature represents speed, direction, acceleration, jerk
            if any(term in column for term in ["speed", "direction", "accel", "jerk"]):
                direction_changes = np.where(np.diff(np.sign(data)))[0]
                results[f'{column}_num_dir_changes'] = len(direction_changes)

    return results

# Execute the script here
if __name__ == "__main__":
    # Place your dataframe path
    df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv")

    # Extract features, fill NaNs, and processes statistical data
    df = extract_all_features(df)
    df_filled = fill_nans(df)
    result = extract_landmark_features_statistics(df_filled)

    # Converts dictionary to DataFrame and saves results. Edit path below
    result_df = pd.DataFrame([result])
    result_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features\twisthand_landmarks_intermission2.csv", index=False)
    print("Data processed and saved.")