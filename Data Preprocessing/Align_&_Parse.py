import pandas as pd
import re

# This script is designed to be used after collecting the data 

hand_landmarks_df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_hand_landmarks_20230801-165125.csv")  
imu_data_df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\1stAug\twisthand_imu_data_20230801-165125.csv")  

def update_hand1_with_hand2_values(hand_landmarks_df):
    """
    For rows where "Hand 2 Landmarks" has a value, replaces the value in "Hand 1 Landmarks" with it.
    Justification: During my experiment, I only kept one hand in view throughout recording. However,
    there were instances in which Mediapipe detected Hand 1 as Hand 2, usually during erratic movement.
    The data in the Hand 2 column represented the actual hand landmarks during those instances, whereas
    the Hand 1 data (when both hands were detected together) would look more amalgamated in the video ground truth. 

    """
    mask = hand_landmarks_df['Hand 2 Landmarks'].notna()
    hand_landmarks_df.loc[mask, 'Hand 1 Landmarks'] = hand_landmarks_df.loc[mask, 'Hand 2 Landmarks']
    return hand_landmarks_df

def nearest_neighbor_alignment(hand_landmarks_df, imu_data_df):
    """
    Aligns the hand landmarks dataframe with the imu data dataframe using nearest neighbor interpolation based on timestamps.
    """
    # Convert 'Timestamp' columns to datetime objects for both dataframes 
    hand_landmarks_df['Timestamp'] = pd.to_datetime(hand_landmarks_df['Timestamp'])
    imu_data_df['Timestamp'] = pd.to_datetime(imu_data_df['Timestamp'])

    # Set the 'Timestamp' columns as the indices for both dataframes for easier alignment
    hand_landmarks_df.set_index('Timestamp', inplace=True)
    imu_data_df.set_index('Timestamp', inplace=True)

    # Perform the alignment using 'merge_asof'
    aligned_df = pd.merge_asof(hand_landmarks_df.sort_index(), imu_data_df.sort_index(), 
                               left_index=True, right_index=True, direction='nearest')

    # Reset the index
    aligned_df.reset_index(inplace=True)
    return aligned_df

def extract_coordinates(landmark_str):
    """
    Extracts x, y, and z coordinates from the Mediapipe string format. 
    This accounts for small values of Z with magnitudes 10^-7
    """
    x_values = re.findall(r'x: ([\d.-]+(?:e-\d{1,2})?)', landmark_str)
    y_values = re.findall(r'y: ([\d.-]+(?:e-\d{1,2})?)', landmark_str)
    z_values = re.findall(r'z: ([\d.-]+(?:e-\d{1,2})?)', landmark_str)
    return x_values, y_values, z_values

def parse_landmarks(aligned_df):
    """
    Parses the hand landmarks from the aligned dataframe.
    """
    aligned_df['x_values'], aligned_df['y_values'], aligned_df['z_values'] = zip(*aligned_df['Hand 1 Landmarks'].map(extract_coordinates))

    # Expand the lists into 63 separate columns
    for i in range(21):  
        aligned_df[f'x_{i}'] = aligned_df['x_values'].str[i].astype(float)
        aligned_df[f'y_{i}'] = aligned_df['y_values'].str[i].astype(float)
        aligned_df[f'z_{i}'] = aligned_df['z_values'].str[i].astype(float)

    # Drops irrelevant columns
    aligned_df.drop(columns=['Hand 1 Landmarks', 'Hand 2 Landmarks', 'x_values', 'y_values', 'z_values'], inplace=True)
    
    return aligned_df


# Calling all defined functions
#Input recording data
hand_landmarks_df = update_hand1_with_hand2_values(hand_landmarks_df)
aligned_df = nearest_neighbor_alignment(hand_landmarks_df, imu_data_df)
final_df = parse_landmarks(aligned_df)
final_df.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
#Saves to dataframe
final_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!tastetestnozddddd.csv", index=False)  

