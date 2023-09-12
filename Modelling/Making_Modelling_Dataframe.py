import pandas as pd
import os

# Define the root path for the features (this should contain all feature extracted segments)
# The code below is structured to access specific folders within this root path
root_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments"

# Defines the gesture names and segments
gesture_segments = [
    ("twisthand", ["intermission1", "gesture1", "gesture2", "gesture3", "gesture4", "gesture5", "gesture6", "gesture7", "gesture8", "gesture9", "gesture10", "intermission2"]),
    ("openclose", ["intermission1", "gesture1", "gesture2", "gesture3", "gesture4", "gesture5", "gesture6", "gesture7", "gesture8", "gesture9", "gesture10", "intermission2"]),
    ("insertRTL", ["intermission1", "gesture1", "gesture2", "gesture3", "gesture4", "gesture5", "gesture6", "gesture7", "gesture8", "gesture9", "gesture10", "intermission2"]),
    ("insertFTB", ["intermission1", "gesture1", "gesture2", "gesture3", "gesture4", "gesture5", "gesture6", "gesture7", "gesture8", "gesture9", "gesture10", "intermission2"]),
    ("twistpen", ["intermission1", "gesture1", "gesture2", "gesture3", "gesture4", "gesture5", "gesture6", "gesture7", "gesture8", "gesture9", "gesture10", "intermission2"])
]

# Defines the feature types and corresponding folder names
feature_folders = {
    "statistical": "Statistical Features",
    "timedomain": "Time Domain Features",
    "frequency": "Frequency Features",
    "landmarks": "Landmark Features",
    "acc_magnitude": "Acceleration Magnitude Features"
}

#creating an empty dataframe to store the final output
modeling_dataframe = pd.DataFrame()

#This iterates through gesture names and segments
for gesture_name, segments in gesture_segments:
    for segment in segments:
        #initializing a dictionary to store the data for this segment
        segment_data = {"Gesture Name": f"{gesture_name}_{segment}"}

        #iterating through feature types
        for feature_type, folder_name in feature_folders.items():
            #Constructs the full path for this feature type
            base_path = os.path.join(root_path, gesture_name, folder_name)
            #Constructs the filename
            filename = f"{gesture_name}_{feature_type}_{segment}.csv"
            filepath = os.path.join(base_path, filename)

            #Reading dataframe path
            df = pd.read_csv(filepath)
            segment_data.update(df.iloc[0].to_dict())  

        #Append the segment data to the final dataframe
        modeling_dataframe = modeling_dataframe.append(segment_data, ignore_index=True)

#Saving results to chosen path below
modeling_dataframe.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_2.csv", index=False)
