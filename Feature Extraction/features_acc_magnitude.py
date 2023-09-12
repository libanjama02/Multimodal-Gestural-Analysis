import pandas as pd
import numpy as np

def compute_acceleration_magnitude(df):
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    return df

# Computing statistical features from result of calculation above
def compute_statistical_features(df, column_name):
    features = {}
    features[f"{column_name}_mean"] = df[column_name].mean()
    features[f"{column_name}_std"] = df[column_name].std()
    features[f"{column_name}_max"] = df[column_name].max()
    features[f"{column_name}_min"] = df[column_name].min()
    features[f"{column_name}_median"] = df[column_name].median()
    return features

if __name__ == "__main__":
    #input your path here
    df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv")

    # Computing functions
    df = compute_acceleration_magnitude(df)
    acc_mag_features = compute_statistical_features(df, 'acc_mag')
    
    #Saving results. Edit path below
    result_df = pd.DataFrame([acc_mag_features])
    result_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Acceleration Magnitude Features\twisthand_acc_magnitude_intermission2.csv", index=False)
    print("Data processed and saved.") #debug 