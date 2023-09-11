import pandas as pd
import numpy as np

def compute_acceleration_magnitude(df):
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    return df

def process_dataframe():
    # Load the dataframe
    df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_gesture10.csv")

    # Compute acceleration magnitude
    df = compute_acceleration_magnitude(df)

    # Retain only the relevant columns
    relevant_columns = ['Timestamp', 'Index', 'Frame ID', 'ax', 'ay', 'az', 'acc_mag']
    df = df[relevant_columns]

    # Save to a new CSV
    df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Acceleration Magnitude Features\twisthand_acc_magnitude_gesture10.csv", index=False)
    print("Processed dataframe saved.")

if __name__ == "__main__":
    process_dataframe()
