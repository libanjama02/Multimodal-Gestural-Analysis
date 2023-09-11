import pandas as pd

def compute_statistical_features(df, column_name):
    """Compute statistical features for a given column."""
    features = {}
    features[f"{column_name}_mean"] = df[column_name].mean()
    features[f"{column_name}_std"] = df[column_name].std()
    features[f"{column_name}_max"] = df[column_name].max()
    features[f"{column_name}_min"] = df[column_name].min()
    features[f"{column_name}_median"] = df[column_name].median()
    
    return features

def main():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Acceleration Magnitude Features\twisthand_acc_magnitude_gesture10.csv")
    
    # Compute the statistical features for acceleration magnitude
    acc_mag_features = compute_statistical_features(df, 'acc_mag')
    
    # Create a new dataframe with the features
    result_df = pd.DataFrame([acc_mag_features])
    
    # Save the results to a new CSV file
    result_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Acceleration Magnitude Features Processed\twisthand_acc_magnitude_P_gesture10.csv", index=False)

if __name__ == "__main__":
    main()
