import pandas as pd
from scipy.stats import skew, kurtosis

#Place path below to read data
data = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Segmentations\twisthand_intermission2.csv")

#Columns to extract statistical features
hand_landmark_columns = [f'x_{i}' for i in range(21)] + [f'y_{i}' for i in range(21)] + [f'z_{i}' for i in range(21)]
quaternion_columns = ['w', 'x', 'y', 'z']
acceleration_columns = ['ax', 'ay', 'az']
columns_to_process = hand_landmark_columns + quaternion_columns + acceleration_columns

# Extracting features
features = {}
for col in columns_to_process:
    features[f'stat_mean_{col}'] = data[col].mean()
    features[f'stat_median_{col}'] = data[col].median()
    features[f'stat_std_{col}'] = data[col].std()
    features[f'stat_range_{col}'] = data[col].max() - data[col].min()
    features[f'stat_skewness_{col}'] = skew(data[col])
    features[f'stat_kurtosis_{col}'] = kurtosis(data[col])

# Converts features to a dataframe
features_df = pd.DataFrame(features, index=[0])

# Saves the dataframe to path chosen 
features_df.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Statistical Features\twisthand_statistical_intermission2.csv", index=False)
