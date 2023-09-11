#this is a jumbled script that misses a lot of important features to extract, but it kinda works ig.

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_features(input_path, output_path):
    data = pd.read_csv(input_path)

    # 1. Statistical Features
    mean = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].mean(axis=0)
    std_dev = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].std(axis=0)
    skewness = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].apply(skew)
    kurt = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].apply(kurtosis)

    # 2. Time-Domain Features
    max_values = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].max(axis=0)
    min_values = data[['ax', 'ay', 'az', 'w', 'x', 'y', 'z']].min(axis=0)

    # 3. Frequency-Domain Features
    frequencies = []
    for col in ['ax', 'ay', 'az', 'w', 'x', 'y', 'z']:
        f, Pxx = welch(data[col], fs=200, nperseg=256)
        frequencies.append(pd.Series(Pxx, index=f).sort_values(ascending=False).index[0])
    
    dominant_freq = pd.Series(frequencies, index=['ax', 'ay', 'az', 'w', 'x', 'y', 'z'])

    # 4. Landmark-Based Features
    landmarks = [f'x_{i}' for i in range(0, 21)] + [f'y_{i}' for i in range(0, 21)] + [f'z_{i}' for i in range(0, 21)]
    avg_distances = []
    for i in [8, 12, 16, 20]:
        distance = np.sqrt((data[f'x_{4}'] - data[f'x_{i}'])**2 + 
                           (data[f'y_{4}'] - data[f'y_{i}'])**2 + 
                           (data[f'z_{4}'] - data[f'z_{i}'])**2)
        avg_distances.append(distance.mean())

    # Compile all features
    features = pd.concat([mean, std_dev, skewness, kurt, max_values, min_values, dominant_freq], axis=0)
    features.index = ['mean_' + col for col in mean.index] + \
                     ['std_dev_' + col for col in std_dev.index] + \
                     ['skewness_' + col for col in skewness.index] + \
                     ['kurtosis_' + col for col in kurt.index] + \
                     ['max_' + col for col in max_values.index] + \
                     ['min_' + col for col in min_values.index] + \
                     ['dominant_freq_' + col for col in dominant_freq.index]

    features = pd.concat([features, pd.Series(avg_distances, index=[f'avg_distance_landmark_{i}' for i in [8, 12, 16, 20]])])
    
    # Save the features to the specified path
    features.to_csv(output_path)

# Usage
extract_features(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\preprocessed_aligned_twisthand_data_20230801-165125.csv", r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand_extracted_features.csv")
