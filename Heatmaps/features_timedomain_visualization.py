import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_time_domain_data(df, time_domain_feature):
    """
    This function aggregates the data for a given time domain feature.
    """
    aggregated_data = []
    
    # Aggregate for x, y, z coordinates for all landmarks
    for axis in ['x', 'y', 'z']:
        col_names = [f'td_{axis}_{i}_{time_domain_feature}' for i in range(21)]
        aggregated_value = df[col_names].mean(axis=1).mean()
        aggregated_data.append(aggregated_value)
    
    # Taking Direct values for quaternion and acceleration
    for axis in ['w', 'x', 'y', 'z']:
        col_name = f'td_{axis}_{time_domain_feature}'
        aggregated_data.append(df[col_name].mean())
        
    for axis in ['ax', 'ay', 'az']:
        col_name = f'td_{axis}_{time_domain_feature}'
        aggregated_data.append(df[col_name].mean())
        
    return aggregated_data

#change feature in the line below
def visualize_time_domain_features(time_domain_feature='peak_count'):
    # Put your dataframe paths here
    paths = [
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Time Domain Features\twisthand_timedomain_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\openclose\Time Domain Features\openclose_timedomain_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertRTL\Time Domain Features\insertRTL_timedomain_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertFTB\Time Domain Features\insertFTB_timedomain_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twistpen\Time Domain Features\twistpen_timedomain_gesture5.csv"
    ]
    gestures = ["Twist Hand", "Open Close", "Insert RTL", "Insert FTB", "Twist Pen"]
    
    all_data = []

    for path in paths:
        df = pd.read_csv(path)
        aggregated_data = aggregate_time_domain_data(df, time_domain_feature)
        all_data.append(aggregated_data)
    
    # Convert to DataFrame for visualization
    columns = [f'avg_{axis}' for axis in ['x', 'y', 'z']] + \
              [f'quat_{axis}' for axis in ['w', 'x', 'y', 'z']] + \
              [f'acc_{axis}' for axis in ['ax', 'ay', 'az']]
    
    df_aggregated = pd.DataFrame(all_data, columns=columns)
    df_aggregated['Gesture'] = gestures
    
    # Plotting the heatmap
    plt.figure(figsize=(15, 7))
    sns.heatmap(df_aggregated.set_index('Gesture'), cmap='coolwarm', annot=True, linewidths=.5)
    plt.title(f'Heatmap of {time_domain_feature.capitalize()} Time Domain Features for Different Gestures')
    plt.show()

# Test with 'peak_count'. Highlight all insances of the feature and use Ctrl D to cycle through for convenience 
if __name__ == "__main__":  
    visualize_time_domain_features('peak_count')

'''
time domain features to cycle: mean_abs_diff, mean_diff, rms, peak_count, mean_time_between_peaks(removed this feature later on due to NaNs)
'''