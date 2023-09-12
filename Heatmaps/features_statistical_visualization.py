import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_stat_data(df, statistical_feature):
    """
    This function aggregates the data for a given statistical feature.
    """
    aggregated_data = []
    
    # Aggregates for x, y, z coordinates for all landmarks
    for axis in ['x', 'y', 'z']:
        col_names = [f'stat_{statistical_feature}_{axis}_{i}' for i in range(21)]
        aggregated_value = df[col_names].mean(axis=1).mean()
        aggregated_data.append(aggregated_value)
    
    # Taking Direct values for quaternion and acceleration
    for axis in ['w', 'x', 'y', 'z']:
        col_name = f'stat_{statistical_feature}_{axis}'
        aggregated_data.append(df[col_name].mean())
        
    for axis in ['ax', 'ay', 'az']:
        col_name = f'stat_{statistical_feature}_{axis}'
        aggregated_data.append(df[col_name].mean())
        
    return aggregated_data

#change feature in the line below
def visualize_stat_features(statistical_feature='kurtosis'):
    # Place your dataframe paths here
    paths = [
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Statistical Features\twisthand_statistical_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\openclose\Statistical Features\openclose_statistical_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertRTL\Statistical Features\insertRTL_statistical_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertFTB\Statistical Features\insertFTB_statistical_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twistpen\Statistical Features\twistpen_statistical_gesture5.csv"
    ]
    gestures = ["Twist Hand", "Open Close", "Insert RTL", "Insert FTB", "Twist Pen"]
    
    all_data = []

    for path in paths:
        df = pd.read_csv(path)
        aggregated_data = aggregate_stat_data(df, statistical_feature)
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
    plt.title(f'Heatmap of {statistical_feature.capitalize()} Statistical Features for Different Gestures')
    plt.show()

# Test with 'kurtosis'. Highlight all insances of the feature and use Ctrl D to cycle through for convenience  
if __name__ == "__main__":  
    visualize_stat_features('kurtosis')

'''
statistical features to cycle: mean, median, range, std, skewness, kurtosis
'''