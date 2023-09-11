import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_frequency_domain_data(df, frequency_domain_feature):
    """
    This function aggregates the data for a given frequency domain feature.
    """
    aggregated_data = []
    
    # Aggregate for x, y, z coordinates for all landmarks
    for axis in ['x', 'y', 'z']:
        col_names = [f'fr_{axis}_{i}_{frequency_domain_feature}' for i in range(21)]
        aggregated_value = df[col_names].mean(axis=1).mean()
        aggregated_data.append(aggregated_value)
    
    # Direct values for quaternion
    for axis in ['w', 'x', 'y', 'z']:
        col_name = f'fr_{axis}_{frequency_domain_feature}'
        aggregated_data.append(df[col_name].mean())
        
    for axis in ['ax', 'ay', 'az']:
        col_name = f'fr_{axis}_{frequency_domain_feature}'
        aggregated_data.append(df[col_name].mean())
        
    return aggregated_data

#change feature in line below
def visualize_frequency_domain_features(frequency_domain_feature='low_energy'):
    # Paths for dataframes here
    paths = [
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Frequency Features\twisthand_frequency_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\openclose\Frequency Features\openclose_frequency_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertRTL\Frequency Features\insertRTL_frequency_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertFTB\Frequency Features\insertFTB_frequency_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twistpen\Frequency Features\twistpen_frequency_gesture5.csv"
    ]
    gestures = ["Twist Hand", "Open Close", "Insert RTL", "Insert FTB", "Twist Pen"]
    
    all_data = []

    for path in paths:
        df = pd.read_csv(path)
        aggregated_data = aggregate_frequency_domain_data(df, frequency_domain_feature)
        all_data.append(aggregated_data)
    
    # Convert to DataFrame for visualization
    columns = [f'avg_{axis}' for axis in ['x', 'y', 'z']] + \
              [f'avg_quat_{axis}' for axis in ['w', 'x', 'y', 'z']] + \
              [f'avg_{axis}' for axis in ['ax', 'ay', 'az']]
    
    df_aggregated = pd.DataFrame(all_data, columns=columns)
    df_aggregated['Gesture'] = gestures
    
    # Plotting the heatmap
    plt.figure(figsize=(15, 7))
    sns.heatmap(df_aggregated.set_index('Gesture'), cmap='coolwarm', annot=True, linewidths=.5)
    plt.title(f'Heatmap of {frequency_domain_feature.capitalize()} Frequency Domain Features for Different Gestures')
    plt.show()

# Test with 'low_energy'. Highlight all insances of the feature and use Ctrl D to cycle through for convenience 
if __name__ == "__main__":  
    visualize_frequency_domain_features('low_energy')

'''
frequency features to cycle: low_energy, dominant_frequency(removed this feature later due to being mostly 0)
'''