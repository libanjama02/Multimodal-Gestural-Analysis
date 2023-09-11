import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_acc_mag_data(df):
    """
    Aggregates acceleration magnitude data for each feature column.
    """
    aggregated_data = {}
    for col in df.columns:
        aggregated_data[col] = df[col].mean()
    return aggregated_data

def visualize_acc_mag_features():
    #Place your paths below
    paths = [
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Acceleration Magnitude Features\twisthand_acc_magnitude_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\openclose\Acceleration Magnitude Features\openclose_acc_magnitude_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertRTL\Acceleration Magnitude Features\insertRTL_acc_magnitude_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertFTB\Acceleration Magnitude Features\insertFTB_acc_magnitude_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twistpen\Acceleration Magnitude Features\twistpen_acc_magnitude_gesture5.csv",
    ]

    gestures = ["Twist Hand", "Open Close", "Insert RTL", "Insert FTB", "Twist Pen"]  
    
    all_data = []
    for path in paths:
        df = pd.read_csv(path)
        aggregated_data = aggregate_acc_mag_data(df)
        all_data.append(aggregated_data)
    
    # Convert to DataFrame for visualization
    df_aggregated = pd.DataFrame(all_data)
    df_aggregated['Gesture'] = gestures
    
    # Plotting the heatmap
    plt.figure(figsize=(12, len(gestures)))
    sns.heatmap(df_aggregated.set_index('Gesture'), cmap='coolwarm', annot=True, linewidths=.5)
    plt.title('Heatmap of Acceleration Magnitude Features for Different Gestures')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Execute visualization
if __name__ == "__main__":  
    visualize_acc_mag_features()

'''
acc mag features (no need to cycle): mean, std, max, min, median
'''