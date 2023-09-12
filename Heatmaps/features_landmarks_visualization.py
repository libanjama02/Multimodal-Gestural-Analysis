import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def aggregate_distance_data(df, feature_type):
    """
    Aggregates distance data based on the given feature type.
    """
    aggregated_data = {}
    landmark_pairs = [
        '4_8', '4_12', '4_16', '4_20', '0_4', '0_8', '0_12', '0_16', '0_20', 
        '8_12', '12_16', '16_20'
    ]
    
    for pair in landmark_pairs:
        col_name = f'lm_distance_{pair}_{feature_type}'
        if col_name in df.columns:
            aggregated_data[col_name] = df[col_name].mean()
    return aggregated_data

def aggregate_angle_data(df, feature_type):
    """
    Aggregates angle data based on the given feature type.
    """
    aggregated_data = {}
    angles = ['thumb_wrist_index', 'wrist_middleMCP_middleTip']
    
    for angle in angles:
        col_name = f'lm_angle_{angle}_{feature_type}'
        if col_name in df.columns:
            aggregated_data[col_name] = df[col_name].mean()
    return aggregated_data

def aggregate_curvature_data(df, feature_type):
    """
    Aggregates curvature data based on the given feature type.
    """
    aggregated_data = {}
    curvatures = ['thumb', 'index', 'middle', 'ring', 'pinky', 'hand']
    
    for curvature in curvatures:
        col_name = f'lm_{curvature}_curvature_{feature_type}'
        if col_name in df.columns:
            aggregated_data[col_name] = df[col_name].mean()
    return aggregated_data

def aggregate_metric_data(df, feature_type, metric):
    """
    Aggregates metric (speed, direction, acceleration, jerk) data based on the given feature type.
    """
    aggregated_data = {}
    for axis in ['x', 'y', 'z']:
        values = []
        for i in range(21):  
            col_name = f'lm_{metric}_{axis}_{i}_{feature_type}'
            if col_name in df.columns:
                values.append(df[col_name].mean())
        
        # Here we take the mean of the means to consolidate the data for all landmarks
        aggregated_data[f'{metric}_{axis}_{feature_type}'] = sum(values) / len(values) if values else None
    
    return aggregated_data


def aggregate_data(df, feature_type, category):
    """
    Calls the appropriate aggregation function based on the feature category.
    """
    if category == "distance":
        return aggregate_distance_data(df, feature_type)
    elif category == "angle":
        return aggregate_angle_data(df, feature_type)
    elif category == "curvature":
        return aggregate_curvature_data(df, feature_type)
    elif category in ["speed", "direction", "accel", "jerk"]:
        return aggregate_metric_data(df, feature_type, category)
    else:
        return {}

#edit the feature type and catergory below.
def visualize_features(feature_type='num_dir_changes', category='jerk'):
    #Place your paths below
    paths = [
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features\twisthand_landmarks_gesture5.csv",
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\openclose\Landmark Features\openclose_landmarks_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertRTL\Landmark Features\insertRTL_landmarks_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\insertFTB\Landmark Features\insertFTB_landmarks_gesture5.csv", 
        r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twistpen\Landmark Features\twistpen_landmarks_gesture5.csv"
    ]
    gestures = ["Twist Hand", "Open Close", "Insert RTL", "Insert FTB", "Twist Pen"]
    
    all_data = []

    for path in paths:
        df = pd.read_csv(path)
        aggregated_data = aggregate_data(df, feature_type, category)
        all_data.append(aggregated_data)
    
    # Convert to DataFrame for visualization
    df_aggregated = pd.DataFrame(all_data)
    df_aggregated['Gesture'] = gestures
    
    # Plotting the heatmap
    plt.figure(figsize=(15, 7))
    sns.heatmap(df_aggregated.set_index('Gesture'), cmap='coolwarm', annot=True, linewidths=.5)
    plt.title(f'Heatmap of {category.capitalize()} {feature_type.capitalize()} for Different Gestures')
    plt.xticks(rotation=45, ha='right') 
    plt.show()

# Test with 'num_dir_changes' feature and 'jerk' category. Highlight all insances of the feature and use Ctrl D to cycle through for convenience  
visualize_features('num_dir_changes', 'jerk')

'''
>feature types: 'distance','angle','curvature','speed', 'direction','accel','jerk'
>you can use 'mean','median','std','max','min' on all feature types
>you can only use 'num_peaks' on 'distance' & 'angle' features
>you can only use 'num_dir_changes' on 'speed','accel','direction','jerk'
'''