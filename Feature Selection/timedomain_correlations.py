import pandas as pd
#place path here
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv") 

def calculate_and_print_correlations(df, feature_groups, axis_list, index_list=None, output_file=None):
    with open(output_file, 'a') as f:
        for axis in axis_list:
            for feature1 in feature_groups:
                for feature2 in feature_groups:
                    if feature1 >= feature2: #Skipping redundant comparisons
                        continue
                    
                    if index_list:
                        for i in index_list:
                            col1 = f"td_{axis}_{i}_{feature1}"
                            col2 = f"td_{axis}_{i}_{feature2}"
                            if col1 in df.columns and col2 in df.columns:
                                corr_value = df[col1].corr(df[col2])
                                f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")
                    else:
                        col1 = f"td_{axis}_{feature1}"
                        col2 = f"td_{axis}_{feature2}"
                        if col1 in df.columns and col2 in df.columns:
                            corr_value = df[col1].corr(df[col2])
                            f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")


# Removes label columns
df = df.drop(columns=['Gesture Name', 'Gesture Type'])  

# Defines feature groups and axis lists
feature_groups = ['mean_abs_diff', 'mean_diff', 'rms', 'peak_count']
axis_list_landmarks = ['x', 'y', 'z']
axis_list_quaternions = ['w', 'x', 'y', 'z']
axis_list_accelerations = ['ax', 'ay', 'az']
index_list_landmarks = list(range(0, 21))

# Creates file
with open('correlations_td_output.txt', 'w') as f:
    f.write("Calculating correlations for time-domain features...\n")

# Landmark correlation
with open('correlations_td_output.txt', 'a') as f:
    f.write("\nCalculating correlations for landmarks...\n")
calculate_and_print_correlations(df, feature_groups, axis_list_landmarks, index_list_landmarks, 'correlations_td_output.txt')

# Quat correlation
with open('correlations_td_output.txt', 'a') as f:
    f.write("\nCalculating correlations for quaternions...\n")
calculate_and_print_correlations(df, feature_groups, axis_list_quaternions, None, 'correlations_td_output.txt')

# Accel correlation
with open('correlations_td_output.txt', 'a') as f:
    f.write("\nCalculating correlations for accelerations...\n")
calculate_and_print_correlations(df, feature_groups, axis_list_accelerations, None, 'correlations_td_output.txt')
