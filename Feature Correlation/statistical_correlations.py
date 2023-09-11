import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final_2.csv")

def calculate_and_print_correlations(df, feature_groups, axis_list, index_list=None, output_file=None):
    with open(output_file, 'a') as f:  
        for axis in axis_list:
            for feature1 in feature_groups:
                for feature2 in feature_groups:
                    if feature1 >= feature2:  # Skips redundant comparisons
                        continue
                    
                    if index_list:
                        for i in index_list:
                            col1 = f"{feature1}_{axis}_{i}"
                            col2 = f"{feature2}_{axis}_{i}"
                            
                            if col1 in df.columns and col2 in df.columns:
                                corr_value = df[col1].corr(df[col2])
                                f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")
                    else:
                        col1 = f"{feature1}_{axis}"
                        col2 = f"{feature2}_{axis}"
                        
                        if col1 in df.columns and col2 in df.columns:
                            corr_value = df[col1].corr(df[col2])
                            f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")


# Defines feature groups and axis lists
feature_groups_stat = ['stat_mean', 'stat_median', 'stat_std', 'stat_range', 'stat_skewness', 'stat_kurtosis']
axis_list_landmarks = ['x', 'y', 'z']
axis_list_quaternions = ['w', 'x', 'y', 'z']
axis_list_accelerations = ['ax', 'ay', 'az']
index_list_landmarks = list(range(0, 21))

# Calculate correlations for landmarks. Writing to file
with open('correlations_output.txt', 'w') as f:  
    f.write("Calculating correlations for landmarks...\n")
calculate_and_print_correlations(df, feature_groups_stat, axis_list_landmarks, index_list_landmarks, output_file='correlations_stat_output.txt')

# Calculate correlations for quaternions. Appending to file
with open('correlations_output.txt', 'a') as f: 
    f.write("\nCalculating correlations for quaternions...\n")
calculate_and_print_correlations(df, feature_groups_stat, axis_list_quaternions, output_file='correlations_stat_output.txt')

# Calculate correlations for accelerations. Appending to file
with open('correlations_output.txt', 'a') as f:  
    f.write("\nCalculating correlations for accelerations...\n")
calculate_and_print_correlations(df, feature_groups_stat, axis_list_accelerations, output_file='correlations_stat_output.txt')

