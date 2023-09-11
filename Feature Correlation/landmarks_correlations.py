import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final_2.csv") 

def calculate_and_print_correlations(df, prefix, suffixes, categories=None, output_file=None):
    with open(output_file, 'a') as f:
        for suffix1 in suffixes:
            for suffix2 in suffixes:
                if suffix1 >= suffix2: # Skips redundant comparisons
                    continue
                
                if categories:
                    for cat in categories:
                        col1 = f"{prefix}_{cat}_{suffix1}"
                        col2 = f"{prefix}_{cat}_{suffix2}"
                        if col1 in df.columns and col2 in df.columns:
                            corr_value = df[col1].corr(df[col2])
                            f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")
                else:
                    col1 = f"{prefix}_{suffix1}"
                    col2 = f"{prefix}_{suffix2}"
                    if col1 in df.columns and col2 in df.columns:
                        corr_value = df[col1].corr(df[col2])
                        f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")


# Removing label columns 
df = df.drop(columns=['Gesture Name', 'Gesture Type'])  

# creates the output file
with open('correlations_lm_output.txt', 'w') as f:
    f.write("Calculating correlations for landmark features...\n")

# For distance features
with open('correlations_lm_output.txt', 'a') as f:
    f.write("\nCalculating correlations for distance features...\n")
calculate_and_print_correlations(df, 'lm_distance', ['mean', 'median', 'std', 'max', 'min', 'num_peaks'], 
                                 categories=['4_8', '4_12', '4_16', '4_20', '0_4', '0_8', '0_12', '0_16', '0_20', '8_12', '12_16', '16_20'], 
                                 output_file='correlations_lm_output.txt')

# For angle features
with open('correlations_lm_output.txt', 'a') as f:
    f.write("\nCalculating correlations for angle features...\n")
calculate_and_print_correlations(df, 'lm_angle', ['mean', 'median', 'std', 'max', 'min', 'num_peaks'], 
                                 categories=['thumb_wrist_index', 'wrist_middleMCP_middleTip'], 
                                 output_file='correlations_lm_output.txt')

# For curvature features
with open('correlations_lm_output.txt', 'a') as f:
    f.write("\nCalculating correlations for curvature features...\n")

for curv in ['thumb', 'index', 'middle', 'ring', 'pinky', 'hand']:
    prefix = f"lm_{curv}_curvature" #account for weird structure of curvature feature naming convention
    calculate_and_print_correlations(df, prefix, ['mean', 'median', 'std', 'max', 'min'], 
                                     output_file='correlations_lm_output.txt')

# For speed, direction, acceleration, and jerk features
sadj_stats = ['mean', 'median', 'std', 'max', 'min', 'num_dir_changes']
for feature in ['speed', 'direction', 'accel', 'jerk']:
    with open('correlations_lm_output.txt', 'a') as f:
        f.write(f"\nCalculating correlations for {feature} features...\n")
    for axis in ['x', 'y', 'z']:
        for i in range(21):  # 0 to 20
            prefix = f"lm_{feature}_{axis}_{i}"
            calculate_and_print_correlations(df, prefix, sadj_stats, output_file='correlations_lm_output.txt')
