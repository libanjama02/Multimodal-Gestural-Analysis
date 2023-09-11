#This script was kind of a mistake as I didn't realize because i removed all Frequency features besides low_energy, there was nothing to really correlate...
#The solution below works similar to 'correlated_features.py' in that it correlates close matching sequential data indiscriminatly.
import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final_2.csv")  

def calculate_and_print_correlations(df, feature_suffix, axis_list, index_list=None, output_file=None):
    with open(output_file, 'a') as f:
        for axis in axis_list:
            if index_list:
                for i in index_list:
                    col1 = f"fr_{axis}_{i}_{feature_suffix}"
                    for axis2 in axis_list:
                        for j in index_list:
                            col2 = f"fr_{axis2}_{j}_{feature_suffix}"
                            if col1 >= col2:
                                continue
                            if col1 in df.columns and col2 in df.columns:
                                corr_value = df[col1].corr(df[col2])
                                f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")
            else:
                col1 = f"fr_{axis}_{feature_suffix}"
                for axis2 in axis_list:
                    col2 = f"fr_{axis2}_{feature_suffix}"
                    if col1 >= col2:
                        continue
                    if col1 in df.columns and col2 in df.columns:
                        corr_value = df[col1].corr(df[col2])
                        f.write(f"Correlation between {col1} and {col2}: {corr_value}\n")


df = df.drop(columns=['Gesture Name', 'Gesture Type'])  

feature_suffix = 'low_energy'
axis_list_landmarks = ['x', 'y', 'z']
axis_list_quaternions = ['w', 'x', 'y', 'z']
axis_list_accelerations = ['ax', 'ay', 'az']
index_list_landmarks = list(range(0, 21))

with open('correlations_fr_output.txt', 'w') as f:
    f.write("Calculating correlations for frequency features...\n")

with open('correlations_fr_output.txt', 'a') as f:
    f.write("\nCalculating correlations for landmarks...\n")
calculate_and_print_correlations(df, feature_suffix, axis_list_landmarks, index_list_landmarks, 'correlations_fr_output.txt')

with open('correlations_fr_output.txt', 'a') as f:
    f.write("\nCalculating correlations for quaternions...\n")
calculate_and_print_correlations(df, feature_suffix, axis_list_quaternions, None, 'correlations_fr_output.txt')

with open('correlations_fr_output.txt', 'a') as f:
    f.write("\nCalculating correlations for accelerations...\n")
calculate_and_print_correlations(df, feature_suffix, axis_list_accelerations, None, 'correlations_fr_output.txt')
