import pandas as pd

# Load the DataFrame
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_filtered.csv")

# Identify columns to be removed (those related to "Mean Time Between Peaks")
cols_to_remove = [col for col in df.columns if 'mean_time_between_peaks' in col]

# Remove the identified columns
df_filtered = df.drop(columns=cols_to_remove)

# Save the new DataFrame to a new CSV file
filtered_df_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_filtered_v2.csv"
df_filtered.to_csv(filtered_df_path, index=False)

filtered_df_path
