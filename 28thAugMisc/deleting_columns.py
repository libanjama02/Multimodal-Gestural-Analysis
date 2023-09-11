import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_4_filtered.csv")


# Remove columns with "mid_energy" or "high_energy" in their names and save it as a new DataFrame

# Identify columns to be removed
cols_to_remove = [col for col in df.columns if 'mid_energy' in col or 'high_energy' in col or 'mean_time_between_peaks' in col or 'dominant_frequency' in col or 'stat_median' in col or 'stat_range' in col or '' ]

# Remove the identified columns
df_filtered = df.drop(columns=cols_to_remove)

# Save the new DataFrame to a new CSV file
filtered_df_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_27thaug_filtered.csv"
df_filtered.to_csv(filtered_df_path, index=False)

filtered_df_path
