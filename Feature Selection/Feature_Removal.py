import pandas as pd

#place path here
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_27thaug_filtered.csv")

# Place columns to be removed here
cols_to_remove = [col for col in df.columns if 
                  'mid_energy' in col or 
                  'high_energy' in col or 
                  'mean_time_between_peaks' in col or 
                  'dominant_frequency' in col or 
                  'stat_median' in col or 
                  'stat_range' in col or
                  'acc_mag_max' in col or
                  'acc_mag_min' in col]

#Place columns with specific patterns needed to remove e.g. removing median from scripts that start with lm and contain distance, angle or curvature in it's name. 
additional_cols_to_remove_1 = [col for col in df.columns if 'median' in col and col.startswith('lm_') and ('distance' in col or 'angle' in col or 'curvature' in col)]

#Another set of columns to remove
additional_cols_to_remove_2 = [col for col in df.columns if 'std' in col and col.startswith('lm_') and ('speed' in col or 'direction' in col or 'accel' in col or 'jerk' in col)]

#Combine all lists
cols_to_remove += additional_cols_to_remove_1 + additional_cols_to_remove_2

#Removes the columns
df_filtered = df.drop(columns=cols_to_remove)

#Saving results to chosen path
filtered_df_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_27thaug_filtered2.csv"
df_filtered.to_csv(filtered_df_path, index=False)

filtered_df_path
