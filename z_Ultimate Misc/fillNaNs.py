import pandas as pd

# Load the dataframe
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features\twisthand_landmarks_gesture10.csv", index_col=0)

# Fill NaN values with 0
df_filled = df.fillna(0)

# Save the filled dataframe to a new .csv file
df_filled.to_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\Landmark Features Filled\twisthand_landmarks_f_gesture10.csv")
