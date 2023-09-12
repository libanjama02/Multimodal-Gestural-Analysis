import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_27thaug_filtered2.csv")

#Identifies feature columns and removes label columns
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

scaler = MinMaxScaler()

#Apply Min-Max scaling to the feature columns
df_normalized = df.copy()
df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])

#saves the normalized dataframe to chosen path 
normalized_df_path = r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv"
df_normalized.to_csv(normalized_df_path, index=False)

normalized_df_path
