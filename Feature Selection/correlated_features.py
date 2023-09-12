#This script was only used as a means for feature selection for acc_mag

import pandas as pd

# Function to find highly correlated features in batches for over 90% correlation value
def find_highly_correlated_features(df, feature_prefix, correlation_threshold=0.9):
    selected_columns = [col for col in df.columns if col.startswith(feature_prefix)]
    selected_df = df[selected_columns]
    
    correlation_matrix = selected_df.corr()
    highly_correlated_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= correlation_threshold:
                highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
    
    highly_correlated_pairs = sorted(highly_correlated_pairs, key=lambda x: abs(x[2]), reverse=True)
    return highly_correlated_pairs

#place dataframe path here
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final_2.csv")

#Dropping the non-feature columns
feature_df = df.drop(columns=['Gesture Name', 'Gesture Type'])

#Assigning feature prefixes for different types of features
feature_prefixes = ['stat', 'td', 'fr', 'lm', 'acc_mag']

for prefix in feature_prefixes:
    print(f"Processing {prefix}-based features...")
    highly_correlated_pairs = find_highly_correlated_features(feature_df, prefix)
    
    for pair in highly_correlated_pairs[:200]:  # Displaying top 200; feel free to change this number
        print(f"Features: {pair[0]}, {pair[1]} | Correlation: {pair[2]}")
    
    input("Press Enter to continue to the next batch...")
