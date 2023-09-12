import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe.csv")

#Sampling a few columns from each feature set for visualization (in this case 5)
sample_cols_stat = [col for col in df.columns if 'stat_' in col][:5]
sample_cols_time = [col for col in df.columns if 'td_' in col and 'stat_' not in col][:5] # to prevent it from reading stat_std 
sample_cols_freq = [col for col in df.columns if 'fr_' in col][:5]
sample_cols_lm = [col for col in df.columns if 'lm_' in col][:5]
sample_cols_acc = [col for col in df.columns if 'acc_' in col][:5] #only 3 feature columns exist as of 27thaug (others deleted)

#Consolidating sample columns
sample_cols = sample_cols_stat + sample_cols_time + sample_cols_freq + sample_cols_lm + sample_cols_acc

#Plotting the distributions
plt.figure(figsize=(20, 15))
for i, col in enumerate(sample_cols, 1):
    plt.subplot(5, 5, i)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()

#Check for missing vals
missing_values = df.isnull().sum().sum()

#Summary statistics for the sample columns
stat_summary = df[sample_cols].describe()

print("Missing Values: ")
print(missing_values)
#print("Stat Summary: ")
#print(stat_summary)
