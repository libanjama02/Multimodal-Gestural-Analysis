from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Path to dataframe goes here
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

#Remove category columns
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

# Perform MDS
mds = MDS(n_components=2, random_state=41)
df_mds = mds.fit_transform(df[feature_columns])

#Creates a dataFrame for MDS results
df_mds_result = pd.DataFrame(df_mds, columns=['x', 'y'])
df_mds_result['Gesture Name'] = df['Gesture Name']
df_mds_result['Gesture Type'] = df['Gesture Type']

# Visualizing the MDS results
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_mds_result, x='x', y='y', hue='Gesture Type', style='Gesture Name', palette='Set2')
plt.title('MDS Visualization of Gestures')
plt.show()
