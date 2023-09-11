#used this version to determine which dots are intermissions
#this needs to be modified to account for the "Gesture Type" column 
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the normalized DataFrame
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final.csv")

# Identify feature columns (excluding 'Gesture Name' as it's categorical)
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

# Initialize MDS
mds = MDS(n_components=2, random_state=42)

# Fit the MDS model and obtain the embedded coordinates
X_mds = mds.fit_transform(df[feature_columns])

# Create a new DataFrame to hold the 2D coordinates along with Gesture Name labels
mds_df = pd.DataFrame(X_mds, columns=['x', 'y'])
mds_df['Gesture Name'] = df['Gesture Name']

# Create a custom palette
unique_gestures = df['Gesture Name'].unique()
default_colors = sns.color_palette("Set2", len(unique_gestures))

custom_palette = {}
for i, gesture in enumerate(unique_gestures):
    if 'intermission' in gesture:
        custom_palette[gesture] = 'black'
    else:
        custom_palette[gesture] = default_colors[i % len(default_colors)]

# Plotting the MDS 2D coordinates
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='x', y='y', hue='Gesture Name', data=mds_df, palette=custom_palette, s=60, ax=ax)
plt.title('2D Multi-dimensional Scaling (MDS) of Gestures')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')

# Adjust subplot parameters to make room for the legend
plt.subplots_adjust(right=0.7)
plt.legend(title='Gesture Name', bbox_to_anchor=(1.4, 1.1), loc='upper left')

plt.show()
