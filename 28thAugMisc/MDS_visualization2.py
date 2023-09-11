#The one below is for simplifying the colours and seeing shiz 
#this needs to be modified to account for the "Gesture Type" column 
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the normalized DataFrame
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final.csv")

# Identify feature columns (excluding 'Gesture Name' as it's categorical)
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

# Simplify the gesture names to their base form
df['Simple Gesture Name'] = df['Gesture Name'].apply(lambda x: x.split('_')[0])

# Initialize MDS
mds = MDS(n_components=2, random_state=42)

# Fit the MDS model and obtain the embedded coordinates
X_mds = mds.fit_transform(df[feature_columns])

# Create a new DataFrame to hold the 2D coordinates along with Gesture Name labels
mds_df = pd.DataFrame(X_mds, columns=['x', 'y'])
mds_df['Simple Gesture Name'] = df['Simple Gesture Name']

# Custom color palette
custom_palette = {
    'twisthand': 'blue',
    'openclose': 'red',
    'insertRTL': 'green',
    'insertFTB': 'purple',
    'twistpen': 'orange'
}

# Plotting the MDS 2D coordinates
plt.figure(figsize=(12, 8))
sns.scatterplot(x='x', y='y', hue='Simple Gesture Name', data=mds_df, palette=custom_palette, s=60)
plt.title('2D Multi-dimensional Scaling (MDS) of Gestures')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')
plt.legend(title='Simple Gesture Name', bbox_to_anchor=(1.2, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()