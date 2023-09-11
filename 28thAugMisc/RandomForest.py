import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Step 1: Read the initial DataFrame
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized.csv")

# Step 2: Remove 'intermission2' rows
df = df[df['Gesture Name'].str.contains('intermission2') == False]

# Step 3: Aggregate data by base gesture name
def extract_base_gesture_name(name):
    return re.sub(r'(_gesture\d+|_intermission\d+)', '', name)

df['Base Gesture Name'] = df['Gesture Name'].apply(extract_base_gesture_name)
agg_df = df.groupby('Base Gesture Name').mean().reset_index()

# Step 4: Run Random Forest Classifier to get feature importance
X = agg_df.drop(['Base Gesture Name'], axis=1)
y = agg_df['Base Gesture Name']

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

# Plot feature importance
feature_importance = clf.feature_importances_
sorted_idx = feature_importance.argsort()

plt.figure(figsize=(10, 12))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.show()


