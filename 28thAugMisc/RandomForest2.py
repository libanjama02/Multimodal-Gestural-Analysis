# Import the necessary libraries for Random Forest
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_final_2.csv")

# Separate the data into features and labels
X = df.drop(columns=['Gesture Name', 'Gesture Type'])
y = df['Gesture Type']

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=37)

# Fit the model to the entire dataset
rf_model.fit(X, y)

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Create a dataframe for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the dataframe by the importances
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize the top 20 most important features
plt.figure(figsize=(15, 8))
plt.barh(feature_importance_df['Feature'][:40], feature_importance_df['Importance'][:40])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Most Important Features')
plt.show()
