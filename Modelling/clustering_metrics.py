import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Read your data below
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

# Removing category columns
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

# Prepares the feature matrix from the dataframe
X = df[feature_columns].values

# Perform K-means clustering (for 5 gestures)
kmeans = KMeans(n_clusters=5, random_state=42) 
kmeans_labels = kmeans.fit_predict(X)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(X, kmeans_labels)
print(f'Silhouette Score: {silhouette_avg}')

# Calculate the Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X, kmeans_labels)
print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
