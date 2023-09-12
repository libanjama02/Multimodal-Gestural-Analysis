import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

#path to normalized dataframe here
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized.csv")

# Removing category columns
feature_columns = [col for col in df.columns if col not in ['Gesture Name', 'Gesture Type']]

# Prepaing the feature matrix from the dataframe
X = df[feature_columns].values

# Performing K-means clustering (for 5 gestures + intermission)
kmeans = KMeans(n_clusters=6, random_state=17) 
kmeans_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, kmeans_labels)
print(f'Silhouette Score: {silhouette_avg}')

davies_bouldin_avg = davies_bouldin_score(X, kmeans_labels)
print(f'Davies-Bouldin Score: {davies_bouldin_avg}')
