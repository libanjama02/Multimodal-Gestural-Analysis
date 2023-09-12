import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split

#place dataframe path
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

#Excluding category columns
feature_columns = df.columns.difference(['Gesture Name', 'Gesture Type'])

X = df[feature_columns]
y = df['Gesture Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31, stratify=y)

#remember to tune K
knn_model = KNeighborsClassifier(n_neighbors=4)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy, classification_rep)

#5 fold cross-validation
cv_scores = cross_val_score(knn_model, X, y, cv=5)

#print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
#print(f"Standard Deviation of CV Score: {np.std(cv_scores)}")