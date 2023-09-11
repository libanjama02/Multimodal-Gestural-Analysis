from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

# Reading the dataset
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

# Excluding category columns
feature_columns = df.columns.difference(['Gesture Name', 'Gesture Type'])

X = df[feature_columns]
y = df['Gesture Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31, stratify=y)

#try using something other than rbf 
svm_model = SVC(kernel='rbf', random_state=33)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

#model accuracy evaluations
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)

print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:", classification_rep_svm)

#5 fold cross-validation
cv_scores = cross_val_score(svm_model, X, y, cv=5)

#print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
#print(f"Standard Deviation of CV Score: {np.std(cv_scores)}")