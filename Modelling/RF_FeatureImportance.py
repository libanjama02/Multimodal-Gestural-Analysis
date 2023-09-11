import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

"""
#check for problematic data
missing_values = df.isnull().sum().sum()
infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

#class distribution checker
class_distribution = df['Gesture Type'].value_counts()
print(class_distribution)
"""

#Excluding category columns
feature_columns = df.columns.difference(['Gesture Name', 'Gesture Type'])

#Extracting features and target variable from the dataframe
#May be worth removing intermission in analysis
X = df[feature_columns]
y = df['Gesture Type']

#Splitting the data into a stratified 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31, stratify=y)

#should random state be same as before? and what is estimators
rf_model = RandomForestClassifier(random_state=33, n_estimators=100)

#Fits the model on the training data
rf_model.fit(X_train, y_train)

# Basic featurea analysis below
"""

#extracting feature importances
feature_importances = rf_model.feature_importances_

#Dataframe for feature names and their importance scores
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
})

#Sorting by importance
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

#Displaying top 20 features
top_20_features = sorted_feature_importance_df.head(20)
print(top_20_features)
"""
############################


#Creating a DF for consistency of feature importances across varying random states
consistency_df = pd.DataFrame(columns=['Feature', 'Consistency_Count'])

#change the iteration count
num_runs = 30

for state in range(num_runs):
    rf_model = RandomForestClassifier(random_state=state, n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    #extracting feature importances
    feature_importances = rf_model.feature_importances_
    
    #Dataframe for feature names and their importance scores
    temp_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    })
    
    #Sorting by importance
    sorted_temp_df = temp_df.sort_values(by='Importance', ascending=False).head(20)
    
    #consistency checker
    for feature in sorted_temp_df['Feature']:
        if feature in consistency_df['Feature'].values:
            consistency_df.loc[consistency_df['Feature'] == feature, 'Consistency_Count'] += 1
        else:
            consistency_df = consistency_df.append({'Feature': feature, 'Consistency_Count': 1}, ignore_index=True)

#Sorting
sorted_consistency_df = consistency_df.sort_values(by='Consistency_Count', ascending=False)

#Displaying top most consistent features
top_20_consistent_features = sorted_consistency_df.head(20)
print(top_20_consistent_features)



###############################################

#was considering using top 20 repeated features for model but results seem promising already

