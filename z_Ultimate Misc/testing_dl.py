import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

#Reading the dataset
df = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\modeling_dataframe_normalized_27thaug.csv")

#Excluding category columns
feature_columns = df.columns.difference(['Gesture Name', 'Gesture Type'])

#Extracting features and target variable from the dataframe
X = df[feature_columns].values
y = df['Gesture Type'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Splitting the data into a stratified 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=31, stratify=y_encoded)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = len(np.unique(y_encoded))

model = SimpleNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(target.cpu().numpy())

all_preds_labels = le.inverse_transform(all_preds)
all_true_labels = le.inverse_transform(all_true)

print("PyTorch Neural Network")
print(classification_report(all_true_labels, all_preds_labels, target_names=le.classes_))
