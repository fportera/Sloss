import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F

from sklearn.metrics import f1_score

from ucimlrepo import fetch_ucirepo 

import sys

# fetch dataset 
ds = fetch_ucirepo(id=43) 

X = ds.data.features
y = ds.data.targets

X.fillna(0, inplace=True)

y = y.values

y = np.where(y == 2, 1, 0)

y = y.ravel()

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% for testing, 80% for training
    random_state=42#,     # Sets a seed so your split is reproducible every time
#    stratify=y           # Highly recommended for Haberman! (Keeps class proportions even)
)

# Now, X_train and y_train are your training sets.
print(f"Training features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")

# 2. Define the Neural Network Architecture, 
class NN(nn.Module):
    def __init__(self, input_dim, nn_lw):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, nn_lw)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(nn_lw, 8)
#        self.fc3 = nn.Linear(16, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 1)
        #        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.drop(x)
        x = self.relu(self.fc2(x))
        # x = self.drop(x)
#        x = self.relu(self.fc3(x))
        x = self.bn(x)
        #x = self.sigmoid(self.fc4(x))
        x = self.fc4(x)
        
        return x

# 3. Training and Evaluation Pipeline
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# hyper-parameters: nn_layer_width, nn_lr

bestCVF1 = 0
best_nn_lw = 0

nn_lr = 0.001

nRuns = 5

# def ourBCE(output, target):

#     loss = (- (target * torch.log(output) + (1-target) * torch.log(1 - output))).sum / target.shape[0]

#     return loss

# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for nn_lw_r in range(8):
    nn_lw = 16 + nn_lw_r * 10
    print(f'\nH-P: nn_lw = {nn_lw}')

    mrunf1 = 0

    for run in range(nRuns):

        print(f'\nRun {run}')

        fold_f1 = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            print(f'--- Fold {fold+1} ---')
                
            # Split data
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_val_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)
                
            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            #            train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, drop_last=True) # X_train_tensor.shape[0], shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size = X_train_tensor.shape[0], shuffle=True)
            
            # Initialize model, loss function, and optimizer
            model = NN(input_dim=X_train_fold.shape[1], nn_lw = nn_lw).to(device)
            num_negatives = (y_train_fold == 0).sum()
            num_positives = (y_train_fold == 1).sum()
            
            # Weight ratio for positive class
            pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            #            criterion = nn.BCELoss()
            # criterion = nn.BCELoss(reduction='none')
            # criterion = nn.BCEWithLogitsLoss()
            print(f'nn_lr = {nn_lr}')
            optimizer = optim.Adam(model.parameters(), lr = nn_lr, weight_decay=1e-4)

            # Train the model
            epochs = 100
            for epoch in range(epochs):
                model.train()
                tloss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()

                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    preds = model(batch_X)
                    probs = torch.sigmoid(preds).cpu()
                    batch_y = batch_y.float()
                    loss = criterion(preds, batch_y)
                    tloss += loss.item()
                    loss.backward()
                    
                    optimizer.step()
                        
                    # Determine F1 score for CV training
                    probs_01 = (probs >= 0.5).int()
                    f1 = f1_score(batch_y.flatten().cpu(), probs_01)                           
#                    print(f'f1 training = {f1:8f}')                
                    
#                print(f'Epoch total loss = {tloss}')                
                
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                X_val_tensor = X_val_tensor.to(device)
                val_preds = model(X_val_tensor).flatten()
                val_probs = torch.sigmoid(val_preds).cpu()
                val_01 = (val_probs >= 0.5).int()
                #print(f'val_01 = {val_01}')
                #print(f'val targets = {y_val_tensor.flatten()}')
                f1 = f1_score(y_val_tensor.flatten().cpu().numpy(), val_01.numpy())
                fold_f1.append(f1)
                print(f'Run {run} Validation f1: {f1:.8f}')
                
                sys.stdout.flush()

        # 4. Overall Results
        print('\n=== Cross-Validation Summary ===')
        print(f'Average validation f1: {np.mean(fold_f1):.8f}')
        print(f'Standard Deviation: {np.std(fold_f1):.8f}')
        sys.stdout.flush()
        
        f1 = np.mean(fold_f1)
        
        mrunf1 += f1

    mrunf1 /= nRuns

    print(f'Mean runs f1 = {mrunf1}')
            
    if (bestCVF1 < mrunf1):
        bestCVF1 = mrunf1
        best_nn_lw = nn_lw
        print(f'Improved bestCVf1 = {bestCVF1}, nn_lw = {nn_lw}')
        sys.stdout.flush()

nn_lw = best_nn_lw

print(f'Best CV H-P: nn_lw = {nn_lw}')

# Split data
X_train_fold, X_test_fold = X_train, X_test
y_train_fold, y_test_fold = y_train, y_test

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fold)
X_test_scaled = scaler.transform(X_test_fold)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_fold).unsqueeze(1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=X_train_tensor.shape[0], shuffle=True)
            
# Initialize model, loss function, and optimizer
num_negatives = (y_train_fold == 0).sum()
num_positives = (y_train_fold == 1).sum()

# Weight ratio for positive class
pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
f1s = []

for run in range(nRuns):

    print(f'\nRun {run}')
    
    model = NN(input_dim=X_train_tensor.shape[1], nn_lw=nn_lw)
        
    optimizer = optim.Adam(model.parameters(), lr = nn_lr)
    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        tloss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            tloss += loss.item()
            loss.backward()
            optimizer.step()
            
#        print(f'Epoch total loss = {tloss}')
            
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).flatten()
        test_probs = torch.sigmoid(test_preds).cpu()
        test_probs_cls = (test_probs >= 0.5).int()
        f1 = f1_score(y_test_tensor.flatten(), test_probs_cls)
        f1s.append(f1)

        print(f'Run {run} Test f1: {f1:.8f}')
        
print(f'Average test f1: {np.mean(f1s):.8f}')
print(f'Standard Deviation test f1: {np.std(f1s):.8f}')
