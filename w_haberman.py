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

from ucimlrepo import fetch_ucirepo 

import sys

# fetch dataset 
ds = fetch_ucirepo(id=43) 

X = ds.data.features
y = ds.data.targets.values.ravel()

X.fillna(0, inplace=True)

X = pd.get_dummies(X, drop_first=True)

# The target is 1 = survived, 2 = died. Convert to binary 0 and 1
y = np.where(y == 2, 1, 0)

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% for testing, 80% for training
    random_state=42     # Sets a seed so your split is reproducible every time
    #stratify=y           # Highly recommended for Haberman! (Keeps class proportions even)
)

# Now, X_train and y_train are your training sets.
print(f"Training features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")

class NN(nn.Module):
    def __init__(self, input_dim, nn_lw):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, nn_lw)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nn_lw, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

def trdot(data, i, j):
    return torch.dot(data[i], data[j])
    
gammaF = 1E24

def ourLoss(output, target):
    global Fmat

    bceloss = nn.BCELoss(reduction='none')

    loss = bceloss(input=output, target=target)

    loss = torch.sqrt(F.relu(loss)).flatten()

    dotsum = torch.square(torch.sum(Fmat, dim=0))

    loss = torch.dot(loss, dotsum) / data.shape[0]
    
    return loss


# 3. Training and Evaluation Pipeline
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# hyper-parameters: nn_lw, gammaF

bestCVAcc = 0
best_nn_lw = 0
best_gammaF = 0

nn_lr = 0.001

nRuns = 3

for nn_lw_r in range(8):
    nn_lw = 260 + nn_lw_r * 20
    for nn_gammaF in range(8):
        gammaF = math.pow(2, 12 - nn_gammaF )

        print(f'\nH-P: nn_lw = {nn_lw}, gammaF = {gammaF}')

        mrunacc = 0

        for run in range(nRuns):

            print(f'\nRun {run}')

            fold_accuracies = []
        
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
                train_loader = DataLoader(train_dataset, batch_size=X_train_tensor.shape[0], shuffle=True)
                
                # Initialize model, loss function, and optimizer
                model = NN(input_dim=X_train.shape[1], nn_lw=nn_lw)
                #            criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr = nn_lr)

                # Generate Fmat
                for batch_X, batch_y in train_loader:
                    data = torch.flatten(batch_X, start_dim=1)

                    Fmat = torch.zeros((data.shape[0], data.shape[0]))
                
                    for i in range(data.shape[0]):
                        Fmat[i, i] = 1
                        
                    for i in range(data.shape[0] - 1):
                        for j in range(i + 1, data.shape[0]):
                            dotprod = trdot(data, i, i) + trdot(data, j, j) - 2 * trdot(data, i, j)
                            Fmat[i, j] = torch.exp( - gammaF * dotprod )
                            Fmat[j, i] = Fmat[i,j]
                            
                #               print(f'Fmat shape = {Fmat.shape}')
                print(f'Fmat sum = {Fmat.sum()}')
                            
                # Train the model
                epochs = 150
                for epoch in range(epochs):
                    model.train()
                    tloss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        predictions = model(batch_X)
                        loss = ourLoss(predictions, batch_y)
                        tloss += loss.item()
                        loss.backward()
                        optimizer.step()
                        
                    # print(f'Epoch total loss = {tloss}')
                    
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_tensor)
                    val_preds = val_preds.flatten()
                    val_preds_cls = (val_preds >= 0.2).int()
                    correct = (val_preds_cls == y_val_tensor.flatten()).sum().float()
                    accuracy = correct / len(y_val_tensor)
                    fold_accuracies.append(accuracy.item())
                    print(f'Fold {fold+1} Accuracy: {accuracy.item():.8f}')
                    sys.stdout.flush()


            # 4. Overall Results
            print('\n=== Cross-Validation Summary ===')
            print(f'Average Accuracy: {np.mean(fold_accuracies):.8f}')
            print(f'Standard Deviation: {np.std(fold_accuracies):.8f}')
            sys.stdout.flush()
        
            acc = np.mean(fold_accuracies)

            mrunacc += acc

        mrunacc /= nRuns

        print(f'Mean runs accuracy = {mrunacc}')
            
        if (bestCVAcc < mrunacc):
            bestCVAcc = mrunacc
            best_nn_lw = nn_lw
            best_gammaF = gammaF
            
            print(f'Improved bestCVACC = {bestCVAcc}, nn_lw = {nn_lw}, gammaF = {gammaF}')
            sys.stdout.flush()

nn_lw = best_nn_lw
gammaF = best_gammaF

print(f'Best CV H-P: nn_lw = {nn_lw}, gammaF = {gammaF}')

# Split data
X_train_fold, X_val_fold = X_train, X_test
y_train_fold, y_val_fold = y_train, y_test

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
train_loader = DataLoader(train_dataset, batch_size=X_train_tensor.shape[0], shuffle=True)

# Generate Fmat
for batch_X, batch_y in train_loader:
    data = torch.flatten(batch_X, start_dim=1)
    
    Fmat = torch.zeros((data.shape[0], data.shape[0]))

    for i in range(data.shape[0]):
        Fmat[i, i] = 1
                        
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            dotprod = trdot(data, i, i) + trdot(data, j, j) - 2 * trdot(data, i, j)
            Fmat[i, j] = torch.exp( - gammaF * dotprod )
            Fmat[j, i] = Fmat[i,j]
                            
            #               print(f'Fmat shape = {Fmat.shape}')

    print(f'Fmat sum = {Fmat.sum()}')

mrunacc = 0
            
for run in range(nRuns):

    print(f'\nRun {run}')

    # Initialize model, loss function, and optimizer
    model = NN(input_dim=X_train.shape[1], nn_lw=nn_lw)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = nn_lr)

    # Train the model
    epochs = 150
    for epoch in range(epochs):
        model.train()
        tloss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = ourLoss(predictions, batch_y)
            tloss += loss
            loss.backward()
            optimizer.step()

        print(f'Epoch total loss = {tloss}')
        
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_preds = val_preds.flatten()
        val_preds_cls = (val_preds >= 0.2).int()
        correct = (val_preds_cls == y_val_tensor.flatten()).sum().float()
        accuracy = correct / len(y_val_tensor)
        fold_accuracies.append(accuracy.item())
        print(f'Run {run} Test Accuracy: {accuracy.item():.8f}')
        
        mrunacc += accuracy

mrunacc /= nRuns

print(f'Mean runs Test accuracy = {mrunacc}')
        
