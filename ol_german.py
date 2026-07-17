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
ds = fetch_ucirepo(id=144) 

X = ds.data.features
y = ds.data.targets

X.fillna(0, inplace=True)

X = pd.get_dummies(X, drop_first=True)

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
        self.fc2 = nn.Linear(nn_lw, 16)
        self.fc3 = nn.Linear(16, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 1)
        #        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.drop(x)
        x = self.relu(self.fc2(x))
        # x = self.drop(x)
        x = self.relu(self.fc3(x))
        x = self.bn(x)
        #x = self.sigmoid(self.fc4(x))
        x = self.fc4(x)
        
        return x

def trdot(data, i, j):
    return torch.dot(data[i], data[j])
    
gammaF = 1E24

def ourLoss(output, target):
    global Fmat

    bcewl_loss = nn.BCEWithLogitsLoss(reduction='none')

    loss = bcewl_loss(input=output, target=target)

    loss = torch.sqrt(F.relu(loss))
    
    vloss_p = torch.zeros((data.shape[0]))
    vloss_m = torch.zeros((data.shape[0]))
    
    for i in range(data.shape[0]):
        if (target[i] == 1):
            vloss_p[i] = loss[i]
            vloss_m[i] = 0
        else:
            vloss_p[i] = 0
            vloss_m[i] = loss[i]

    loss_p = torch.dot(vloss_p, torch.matmul(Fmat, vloss_p))
    loss_m = torch.dot(vloss_m, torch.matmul(Fmat, vloss_m))
    
    return loss_p + loss_m

    
# 3. Training and Evaluation Pipeline
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# hyper-parameters: nn_layer_width, nn_lr

bestCVF1 = 0
best_nn_lw = 0
best_gammaF = 0

nn_lr = 0.001

nRuns = 5

# Set up the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for nn_lw_r in range(8):
    nn_lw = 32 + nn_lw_r * 10
    for nn_gammaF in range(8):
        gammaF = math.pow(2, 12 - nn_gammaF )

        print(f'\nH-P: nn_lw = {nn_lw}, gammaF = {gammaF}')

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
                train_loader = DataLoader(train_dataset, batch_size = X_train_tensor.shape[0], shuffle=True)
            
                # Initialize model, loss function, and optimizer
                model = NN(input_dim=X_train_fold.shape[1], nn_lw = nn_lw).to(device)
                optimizer = optim.Adam(model.parameters(), lr = nn_lr, weight_decay=1e-4)

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
                print(f'Fmat sum = {Fmat.sum()}')

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
                        # probs = torch.sigmoid(preds).cpu()
                        batch_y = batch_y.float()
                        loss = ourLoss(preds, batch_y)
                        tloss += loss.item()
                        loss.backward()
                        optimizer.step()
                        
                    # Determine F1 score for CV training
                    # probs_01 = (probs >= 0.5).int()
                    # f1 = f1_score(batch_y.flatten().cpu(), probs_01)              
                    
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
        
            mrunf1 = np.mean(fold_f1)

            print(f'Mean runs f1 = {mrunf1}')

        if (bestCVF1 < mrunf1):
            bestCVF1 = mrunf1
            best_nn_lw = nn_lw
            best_gammaF = gammaF
            
            print(f'Improved bestCVF1 = {bestCVF1}, nn_lw = {nn_lw}, gammaF = {gammaF}')
            sys.stdout.flush()

nn_lw = best_nn_lw

print(f'Best CV H-P: nn_lw = {nn_lw}, gammaF = {gammaF}')

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

mrunf1 = 0
            
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
            loss = ourLoss(predictions, batch_y)
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
        fold_f1.append(f1)
        print(f'Run {run} Test f1: {f1:.8f}')
        
        mrunf1 += f1

mrunf1 /= nRuns

print(f'Mean runs Test f1 = {mrunf1}')
        
