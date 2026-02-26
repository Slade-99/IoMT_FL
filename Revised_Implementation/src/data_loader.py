import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import RobustScaler

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_and_scale_data(base_path, batch_size):
    """Loads CSVs, handles NaNs/Infs, scales using RobustScaler, and returns DataLoaders."""
    paths = {'train': 'train_benign.csv', 'val': 'val_benign.csv', 'test': 'test_balanced.csv'}
    dfs = {name: pd.read_csv(os.path.join(base_path, f)).select_dtypes(include=[np.number]) for name, f in paths.items()}

    X_train, y_train = dfs['train'].iloc[:, :-1].values, dfs['train'].iloc[:, -1].values
    X_val, y_val = dfs['val'].iloc[:, :-1].values, dfs['val'].iloc[:, -1].values
    X_test, y_test = dfs['test'].iloc[:, :-1].values, dfs['test'].iloc[:, -1].values
    
    # Clean Data
    for X in [X_train, X_val, X_test]:
        np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)
    
    # Scale using RobustScaler (resilient to outliers)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Clip to prevent extreme gradient explosions
    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)
    X_test = np.clip(X_test, -5.0, 5.0)
    
    loaders = {
        'train': DataLoader(SimpleDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        'val': DataLoader(SimpleDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        'test': DataLoader(SimpleDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    }
    return loaders, X_train.shape[1]

def partition_data_dirichlet(dataset, num_clients, alpha=0.5):
    """Partitions data across clients using a Dirichlet distribution for non-IID realistic setups."""
    min_size = 0
    num_classes = len(np.unique(dataset.y.numpy()))
    N = len(dataset)
    
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(dataset.y.numpy() == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    loaders = []
    for indices in idx_batch:
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=512, shuffle=True))
    return loaders