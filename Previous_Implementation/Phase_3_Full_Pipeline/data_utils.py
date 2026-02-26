import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partitions a dataset among clients using a Dirichlet distribution 
    to simulate Non-IID data (heterogeneous hospital traffic).
    """
    np.random.seed(seed)
    
    # Get labels (assuming binary labels for attack/benign balance)
    if hasattr(dataset, 'y'):
        labels = dataset.y.numpy()
        num_classes = len(np.unique(labels))
    else:
        labels = np.zeros(len(dataset))
        num_classes = 1

    min_size = 0
    N = len(dataset)
    
    # Retry until all clients have data
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance the proportions
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_loaders = []
    for i in range(num_clients):
        subset = Subset(dataset, idx_batch[i])
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
        
    return client_loaders