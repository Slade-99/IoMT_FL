import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

class NIDSDataset(Dataset):

    def __init__(self, dataframe):

        self.X = dataframe.iloc[:, :-1].values

        self.y = dataframe.iloc[:, -1].values


        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.int64)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

def get_dataloaders(base_path, batch_size):

    

    paths = {
        'train': os.path.join(base_path, 'train_benign.csv'),
        'val': os.path.join(base_path, 'val_benign.csv'),
        'test_balanced': os.path.join(base_path, 'test_balanced.csv'),
        'test_all': os.path.join(base_path, 'test_all.csv') 
    }
    

    try:
        dataframes = {
            name: pd.read_csv(path) for name, path in paths.items()
        }
    except FileNotFoundError as e:
        print(f"Error: Could not find data file at {e.filename}")
        print("Please check your DATA_PATH.")
        return None, None


    datasets = {
        name: NIDSDataset(df) for name, df in dataframes.items()
    }
    

    loaders = {
        'train': DataLoader(
            datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        ),
        'val': DataLoader(
            datasets['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        ),
        'test_balanced': DataLoader(
            datasets['test_balanced'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        ),
        'test_all': DataLoader(
            datasets['test_all'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
    }


    input_dim = datasets['train'].X.shape[1]

    return loaders, input_dim


if __name__ == "__main__":
    print("--- Testing dataloader.py ---")
    
    
    DATA_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment"
    BATCH_SIZE = 512

    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data path not found: {DATA_PATH}")
    else:
        loaders, input_dim = get_dataloaders(base_path=DATA_PATH, batch_size=BATCH_SIZE)
        
        if loaders:
            print(f"\nSuccessfully created DataLoaders.")
            print(f"Input feature dimension: {input_dim}")
            

            print("\nTesting 'train' loader...")
            x_train, y_train = next(iter(loaders['train']))
            print(f"  Batch X shape: {x_train.shape}")
            print(f"  Batch y shape: {y_train.shape}")
            
            print("\nTesting 'val' loader...")
            x_val, y_val = next(iter(loaders['val']))
            print(f"  Batch X shape: {x_val.shape}")
            print(f"  Batch y shape: {y_val.shape}")

            print("\nTesting 'test_balanced' loader...")
            x_test, y_test = next(iter(loaders['test_balanced']))
            print(f"  Batch X shape: {x_test.shape}")
            print(f"  Batch y shape: {y_test.shape}")

            print("\n--- Test Complete ---")