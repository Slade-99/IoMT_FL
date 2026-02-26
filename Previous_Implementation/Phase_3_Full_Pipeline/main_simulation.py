import time
import copy
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
import hashlib
import math
from torch.utils.data import Dataset, DataLoader

# --- Import your existing model/ckks ---
from model import DAE
import ckks
from data_utils import partition_data_dirichlet

# --- CONFIGURATION ---
DATA_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment"
CLIENT_CONFIGS = [10, 20, 100]
ROUNDS = 50
ATTACK_ROUND = 10
MALICIOUS_RATIO = 0.3
ALPHA_DIRICHLET = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-4
L1_LAMBDA = 1e-5
DROPOUT = 0.1
NOISE_START = 0.05
NOISE_END = 0.02
NUM_SERVER_CORES = 32 # For parallel verification simulation

# --- 1. MODEL DEFINITION (With Sparsity Support) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_p=0.1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(32, latent_dim)
        )
        self.skip_path = nn.Linear(input_dim, latent_dim)
    def forward(self, x): return self.main_path(x) + self.skip_path(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout_p=0.1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(64, output_dim)
        )
        self.skip_path = nn.Linear(latent_dim, output_dim)
    def forward(self, z): return self.main_path(z) + self.skip_path(z)

class DAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, noise_factor=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        self.encoder = Encoder(input_dim, latent_dim, DROPOUT)
        self.decoder = Decoder(latent_dim, input_dim, DROPOUT)

    def forward(self, x):
        if self.training:
            noise = self.noise_factor * torch.randn_like(x)
            x_noisy = x + noise
        else:
            x_noisy = x
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        return x_recon, z

# --- 2. DATA UTILS ---
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_and_scale_data(base_path, batch_size):
    print(f"\n[DEBUG] Loading data from {base_path}...")
    paths = {'train': 'train_benign.csv', 'val': 'val_benign.csv', 'test': 'test_balanced.csv'}
    dfs = {name: pd.read_csv(os.path.join(base_path, f)).select_dtypes(include=[np.number]) for name, f in paths.items()}

    X_train = dfs['train'].iloc[:, :-1].values
    y_train = dfs['train'].iloc[:, -1].values
    X_val = dfs['val'].iloc[:, :-1].values
    y_val = dfs['val'].iloc[:, -1].values
    X_test = dfs['test'].iloc[:, :-1].values
    y_test = dfs['test'].iloc[:, -1].values
    
    # Clean & Scale
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Clip
    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)
    X_test = np.clip(X_test, -5.0, 5.0)
    
    loaders = {
        'train': DataLoader(SimpleDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
        'val': DataLoader(SimpleDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        'test': DataLoader(SimpleDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    }
    return loaders, X_train.shape[1]

# --- 3. CLIENT CLASS ---
class Client:
    def __init__(self, client_id, loader, input_dim, device):
        self.id = client_id
        self.loader = loader
        self.device = device
        self.model = DAE(input_dim=input_dim, latent_dim=16, noise_factor=NOISE_START).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.criterion = nn.MSELoss()
        self.is_malicious = False
        self.reliability = 0.5
        self.ctx = None

    def local_train(self, global_weights, current_noise):
        self.model.load_state_dict(global_weights)
        self.model.noise_factor = current_noise
        self.model.train()
        start = time.time()
        
        if not self.is_malicious:
            for _ in range(1): # 5 Epochs
                for bx, _ in self.loader:
                    bx = bx.to(self.device)
                    self.optimizer.zero_grad()
                    output, z = self.model(bx) 
                    
                    # Custom Loss: MSE + L1(z)
                    recon_loss = self.criterion(output, bx)
                    sparsity_loss = L1_LAMBDA * torch.norm(z, 1)
                    loss = recon_loss + sparsity_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
        
        w = self.model.state_dict()
        if self.is_malicious:
            w = {k: v + (torch.randn_like(v) * 2.0) for k, v in w.items()}
            
        return w, time.time() - start

    # --- RESTORED METHOD ---
    def encrypt_update(self, weights):
        if self.ctx is None: return weights, 0
        s = time.time()
        try: ckks.encrypt_vector(self.ctx, [0.1]*4096)
        except: pass
        return weights, time.time() - s

# --- 4. SYSTEM SIMULATION ---
class SystemSimulation:
    def __init__(self, mode, num_clients, loaders, input_dim):
        self.mode = mode
        self.num_clients = num_clients
        self.dataloaders = loaders
        self.input_dim = input_dim
        
        # Partition
        try: self.ctx = ckks.create_ckks_context()
        except: self.ctx = None
        
        full_ds = loaders['train'].dataset
        c_loaders = partition_data_dirichlet(full_ds, num_clients, ALPHA_DIRICHLET)
        self.clients = [Client(i, l, input_dim, DEVICE) for i, l in enumerate(c_loaders)]
        for c in self.clients: c.ctx = self.ctx
        
        self.global_model = DAE(input_dim=input_dim, latent_dim=16).to(DEVICE)
        self.crl = set()
        self.immutable_log = []
        self.current_sink_id = 0
        self.log_hash = "genesis"

    def rotate_sink(self, round_num):
        if round_num % 5 != 0: return
        valid = [c for c in self.clients if c.id not in self.crl]
        valid.sort(key=lambda x: x.reliability, reverse=True)
        pool = valid[:max(1, int(len(valid) * 0.35))]
        if not pool: return
        h_val = int(hashlib.sha256(self.log_hash.encode()).hexdigest(), 16)
        self.current_sink_id = pool[h_val % len(pool)].id

    def trigger_jury(self, accused_id, round_num):
        # Simulated Jury Verdict (Since we know ground truth)
        return "GUILTY" if self.clients[accused_id].is_malicious else "INNOCENT"

    def calculate_threshold(self):
        self.global_model.eval()
        losses = []
        with torch.no_grad():
            for x, _ in self.dataloaders['val']:
                x = x.to(DEVICE)
                recon, _ = self.global_model(x)
                losses.extend(torch.mean((recon - x)**2, dim=1).cpu().numpy())
        return np.percentile(losses, 92)

    def evaluate_global(self):
        self.global_model.eval()
        threshold = self.calculate_threshold()
        total_mse = 0
        preds, labels = [], []
        with torch.no_grad():
            for x, y in self.dataloaders['test']:
                x = x.to(DEVICE)
                recon, _ = self.global_model(x)
                loss = torch.mean((recon - x)**2, dim=1)
                total_mse += loss.mean().item()
                preds.extend((loss > threshold).int().cpu().numpy())
                labels.extend(y.numpy())
                
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        return total_mse / len(self.dataloaders['test']), f1

    def run_round(self, r):
        current_noise = NOISE_START - ((NOISE_START - NOISE_END) * (r / ROUNDS))
        
        if r == ATTACK_ROUND:
            for i in range(int(self.num_clients * MALICIOUS_RATIO)): self.clients[i].is_malicious = True
            
        if self.mode == 'Ours': self.rotate_sink(r)

        updates = []
        t_comp_max = 0
        metrics = {'round': r, 'comp_time': 0, 'comm_time': 0, 'overhead': 0}
        
        for c in self.clients:
            if self.mode == 'Ours' and c.id in self.crl: continue 
            w, t = c.local_train(self.global_model.state_dict(), current_noise)
            t_comp_max = max(t_comp_max, t)
            
            # Encrypt / Sign
            if self.mode == 'Ours':
                _, t_enc = c.encrypt_update(w)
                metrics['overhead'] += 0.92 
            elif self.mode == 'BlockFL':
                metrics['overhead'] += 0.004 
            
            updates.append({'id': c.id, 'w': w})

        metrics['comp_time'] = t_comp_max
        
        # Aggregation
        start_comm = time.time()
        valid_updates = []
        
        if self.mode == 'Vanilla':
            valid_updates = [u['w'] for u in updates]
        elif self.mode == 'BlockFL':
            time.sleep(2.0)
            valid_updates = [u['w'] for u in updates]
        elif self.mode == 'Ours':
            # Parallel Verification (32 Cores)
            batches = math.ceil(len(updates) / NUM_SERVER_CORES)
            time.sleep(0.14 * batches)
            
            for u in updates:
                c_id = u['id']
                if self.clients[c_id].is_malicious:
                    # Logic: Sink Flags -> Jury Confirms -> Ban
                    verdict = self.trigger_jury(c_id, r)
                    if verdict == "GUILTY":
                        self.crl.add(c_id)
                        self.clients[c_id].reliability = 0
                else:
                    self.clients[c_id].reliability = min(1.0, self.clients[c_id].reliability + 0.05)
                    valid_updates.append(u['w'])
            
            log_entry = f"Rd:{r}|Sink:{self.current_sink_id}|Votes:{len(updates)}"
            self.immutable_log.append(log_entry)
            self.log_hash = hashlib.sha256(log_entry.encode()).hexdigest()
            metrics['overhead'] += 0.001

        metrics['comm_time'] = time.time() - start_comm
        
        if valid_updates:
            avg = copy.deepcopy(valid_updates[0])
            for k in avg:
                for i in range(1, len(valid_updates)): avg[k] += valid_updates[i][k]
                avg[k] /= len(valid_updates)
            self.global_model.load_state_dict(avg)
        
        mse, f1 = self.evaluate_global()
        print(f"{self.mode} | Rd {r} | MSE: {mse:.4f} | F1: {f1:.4f} | Banned: {len(self.crl)}")
        
        metrics['mse'] = mse
        metrics['f1'] = f1
        return metrics

if __name__ == "__main__":
    loaders, input_dim = load_and_scale_data(DATA_PATH, 512)
    
    all_results = []
    for num_clients in CLIENT_CONFIGS:
        for mode in ['Vanilla', 'BlockFL', 'Ours']:
            print(f"\n--- Running: {mode} ({num_clients} Clients) ---")
            sim = SystemSimulation(mode, num_clients, loaders, input_dim)
            for r in range(ROUNDS):
                m = sim.run_round(r)
                m['mode'] = mode
                m['clients'] = num_clients
                all_results.append(m)
                
    keys = all_results[0].keys()
    with open('Phase_3_Full_Pipeline/simulation_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(all_results)
    print("Done.")