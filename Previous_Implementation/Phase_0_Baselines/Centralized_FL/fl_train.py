import torch
import torch.nn as nn
import copy
import os
import numpy as np
import pandas as pd
import joblib
import json
import time
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score

# --- Import your existing modules ---
from Models.model import DAE
from Preprocessing.ph0_th_loaders.data import NIDSDataset
from Phase_0_Baselines.Thresholding import engine

# --- 1. Configuration ---
# Paths
DATA_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment"
ENCODER_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/after_scaling_encoding"
OUTPUT_DIR = "/home/azwad/Works/IoMT_FL/Results/FL_Centralized_FedAvg"
os.makedirs(OUTPUT_DIR, exist_ok=True)


PREVIOUS_OPTIMAL_THRESHOLD = 0.0033788118


NUM_CLIENTS = 5
GLOBAL_ROUNDS = 999
CLIENT_EPOCHS_PER_ROUND = 1
BATCH_SIZE = 1024


LR = 1e-4
WEIGHT_DECAY = 1e-4
LATENT_DIM = 16
DROPOUT_P = 0.1
L1_LAMBDA = 1e-5
INITIAL_NOISE = 0.05
FINAL_NOISE = 0.02
PATIENCE = 10

# --- 2. Setup & Logging ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = engine.setup_logging(log_file=os.path.join(OUTPUT_DIR, 'fl_train.log'))
logger.info(f"Starting FL Simulation (FedAvg) on {DEVICE}")
logger.info(f"Clients: {NUM_CLIENTS}, Rounds: {GLOBAL_ROUNDS}")
logger.info(f"Using Fixed Threshold from Baseline: {PREVIOUS_OPTIMAL_THRESHOLD}")

# --- 3. Data Partitioning ---
def get_client_loaders(data_path, num_clients, batch_size):
    train_path = os.path.join(data_path, "train_benign.csv")
    val_path = os.path.join(data_path, "val_benign.csv") # For calculating val loss
    # We need a mixed validation set to calculate F1/AUROC during training rounds
    # We'll use the balanced test set for this 'monitoring' purpose strictly
    monitor_path = os.path.join(data_path, "test_balanced.csv") 
    
    logger.info("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    monitor_df = pd.read_csv(monitor_path)
    
    # Shuffle and split training data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    client_dfs = np.array_split(train_df, num_clients)
    
    client_loaders = []
    for i, df in enumerate(client_dfs):
        ds = NIDSDataset(df)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        client_loaders.append(loader)

    # Global loaders
    val_loader = torch.utils.data.DataLoader(
        NIDSDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    # Monitoring loader (for F1/AUROC per round)
    monitor_loader = torch.utils.data.DataLoader(
        NIDSDataset(monitor_df), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    input_dim = client_loaders[0].dataset.X.shape[1]
    return client_loaders, val_loader, monitor_loader, input_dim

client_loaders, global_val_loader, global_monitor_loader, input_dim = get_client_loaders(
    DATA_PATH, NUM_CLIENTS, BATCH_SIZE
)

# --- 4. Load Label Encoder (Need Benign Label for Metrics) ---
try:
    label_encoder = joblib.load(os.path.join(ENCODER_PATH, 'label_encoder.joblib'))
    try:
        benign_label = int(label_encoder.transform(['Benign'])[0])
    except:
        benign_label = int(label_encoder.transform(['benign'])[0])
except Exception as e:
    logger.error(f"Could not load label encoder: {e}")
    exit()

# --- 5. Helper Functions ---

def fed_avg(weights_list, sample_counts):
    total_samples = sum(sample_counts)
    global_weights = copy.deepcopy(weights_list[0])
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        
    for weights, count in zip(weights_list, sample_counts):
        weight_factor = count / total_samples
        for key in weights.keys():
            global_weights[key] += weights[key] * weight_factor
    return global_weights

def calculate_communication_cost(model):
    """Estimates model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def evaluate_global_metrics(model, dataloader, threshold, benign_label):
    """Quick evaluation for F1 and AUROC during training rounds."""
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(DEVICE)
            x_recon = model(features)
            errors = criterion(x_recon, features).mean(dim=1).cpu().numpy()
            
            preds = (errors > threshold).astype(int)
            # Convert labels to binary (0=Benign, 1=Attack)
            binary_labels = (labels.cpu().numpy() != benign_label).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(binary_labels)
            all_scores.extend(errors)
            
    f1 = f1_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_scores)
    return f1, auroc

# --- 6. Initialize Global Model ---
global_model = DAE(
    input_dim=input_dim, 
    latent_dim=LATENT_DIM, 
    noise_factor=INITIAL_NOISE, 
    dropout_p=DROPOUT_P
).to(DEVICE)

# Calculate Communication overhead
model_size_mb = calculate_communication_cost(global_model)
logger.info(f"Global Model Size: {model_size_mb:.4f} MB")
# Cost per round = (Broadcast + Upload) * Num_Clients
comm_cost_per_round = model_size_mb * 2 * NUM_CLIENTS 

best_model_path = os.path.join(OUTPUT_DIR, 'global_best_model.pth')
early_stopper = engine.EarlyStopping(
    patience=PATIENCE, verbose=True, path=best_model_path, logger=logger
)
criterion = nn.MSELoss()

# History tracker with FL metrics
history = {
    'round': [], 
    'train_loss': [], 
    'val_loss': [], 
    'val_f1': [], 
    'val_auroc': [],
    'cumulative_comm_mb': []
}

# --- 7. FL Training Loop ---
logger.info("\n--- Starting Federated Training ---")
start_time = time.time()

for round_idx in range(GLOBAL_ROUNDS):
    round_start = time.time()
    current_noise = engine.get_noise_factor(INITIAL_NOISE, FINAL_NOISE, round_idx, GLOBAL_ROUNDS)
    
    local_weights = []
    local_losses = []
    client_sample_counts = []
    
    # --- Client Phase ---
    for client_idx, loader in enumerate(client_loaders):
        local_model = copy.deepcopy(global_model)
        local_model.train()
        local_model.noise_factor = current_noise
        
        optimizer = AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        client_loss_accum = 0.0
        for epoch in range(CLIENT_EPOCHS_PER_ROUND):
            r_loss, l1_loss = engine.train_step(
                local_model, loader, criterion, optimizer, DEVICE, L1_LAMBDA, current_noise
            )
            client_loss_accum += (r_loss + l1_loss)
        
        avg_client_loss = client_loss_accum / CLIENT_EPOCHS_PER_ROUND
        local_losses.append(avg_client_loss)
        local_weights.append(copy.deepcopy(local_model.state_dict()))
        client_sample_counts.append(len(loader.dataset))
    
    # --- Server Phase ---
    # Aggregation
    avg_train_loss = sum(local_losses) / len(local_losses)
    new_global_weights = fed_avg(local_weights, client_sample_counts)
    global_model.load_state_dict(new_global_weights)
    
    # Validation (Recon Loss on Benign)
    val_recon_loss = engine.val_step(global_model, global_val_loader, criterion, DEVICE)
    
    # Metrics (F1/AUROC on Mixed Set using Fixed Threshold)
    val_f1, val_auroc = evaluate_global_metrics(
        global_model, global_monitor_loader, PREVIOUS_OPTIMAL_THRESHOLD, benign_label
    )
    
    # Track History
    total_comm = comm_cost_per_round * (round_idx + 1)
    history['round'].append(round_idx + 1)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_recon_loss)
    history['val_f1'].append(val_f1)
    history['val_auroc'].append(val_auroc)
    history['cumulative_comm_mb'].append(total_comm)
    
    logger.info(f"Round {round_idx+1}/{GLOBAL_ROUNDS} | "
                f"Loss(Val): {val_recon_loss:.6f} | "
                f"F1: {val_f1:.4f} | AUROC: {val_auroc:.4f} | "
                f"Comm: {total_comm:.2f} MB")
    
    # Checkpointing (Based on Benign Reconstruction Loss - standard for DAE)
    early_stopper(val_recon_loss, global_model)
    if early_stopper.early_stop:
        logger.info(f"Early stopping triggered at round {round_idx+1}")
        break

total_training_time = time.time() - start_time
logger.info(f"--- FL Training Complete in {total_training_time:.2f} seconds ---")

# Save History
history['total_training_time_sec'] = total_training_time
engine.save_history(history, os.path.join(OUTPUT_DIR, 'fl_history.json'))

# --- 8. Final Evaluation ---
logger.info("\n--- Starting Final Evaluation (Test Set) ---")

# Load best global model
global_model = engine.load_checkpoint(global_model, best_model_path, DEVICE)

# Reuse the global_monitor_loader (which is test_balanced) for final detailed metrics
results = engine.test_model(
    global_model, 
    global_monitor_loader, 
    DEVICE, 
    PREVIOUS_OPTIMAL_THRESHOLD, # Use fixed threshold
    benign_label, 
    label_encoder.classes_
)

# Add FL Specific Metrics to Results
results['Total_Communication_MB'] = history['cumulative_comm_mb'][-1]
results['Total_Training_Time_Sec'] = total_training_time
results['Rounds_Completed'] = len(history['round'])

# Log Results
logger.info("\n--- FL GLOBAL TEST RESULTS ---")
main_metrics = {k: v for k, v in results.items() if k != 'Per_Attack_Recall'}
for metric, value in main_metrics.items():
    logger.info(f"{metric:<25}: {value}")

logger.info("\n--- PER-ATTACK-CLASS RECALL ---")
per_class_recall = results['Per_Attack_Recall']
for class_name, recall in per_class_recall.items():
    logger.info(f"{class_name:<25}: {recall}")

# Save Results JSON
with open(os.path.join(OUTPUT_DIR, 'fl_test_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

logger.info(f"Results saved to {OUTPUT_DIR}")