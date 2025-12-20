import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np
from tqdm import tqdm
from Models.model import DAE
from Preprocessing.ph0_th_loaders.data import get_dataloaders
from Phase_0_Baselines.Thresholding import engine


DATA_PATH = r"/home/azwad/Works/IoMT_FL/Dataset/for_thresholding_experiment"
OUTPUT_DIR = "/home/azwad/Works/IoMT_FL/Results/Thresholding"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 15       # For early stopping
LATENT_DIM = 16
DROPOUT_P = 0.1
L1_LAMBDA = 1e-5    # Sparsity penalty
INITIAL_NOISE = 0.05
FINAL_NOISE = 0.02

# --- 2. Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = engine.setup_logging(log_file=os.path.join(OUTPUT_DIR, 'train.log'))
logger.info(f"Using device: {DEVICE}")
logger.info(f"Starting training with BATCH_SIZE={BATCH_SIZE}, LR={LR}, MAX_EPOCHS={MAX_EPOCHS}")

# --- 3. Load Data ---
logger.info("Loading dataloaders...")
loaders, input_dim = get_dataloaders(base_path=DATA_PATH, batch_size=BATCH_SIZE)
if loaders is None:
    logger.error("Failed to load dataloaders. Exiting.")
    exit()
    
logger.info(f"Input dimension set to: {input_dim}")
logger.info(f"Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}")


model = DAE(
    input_dim=input_dim, 
    latent_dim=LATENT_DIM, 
    noise_factor=INITIAL_NOISE, # Initial value
    dropout_p=DROPOUT_P
).to(DEVICE)

# Use MSELoss for reconstruction
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01)

# --- 5. Initialize Training Utilities ---
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
early_stopper = engine.EarlyStopping(
    patience=PATIENCE, 
    verbose=True, 
    path=best_model_path,
    logger=logger
)
history = {'train_recon_loss': [], 'train_l1_loss': [], 'val_recon_loss': []}

# --- 6. Main Training Loop ---
logger.info("--- Starting Main Training Loop ---")
for epoch in range(MAX_EPOCHS):
    
    # Get current annealed noise factor
    current_noise_factor = engine.get_noise_factor(
        INITIAL_NOISE, FINAL_NOISE, epoch, MAX_EPOCHS
    )
    
    # Train
    train_recon_loss, train_l1_loss = engine.train_step(
        model, loaders['train'], criterion, optimizer, DEVICE, L1_LAMBDA, current_noise_factor
    )
    
    # Validate
    val_recon_loss = engine.val_step(
        model, loaders['val'], criterion, DEVICE
    )
    
    # Log and Save
    history['train_recon_loss'].append(train_recon_loss)
    history['train_l1_loss'].append(train_l1_loss)
    history['val_recon_loss'].append(val_recon_loss)
    
    logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} | "
                f"Noise: {current_noise_factor:.4f} | "
                f"Train Recon Loss: {train_recon_loss:.6f} | "
                f"Train L1 Loss: {train_l1_loss:.6f} | "
                f"Val Recon Loss: {val_recon_loss:.6f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Check for early stopping
    early_stopper(val_recon_loss, model)
    if early_stopper.early_stop:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

logger.info("--- Training Complete ---")

# --- 7. Save History and Find Threshold ---
engine.save_history(history, os.path.join(OUTPUT_DIR, 'training_history.json'))
logger.info(f"Training history saved.")

# Load the best model
logger.info(f"Loading best model from {best_model_path} for thresholding...")
model = engine.load_checkpoint(model, best_model_path, DEVICE)

# Find and save threshold
threshold = engine.find_threshold(model, loaders['val'], DEVICE, percentile=92)
threshold_path = os.path.join(OUTPUT_DIR, 'threshold.txt')
with open(threshold_path, 'w') as f:
    f.write(str(threshold))
    
logger.info(f"Threshold (99th percentile) calculated: {threshold}")
logger.info(f"Threshold saved to {threshold_path}")

print(f"Training complete. Best model and logs saved to {OUTPUT_DIR}")