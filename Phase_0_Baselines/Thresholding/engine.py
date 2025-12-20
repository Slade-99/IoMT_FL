import torch
import torch.nn as nn
import numpy as np
import os
import logging
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

# --- Logging and Utilities ---

def setup_logging(log_file='training.log'):
    """Sets up a logger that writes to a file."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    return logger

class EarlyStopping:
    """Implements early stopping with patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth', logger=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.logger = logger or logging.getLogger()
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_history(history, path='training_history.json'):
    """Saves the training history dictionary to a JSON file."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)

def load_checkpoint(model, path='best_model.pth', device='cpu'):
    """Loads a model checkpoint from a file."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_noise_factor(initial_noise, final_noise, epoch, total_epochs):
    """Anneals noise factor from initial to final (cosine schedule)."""
    
    cos_val = np.cos(np.pi * epoch / total_epochs)
    return final_noise + 0.5 * (initial_noise - final_noise) * (1 + cos_val)

def train_step(model, dataloader, criterion, optimizer, device, l1_lambda, current_noise_factor):
    """Performs a single training step (epoch)."""
    model.train()
    model.noise_factor = current_noise_factor # Set the noise factor
    
    total_recon_loss = 0.0
    total_l1_loss = 0.0
    
    for (features, _) in dataloader:
        features = features.to(device)
        
        optimizer.zero_grad()
        
        # --- DAE Forward Pass with L1 ---
        # 1. Add noise
        if model.training:
            noise = model.noise_factor * torch.randn_like(features)
            x_noisy = features + noise
        else:
            x_noisy = features # Should not happen in train_step, but safe
            
        # 2. Encode to get latent vector z
        z = model.encoder(x_noisy)
        
        # 3. Decode to get reconstruction
        x_recon = model.decoder(z)
        
        # --- Calculate Losses ---
        # 1. Reconstruction Loss (MSE)
        recon_loss = criterion(x_recon, features)
        
        # 2. L1 Sparsity Loss on latent vector
        l1_loss = l1_lambda * z.abs().mean()
        
        # 3. Total Loss
        loss = recon_loss + l1_loss
        
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_l1_loss += l1_loss.item()
        
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    return avg_recon_loss, avg_l1_loss

def val_step(model, dataloader, criterion, device):
    """Performs a single validation step (epoch)."""
    model.eval()
    total_recon_loss = 0.0
    
    with torch.no_grad():
        for (features, _) in dataloader:
            features = features.to(device)
            
            # Forward pass (no noise, no L1)
            x_recon = model(features)
            
            # Reconstruction Loss
            recon_loss = criterion(x_recon, features)
            total_recon_loss += recon_loss.item()
            
    avg_recon_loss = total_recon_loss / len(dataloader)
    return avg_recon_loss

# --- Thresholding and Testing Logic ---

def find_threshold(model, dataloader, device, percentile=95):
    """
    Finds the reconstruction error threshold from the validation set.
    """
    model.eval()
    # Use MSELoss with 'none' to get per-sample errors
    criterion = nn.MSELoss(reduction='none')
    all_errors = []
    
    with torch.no_grad():
        for (features, _) in dataloader:
            features = features.to(device)
            x_recon = model(features)
            
            # Calculate per-sample error
            errors = criterion(x_recon, features).mean(dim=1)
            all_errors.append(errors.cpu().numpy())
            
    all_errors = np.concatenate(all_errors)
    
    # Set threshold at the nth percentile
    threshold = np.percentile(all_errors, percentile)
    return threshold

def test_model(model, dataloader, device, threshold, benign_label, label_encoder_classes):
    """
    Tests the model on the balanced test set and computes all metrics.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    all_y_true = []
    all_y_true_binary = []
    all_y_pred_binary = []
    all_y_scores = []

    with torch.no_grad():
        for (features, labels) in dataloader:
            features = features.to(device)
            
            x_recon = model(features)
            
            # Per-sample reconstruction error (our anomaly score)
            errors = criterion(x_recon, features).mean(dim=1).cpu().numpy()
            
            # Predicted labels (1 = Attack, 0 = Benign)
            preds = (errors > threshold).astype(int)
            
            # True labels (binary)
            labels_np = labels.cpu().numpy()
            true_binary = (labels_np != benign_label).astype(int)
            
            all_y_true.extend(labels_np)
            all_y_true_binary.extend(true_binary)
            all_y_pred_binary.extend(preds)
            all_y_scores.extend(errors)

    # --- Calculate All Metrics ---
    results = {}
    
    y_true_b = np.array(all_y_true_binary)
    y_pred_b = np.array(all_y_pred_binary)
    y_scores = np.array(all_y_scores)
    y_true_multi = np.array(all_y_true)
    
    results['AUROC'] = roc_auc_score(y_true_b, y_scores)
    results['AUPRC'] = average_precision_score(y_true_b, y_scores)
    results['F1_Score'] = f1_score(y_true_b, y_pred_b)
    results['Precision'] = precision_score(y_true_b, y_pred_b)
    results['Recall'] = recall_score(y_true_b, y_pred_b)
    results['Accuracy'] = accuracy_score(y_true_b, y_pred_b)
    results['MCC'] = matthews_corrcoef(y_true_b, y_pred_b)
    results['Balanced_Accuracy'] = balanced_accuracy_score(y_true_b, y_pred_b)
    
    # TPR at 1% FPR
    fpr, tpr, _ = roc_curve(y_true_b, y_scores)
    results['TPR_at_1_FPR'] = np.interp(0.01, fpr, tpr)
    
    # FPR at 95% TPR
    # We need to sort by TPR to use interp
    sort_idx = np.argsort(tpr)
    tpr_sorted = tpr[sort_idx]
    fpr_sorted = fpr[sort_idx]
    results['FPR_at_95_TPR'] = np.interp(0.95, tpr_sorted, fpr_sorted)
    
    # Per-attack class recall
    per_class_recall = {}
    attack_labels = [i for i, cls in enumerate(label_encoder_classes) if cls not in ['Benign', 'benign']]
    
    for label_int in attack_labels:
        class_name = label_encoder_classes[label_int]
        
        # Get mask for samples of this specific attack class
        class_mask = (y_true_multi == label_int)
        
        if np.sum(class_mask) == 0:
            per_class_recall[class_name] = "N/A (No samples in test set)"
            continue
            
        class_y_true = y_true_b[class_mask]
        class_y_pred = y_pred_b[class_mask]
        
        # Calculate recall for this class (all should be '1')
        class_recall = recall_score(class_y_true, class_y_pred, zero_division=0)
        per_class_recall[class_name] = class_recall
        
    results['Per_Attack_Recall'] = per_class_recall

    return results