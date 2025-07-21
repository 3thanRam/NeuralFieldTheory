# training.py

import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import config
from data_handling import save_model, savedata

import json
import matplotlib.pyplot as plt
import os
from config import config

def plot_learning_curves(use_log_scale=True):
    """
    Loads the training history from the JSON file and plots the
    learning curves for all models found in the file.

    Args:
        use_log_scale (bool): If True, the y-axis (Loss) will be on a logarithmic scale.
    """
    history_path = config["history_save_path"]
    plot_path = config["learning_curve_plot_path"]

    if not os.path.exists(history_path):
        print(f"Error: History file not found at {history_path}")
        print("Please run training first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    if not history:
        print("History file is empty. Nothing to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    colors = {'hamiltonian': 'C0', 'transformer': 'C1'}
    
    # --- Plot Training Loss ---
    ax1.set_title('Training Loss vs. Epochs')
    ax1.set_ylabel('Loss' + (' (Log Scale)' if use_log_scale else ''))
    for model_type, data in history.items():
        train_loss = np.array(data['train_loss'])
        epochs = np.arange(1, len(train_loss) + 1)
        if use_log_scale:
            valid_indices = train_loss > 0
            ax1.plot(epochs[valid_indices], train_loss[valid_indices], label=f'{model_type.capitalize()} Train Loss', color=colors.get(model_type))
        else:
            ax1.plot(epochs, train_loss, label=f'{model_type.capitalize()} Train Loss', color=colors.get(model_type))

    ax1.legend()
    if use_log_scale:
        ax1.set_yscale('log')

    # --- Plot Validation Loss ---
    ax2.set_title('Validation Loss vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss' + (' (Log Scale)' if use_log_scale else ''))
    for model_type, data in history.items():
        val_loss = np.array(data['val_loss'])
        epochs = np.arange(1, len(val_loss) + 1)
        if use_log_scale:
            valid_indices = val_loss > 0
            ax2.plot(epochs[valid_indices], val_loss[valid_indices], label=f'{model_type.capitalize()} Val Loss', linestyle='--', color=colors.get(model_type))
        else:
            ax2.plot(epochs, val_loss, label=f'{model_type.capitalize()} Val Loss', linestyle='--', color=colors.get(model_type))

    ax2.legend()
    if use_log_scale:
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Learning curve plot saved to {plot_path}")

def differentiable_direction_loss(preds, targets):
    """Differentiable version of directional accuracy - more efficient"""
    # Use tanh to approximate sign function
    pred_changes = preds[..., 1, 1:] - preds[..., 1, :-1]  # Close price changes
    true_changes = targets[..., 1, 1:] - targets[..., 1, :-1]
    
    # Sigmoid with large multiplier approximates step function
    direction_match = torch.sigmoid(5 * pred_changes * true_changes)  # Reduced multiplier
    return 1 - direction_match.mean()

class FinancialLoss(nn.Module):
    """Memory-efficient financial loss function"""
    def __init__(self, huber_weight=0.7, direction_weight=0.3):
        super().__init__()
        self.huber_weight = huber_weight
        self.direction_weight = direction_weight

    def forward(self, preds, targets):
        # 1. Magnitude Loss (using Huber Loss for robustness)
        magnitude_loss = F.smooth_l1_loss(preds, targets)
        
        # 2. Directional Loss - simplified computation
        # Compute price changes more efficiently
        pred_close = preds[:, :, 1]  # Close price
        true_close = targets[:, :, 1]
        
        if pred_close.size(1) > 1:
            pred_changes = pred_close[:, 1:] - pred_close[:, :-1]
            true_changes = true_close[:, 1:] - true_close[:, :-1]
            
            # Cosine similarity between change vectors
            pred_changes_norm = F.normalize(pred_changes, dim=1, eps=1e-8)
            true_changes_norm = F.normalize(true_changes, dim=1, eps=1e-8)
            
            cosine_sim = (pred_changes_norm * true_changes_norm).sum(dim=1).mean()
            directional_loss = 1 - cosine_sim
        else:
            directional_loss = 0.0
        
        # 3. Combine the losses
        total_loss = self.huber_weight * magnitude_loss + self.direction_weight * directional_loss
        
        return total_loss


def get_current_weights(epoch, max_epochs):
    """Gradually increase metric-based loss weights"""
    progress = epoch / max_epochs
    return {
        'huber_weight': max(0.5, 1.0 - progress * 0.3),
        'direction_weight': min(0.5, progress * 0.5),
    }

def compute_conservation_loss(Q_i, P_i, Q_f, P_f):
    """Memory-efficient conservation loss computation"""
    # Compute norms more efficiently
    N_initial = (Q_i * Q_i + P_i * P_i).sum(dim=-1).mean()
    N_final = (Q_f * Q_f + P_f * P_f).sum(dim=-1).mean()
    return F.mse_loss(N_final, N_initial)


def get_conservation_weight(epoch, max_epochs, start_epoch=10, max_weight=0.1):
    """
    Anneals the conservation loss weight. Starts at 0 and ramps up to max_weight.
    """
    if epoch < start_epoch:
        return 0.0
    # Linear ramp-up over a fraction of the total epochs
    ramp_duration = max_epochs // 4 
    progress = min(1.0, (epoch - start_epoch) / ramp_duration)
    return progress * max_weight

def training(model, optimizer, start_epoch, all_X, all_Y, model_type):
    device = torch.device(config["device"])
    model.to(device)
    print(f"Training on device: {device}")

    # Memory optimization settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    # --- Load or Initialize Training History ---
    history_path = config["history_save_path"]
    try:
        with open(history_path, 'r') as f:
            full_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        full_history = {}
    
    # Get history for the current model, or initialize it
    model_history = full_history.get(model_type, {"train_loss": [], "val_loss": []})

    # If starting from scratch, clear old history for this model
    if start_epoch == 0:
        model_history = {"train_loss": [], "val_loss": []}
    else:
        model_history["train_loss"] = model_history["train_loss"][:start_epoch]
        model_history["val_loss"] = model_history["val_loss"][:start_epoch]

    print("Splitting prepared dataset into training and validation sets...")
    
    # Shuffle indices to ensure random split
    indices = np.arange(len(all_X))
    np.random.shuffle(indices)
    
    val_split_index = int(len(indices) * (1 - config["VAL_SPLIT_RATIO"]))
    train_indices, val_indices = indices[:val_split_index], indices[val_split_index:]
    X_train, X_val = all_X[train_indices], all_X[val_indices]
    Y_train, Y_val = all_Y[train_indices], all_Y[val_indices]

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Optional: Save the split data for faster re-runs
    savedata(X_train, Y_train, X_val, Y_val, config["data_save_path"])

    # Create PyTorch Datasets and DataLoaders with memory-efficient settings
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                            shuffle=True, num_workers=0, pin_memory=True)
    
    val_loader = None
    if len(X_val) > 0:
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                  torch.tensor(Y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
                              num_workers=0, pin_memory=True)
    
    model_save_path = os.path.join(config["model_save_dir"], f"{model_type}_model.pth")
    print(f"Checkpoints for this run will be saved to: {model_save_path}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    best_val_loss = float('inf')

    # Training loop with memory optimizations
    for epoch in range(start_epoch, config["num_epoch"]):
        model.train()
        total_train_loss = 0
        weights = get_current_weights(epoch, config["num_epoch"])
        criterion = FinancialLoss(**weights)
        
        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(device, non_blocking=True), batch_Y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with gradient accumulation for large models
            if model_type == "hamiltonian":
                predictions, (Q_i, P_i, Q_f, P_f) = model(batch_X, return_internals=True)
                
                # Main prediction loss
                prediction_loss = criterion(predictions, batch_Y)
                                
                # Combine losses
                conservation_weight =0.1
                
                total_loss = prediction_loss + compute_conservation_loss(Q_i, P_i, Q_f, P_f) * conservation_weight

            else:
                predictions = model(batch_X)
                total_loss = criterion(predictions, batch_Y)

            # Backward pass with gradient scaling
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += total_loss.item()
            
            # Memory cleanup every few batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop with memory efficiency
        avg_val_loss = float('nan')
        if val_loader:
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_X_val, batch_Y_val in val_loader:
                    batch_X_val = batch_X_val.to(device, non_blocking=True)
                    batch_Y_val = batch_Y_val.to(device, non_blocking=True)
                    
                    if model_type == "hamiltonian":
                        # Use no_grad context for validation - no need for internals
                        with torch.enable_grad():
                            predictions = model(batch_X_val, return_internals=False)
                        v_loss = criterion(predictions, batch_Y_val)
                    else:
                        predictions = model(batch_X_val)
                        v_loss = criterion(predictions, batch_Y_val)
                    
                    total_val_loss += v_loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"*** New best validation loss: {best_val_loss:.6f}. Saving model... ***")
                save_model(model, optimizer, epoch + 1, model_save_path)
        
        # --- Record and Save History ---
        model_history["train_loss"].append(avg_train_loss)
        model_history["val_loss"].append(avg_val_loss)
        full_history[model_type] = model_history
        
        # Save history every 5 epochs to reduce I/O
        if epoch % 5 == 0:
            with open(history_path, 'w') as f:
                json.dump(full_history, f, indent=4)

        print(f"Epoch [{epoch+1}/{config['num_epoch']}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final history save
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=4)
    
    print(f"Training finished. History saved to {history_path}")