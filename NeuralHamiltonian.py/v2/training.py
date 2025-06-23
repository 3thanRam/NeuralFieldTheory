# training.py

import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import config
from data_handling import save_model, savedata, gen_data_for_model

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
        # Filter out any non-positive values if using log scale, as log(x) is undefined for x<=0
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
    # Add a suffix to the filename if using log scale to avoid overwriting
    #if use_log_scale:
    #    base, ext = os.path.splitext(plot_path)
    #    plot_path = f"{base}_log{ext}"
        
    plt.savefig(plot_path)
    print(f"Learning curve plot saved to {plot_path}")
    #plt.show()

    

def differentiable_direction_loss(preds, targets):
    """Differentiable version of directional accuracy"""
    # Use tanh to approximate sign function
    pred_changes = preds[..., 1, 1:] - preds[..., 1, :-1]  # Close price changes
    true_changes = targets[..., 1, 1:] - targets[..., 1, :-1]
    
    # Sigmoid with large multiplier (10) approximates step function
    direction_match = torch.sigmoid(10 * pred_changes * true_changes)
    return 1 - direction_match.mean()  # Minimize this

class FinancialLoss(nn.Module):
    def __init__(self, mse_weight=1.0, direction_weight=0.3, volatility_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        
    def forward(self, preds, targets):
        # Standard MSE - now using correct F import
        mse_loss = F.mse_loss(preds, targets)
        
        # Directional loss
        pred_changes = preds[:, 1:, 1] - preds[:, :-1, 1]  # Close price changes
        true_changes = targets[:, 1:, 1] - targets[:, :-1, 1]
        direction_match = torch.sigmoid(10 * pred_changes * true_changes)
        dir_loss = 1 - direction_match.mean()
        
        # Volatility matching loss
        pred_vol = preds[:, :, 1].std(dim=1)  # Close price volatility
        true_vol = targets[:, :, 1].std(dim=1)
        vol_loss = F.mse_loss(pred_vol, true_vol)
        
        return (self.mse_weight * mse_loss + 
                self.direction_weight * dir_loss +
                self.volatility_weight * vol_loss)

def get_training_data_loader():
    all_X, all_Y = [], []
    print(f"Generating {config['num_training_samples']} stock data samples...")
    generated_samples = 0
    attempts = 0
    max_attempts = config["num_training_samples"] * 5 # Allow for failures

    while generated_samples < config["num_training_samples"] and attempts < max_attempts:
        attempts += 1
        X, Y = gen_data_for_model()
        if X is not None and Y is not None:
            all_X.append(X)
            all_Y.append(Y)
            generated_samples += 1
        
        if attempts % 100 == 0:
            print(f"Attempt {attempts}/{max_attempts}, Generated {generated_samples}/{config['num_training_samples']} samples...")

    if generated_samples == 0:
        raise RuntimeError("Failed to generate ANY valid training samples. Check Alpaca API connection and gen_data_for_model logic.")

    X_np = np.array(all_X) # Shape: (num_samples, seq_len, input_dim)
    Y_np = np.array(all_Y) # Shape: (num_samples, seq_len, output_dim)

    # Shuffle and split data
    indices = np.arange(len(X_np))
    np.random.shuffle(indices)
    X_np, Y_np = X_np[indices], Y_np[indices]

    val_split_index = int(len(X_np) * (1 - config["VAL_SPLIT_RATIO"]))
    X_train, X_val = X_np[:val_split_index], X_np[val_split_index:]
    Y_train, Y_val = Y_np[:val_split_index], Y_np[val_split_index:]

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Save the processed data
    savedata(X_train, Y_train, X_val, Y_val, config["data_save_path"])
    
    return (X_train, Y_train), (X_val, Y_val)

def get_current_weights(epoch, max_epochs):
    """Gradually increase metric-based loss weights"""
    progress = epoch / max_epochs
    return {
        'mse_weight': max(0.7, 1.0 - progress * 0.5),  # Reduce MSE over time
        'direction_weight': min(0.5, progress * 0.7),  # Increase direction focus
        'volatility_weight': min(0.3, progress * 0.4)
    }

def training(model, optimizer, start_epoch, all_X, all_Y, model_type):
    device = torch.device(config["device"])
    model.to(device)
    #max_grad_norm = config.get("max_grad_norm", 1.0)
    print(f"Training on device: {device}")

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
    else: # If continuing, truncate history to the start_epoch to avoid duplicates
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

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = None
    if len(X_val) > 0:
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    model_save_path = os.path.join(config["model_save_dir"], f"{model_type}_model.pth")
    print(f"Checkpoints for this run will be saved to: {model_save_path}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config["num_epoch"]):
        model.train()
        total_train_loss = 0
        weights = get_current_weights(epoch, config["num_epoch"])
        criterion = FinancialLoss(**weights)
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        avg_val_loss = float('nan')
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_X_val, batch_Y_val in val_loader:
                    batch_X_val, batch_Y_val = batch_X_val.to(device), batch_Y_val.to(device)
                    with torch.enable_grad():
                        val_preds = model(batch_X_val)
                    v_loss = criterion(val_preds, batch_Y_val)
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
        with open(history_path, 'w') as f:
            json.dump(full_history, f, indent=4)

        print(f"Epoch [{epoch+1}/{config['num_epoch']}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
    print(f"Training finished. History saved to {history_path}")



