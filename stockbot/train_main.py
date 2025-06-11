# train_main.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import the global config dictionary
from config import config

# Project-specific imports
from nft_network import NFTnetworkBlock
from data_utils import get_training_data


import subprocess
import cProfile
import pstats

# --- Add Model Save and Load Functions ---
def save_model(model, optimizer, epoch, save_path):
    """Saves the model state, optimizer state, and current epoch."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # It's good practice to save more than just the model state_dict
    # for resuming training or full model understanding.
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You can add other things here like loss, config used for this model, etc.
        'config_model_params': { # Save model instantiation params from config
            "embed_dim": config["model"]["embed_dim"],
            "max_order_stats": config["model"]["max_order_stats"],
            "num_configs": config["model"]["num_configs"]
        }
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")

def load_model(model_architecture_class, save_path, device):
    """
    Loads a model checkpoint.
    Returns the model, optimizer (if saved), and epoch.
    The model architecture must be defined before calling this.
    """
    if not os.path.exists(save_path):
        print(f"No checkpoint found at {save_path}. Starting from scratch.")
        return None, None, 0 # Model, Optimizer, Epoch

    checkpoint = torch.load(save_path, map_location=device) # map_location handles loading to current device
    
    # Re-create model instance with saved parameters
    # This ensures model architecture matches the saved state_dict
    model_params = checkpoint.get('config_model_params')
    if not model_params:
        # Fallback if old checkpoint without these params (less robust)
        print("Warning: Checkpoint does not contain model instantiation parameters. Using current config.")
        model_params = {
            "embed_dim": config["model"]["embed_dim"],
            "max_order_stats": config["model"]["max_order_stats"],
            "num_configs": config["model"]["num_configs"]
        }

    model = model_architecture_class(
        embed_dim=model_params["embed_dim"],
        max_order=model_params["max_order_stats"],
        num_configs=model_params["num_configs"]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer can also be loaded if you plan to resume training
    optimizer_state_dict = checkpoint.get('optimizer_state_dict') # Use .get for graceful missing key
    
    epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
    
    print(f"Model checkpoint loaded from {save_path}. Resuming from epoch {epoch}.")
    return model, optimizer_state_dict, epoch
# --- End of Model Save and Load Functions ---


def plot_predictions(model, device, data_loader_or_sample, psymbol_name, 
                     feature_index_to_plot, num_samples_to_plot):
    # ... (plot_predictions function remains the same)
    model.eval()
    feature_names = ["Timestamp", "Open", "Close", "High", "Low", "Volume"]
    if feature_index_to_plot < 0 or feature_index_to_plot >= len(feature_names):
        print(f"Invalid feature_index_to_plot: {feature_index_to_plot}. Defaulting to 2 (Close).")
        feature_index_to_plot = 2
    
    plotted_count = 0
    is_data_loader = hasattr(data_loader_or_sample, '__iter__') and hasattr(data_loader_or_sample, 'dataset')

    with torch.no_grad():
        if is_data_loader:
            for batch_idx, (batch_x_enc, batch_x_dec_in, batch_y_tgt) in enumerate(data_loader_or_sample):
                if plotted_count >= num_samples_to_plot: break
                batch_x_dec_in, batch_y_tgt = batch_x_dec_in.to(device), batch_y_tgt.to(device)
                predictions = model(batch_x_dec_in, mask=None, return_diagnostics=False, sampling_mode="expectation")
                for i in range(batch_x_dec_in.size(0)):
                    if plotted_count >= num_samples_to_plot: break
                    actual = batch_y_tgt[i, :, feature_index_to_plot].cpu().numpy()
                    predicted = predictions[i, :, feature_index_to_plot].cpu().numpy()
                    timestamps_mdates = batch_y_tgt[i, :, 0].cpu().numpy()
                    timestamps_datetime = [mdates.num2date(ts) for ts in timestamps_mdates]
                    plt.figure(figsize=(12, 6))
                    plt.plot(timestamps_datetime, actual, label=f'Actual {feature_names[feature_index_to_plot]}', marker='.')
                    plt.plot(timestamps_datetime, predicted, label=f'Predicted {feature_names[feature_index_to_plot]}', linestyle='--')
                    plt.title(f'{psymbol_name} - {feature_names[feature_index_to_plot]} Prediction vs Actual (Sample {plotted_count+1})')
                    plt.xlabel('Time'); plt.ylabel(f'{feature_names[feature_index_to_plot]} Value')
                    plt.legend(); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    plt.xticks(rotation=45); plt.tight_layout(); plt.show()
                    plotted_count += 1
        else: 
            X_dec_in_sample, Y_tgt_sample = data_loader_or_sample
            if X_dec_in_sample.ndim == 2:
                X_dec_in_sample = X_dec_in_sample.unsqueeze(0)
                Y_tgt_sample = Y_tgt_sample.unsqueeze(0)
            X_dec_in_sample, Y_tgt_sample = X_dec_in_sample.to(device), Y_tgt_sample.to(device)
            predictions = model(X_dec_in_sample, mask=None, return_diagnostics=False, sampling_mode="expectation")
            for i in range(X_dec_in_sample.size(0)):
                if plotted_count >= num_samples_to_plot: break
                actual = Y_tgt_sample[i, :, feature_index_to_plot].cpu().numpy()
                predicted = predictions[i, :, feature_index_to_plot].cpu().numpy()
                timestamps_mdates = Y_tgt_sample[i, :, 0].cpu().numpy()
                timestamps_datetime = [mdates.num2date(ts) for ts in timestamps_mdates]
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps_datetime, actual, label=f'Actual {feature_names[feature_index_to_plot]}', marker='.')
                plt.plot(timestamps_datetime, predicted, label=f'Predicted {feature_names[feature_index_to_plot]}', linestyle='--')
                plt.title(f'{psymbol_name} - {feature_names[feature_index_to_plot]} Prediction vs Actual (Test Sample {plotted_count+1})')
                plt.xlabel('Time'); plt.ylabel(f'{feature_names[feature_index_to_plot]} Value')
                plt.legend(); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.xticks(rotation=45); plt.tight_layout(); plt.show()
                plotted_count +=1



def main_training():
    if config["training"]["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(config["training"]["device"])
    print(f"Using device: {DEVICE}")

    # --- Configuration for saving/loading model ---
    MODEL_SAVE_DIR = os.path.join(config["PROJECT_ROOT"], "saved_models") # Defined in config.py
    MODEL_FILENAME = "nft_model_checkpoint.pth"
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    
    # --- Attempt to Load Model ---
    # Define model architecture first (dummy instance to get class)
    # The actual parameters used will be from the checkpoint if loaded,
    # or from config if starting fresh.
    loaded_model=None
    if not config["data"]["force_regenerate_model"]:
        loaded_model, optimizer_state_dict, start_epoch = load_model(NFTnetworkBlock, MODEL_SAVE_PATH, DEVICE)

    if loaded_model:
        model = loaded_model
        print(f"Resuming training from epoch {start_epoch}")
    else:
        # --- Initialize Model from Scratch ---
        model = NFTnetworkBlock(
            embed_dim=config["model"]["embed_dim"],
            max_order=config["model"]["max_order_stats"],
            num_configs=config["model"]["num_configs"]
        ).to(DEVICE)
        start_epoch = 0 # Start from epoch 0 if not loaded
        optimizer_state_dict = None # No optimizer state to load
        print("Starting training from scratch.")

    # --- Get Data ---
    print("Getting training data...")

    #with cProfile.Profile() as pr:
    (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), \
    (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t) = get_training_data()
    #    print('\n',"Finished Inspection")
    #    stats = pstats.Stats(pr).strip_dirs()
    #    stats.sort_stats(pstats.SortKey.TIME)
    #    stats.dump_stats(filename="statdump.prof")  # snakeviz ./statdump.prof
    #    subprocess.run(["snakeviz", "./statdump.prof"])
    #exit(0)
    if Xtrain_dec_in_t.nelement() == 0:
        print("No training data generated or loaded. Exiting.")
        exit()
        
    Xtrain_dec_in_t = Xtrain_dec_in_t.to(DEVICE)
    Ytrain_tgt_t = Ytrain_tgt_t.to(DEVICE)
    Xval_enc_t = Xval_enc_t.to(DEVICE)
    Xval_dec_in_t = Xval_dec_in_t.to(DEVICE)
    Yval_tgt_t = Yval_tgt_t.to(DEVICE)

    print(f"Training data shapes: Dec_In: {Xtrain_dec_in_t.shape}, Tgt: {Ytrain_tgt_t.shape}")
    print(f"Validation data shapes: Dec_In: {Xval_dec_in_t.shape}, Tgt: {Yval_tgt_t.shape}")

    # --- Initialize Optimizer (load state if available) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
        print("Optimizer state loaded.")
        
    criterion = nn.MSELoss()

    # --- Training Loop ---
    print("\nStarting training...")
    NUM_EPOCHS = config["training"]["num_epochs"]
    BATCH_SIZE_TRAIN = config["training"]["batch_size"]
    CLIP_GRAD_NORM = config["training"]["clip_grad_norm"]

    best_val_loss = float('inf') # For saving the best model

    # Adjust NUM_EPOCHS if resuming: only train for remaining epochs
    # total_epochs_to_run = NUM_EPOCHS 
    # for epoch in range(start_epoch, total_epochs_to_run):
    # This loop ensures we run for `NUM_EPOCHS` total epochs, 
    # regardless of where we started. If `start_epoch` is, say, 10, and `NUM_EPOCHS` is 50,
    # we run from epoch 10 to 49.
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(Xtrain_dec_in_t.size(0))
        
        for i in range(0, Xtrain_dec_in_t.size(0), BATCH_SIZE_TRAIN):
            optimizer.zero_grad()
            indices = permutation[i : i + BATCH_SIZE_TRAIN]
            batch_x_dec_in, batch_y_tgt = Xtrain_dec_in_t[indices], Ytrain_tgt_t[indices]
            
            predictions = model(batch_x_dec_in, mask=None, return_diagnostics=False, sampling_mode="expectation")
            loss = criterion(predictions, batch_y_tgt)
            loss.backward()
            if CLIP_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            optimizer.step()
            epoch_loss += loss.item() * batch_x_dec_in.size(0)
        
        epoch_loss /= Xtrain_dec_in_t.size(0)
        
        val_loss = 0.0
        if Xval_dec_in_t.size(0) > 0:
            model.eval()
            with torch.no_grad():
                val_permutation = torch.randperm(Xval_dec_in_t.size(0))
                for i_val in range(0, Xval_dec_in_t.size(0), BATCH_SIZE_TRAIN):
                    val_indices = val_permutation[i_val : i_val + BATCH_SIZE_TRAIN]
                    batch_x_val_dec_in, batch_y_val_tgt = Xval_dec_in_t[val_indices], Yval_tgt_t[val_indices]
                    val_predictions = model(batch_x_val_dec_in, mask=None, return_diagnostics=False, sampling_mode="expectation")
                    v_loss = criterion(val_predictions, batch_y_val_tgt)
                    val_loss += v_loss.item() * batch_x_val_dec_in.size(0)
            val_loss /= Xval_dec_in_t.size(0)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, optimizer, epoch + 1, os.path.join(MODEL_SAVE_DIR, "nft_model_best.pth"))
                print(f"New best model saved with val_loss: {best_val_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.6f} (No validation data)")
            # If no validation data, save model at the end of each epoch or based on training loss
            # For simplicity here, just save the last one after training loop.

    print("Training finished.")

    # Save the final model checkpoint
    save_model(model, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)


    # --- Plotting Predictions (using the final or loaded model) ---
    if Xval_dec_in_t.size(0) > 0:
        print("\nPlotting predictions for a few validation samples...")
        # Optional: Load the best model for plotting if you saved it
        # best_model, _, _ = load_model(NFTnetworkBlock, os.path.join(MODEL_SAVE_DIR, "nft_model_best.pth"), DEVICE)
        # if best_model: model_to_plot = best_model
        # else: model_to_plot = model # fallback to last model
        model_to_plot = model # Plot with the model currently in memory (last epoch or loaded)

        num_plot_samples = min(config["plotting"]["num_samples_to_plot"], Xval_dec_in_t.size(0))
        val_sample_dec_in = Xval_dec_in_t[:num_plot_samples]
        val_sample_tgt = Yval_tgt_t[:num_plot_samples]
        
        plot_predictions(
            model=model_to_plot, 
            device=DEVICE, 
            data_loader_or_sample=(val_sample_dec_in, val_sample_tgt),
            psymbol_name=config["data"]["primary_symbol"],
            feature_index_to_plot=config["plotting"]["feature_index_to_plot"],
            num_samples_to_plot=num_plot_samples
        )
    else:
        print("No validation data to plot.")
    if Xval_dec_in_t.size(0) > 0:
        model.eval()
        with torch.no_grad():
            example_val_input = Xval_dec_in_t[:min(1, Xval_dec_in_t.size(0))] 
            if example_val_input.nelement() > 0:
                output, diagnostics = model(example_val_input, return_diagnostics=True, sampling_mode="expectation")
                print("\n--- Example Diagnostics (first validation sample) ---")
                print(f"  Output shape: {output.shape}")
                print(f"  Temperature: {diagnostics['temperature']:.4f}")
                print(f"  Log Z shape: {diagnostics['log_Z'].shape}") 
                if diagnostics['log_Z'].nelement() > 0:
                     print(f"  Avg Log Z (first seq): {diagnostics['log_Z'][0].mean().item():.4f}")
                     print(f"  Avg Entropy (first seq): {diagnostics['entropy'][0].mean().item():.4f}")
                     print(f"  Config Probs (first seq, first timestep): {diagnostics['config_probs'][0,0].cpu().numpy()}")
                print(f"--------------------------------------------------")
if __name__ == "__main__":
    # --- Device Setup ---
    main_training()
    