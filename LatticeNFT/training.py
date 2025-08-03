# training.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

from LNFT import LNFT_Noise_Predictor

def train_score_model(config, train_loader, plot_queue=None):
    """
    Trains the LNFT model using the Denoising Score Matching objective.
    This is much faster and more stable than EBM training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Denoising Score Matching Training on {device} ---")

    # The model is now the LNFT_Noise_Predictor
    model = LNFT_Noise_Predictor(config).to(device)
    if hasattr(torch, 'compile'):
        print("Compiling the model for performance...")
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 4, eta_min=1e-7)
    
    # Use Mean Squared Error loss
    criterion = nn.MSELoss()

    try:
        for epoch in range(config['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in pbar:
                optimizer.zero_grad()
                
                # --- 1. Get a batch of clean data ---
                clean_data_ids = batch['input_ids'].to(device)
                clean_embeddings = model.input_head(clean_data_ids)
                batch_size = clean_embeddings.shape[0]

                # --- 2. The Denoising Score Matching Process ---
                # a. Sample random timesteps (noise levels) for each item in the batch
                timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
                
                # b. Create the noise and the noised embeddings
                noise = torch.randn_like(clean_embeddings)
                
                # Get alphas from a pre-computed noise schedule
                alphas_cumprod = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, 1000, device=device), dim=0)
                sqrt_alphas_cumprod_t = alphas_cumprod[timesteps].sqrt().view(batch_size, 1, 1)
                sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod[timesteps]).sqrt().view(batch_size, 1, 1)

                noisy_embeddings = sqrt_alphas_cumprod_t * clean_embeddings + sqrt_one_minus_alphas_cumprod_t * noise
                
                # c. Get the model's prediction of the noise
                predicted_noise = model(noisy_embeddings, timesteps)
                
                # d. Calculate the loss
                loss = criterion(predicted_noise, noise)

                # --- 3. Backward Pass ---
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # --- Logging ---
                if plot_queue:
                    plot_queue.put(loss.item())
                
                pbar.set_postfix({
                    'mse_loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
    except KeyboardInterrupt:
        print("\n--- Training Interrupted. ---")
    
    finally:
        if plot_queue:
            plot_queue.put(None)
        state_to_save = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state_to_save, config['save_path'])
        print(f"Final model saved to {config['save_path']}")