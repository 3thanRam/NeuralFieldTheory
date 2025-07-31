# training.py

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np

from LNFT import LNFT_EBM 

@torch.enable_grad()
def generate_for_training(model, start_embeddings, steps, step_size, noise_scale):
    """
    Improved negative sampling using Langevin dynamics with noise injection.
    This is used inside the training loop.
    """
    model.eval() # Set to eval mode for generation part
    q = start_embeddings
    
    for _ in range(steps):
        energy = model.forward_from_embeddings(q).sum()
        grad_q, = torch.autograd.grad(energy, q, retain_graph=False)
        
        noise = torch.randn_like(q) * noise_scale
        q.data = q.data - step_size * grad_q + noise
    
    model.train() # Return model to training mode
    return q


def improved_train_ebm_model(config, train_loader, plot_queue=None):
    """
    Improved EBM training with better sampling, regularization, and scheduling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Improved EBM Training on {device} ---")

    model = LNFT_EBM(config).to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.99)
    )
    
    # A powerful scheduler that helps escape local minima
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=len(train_loader) * 2,  # Restart every 2 epochs
        T_mult=1, # Can set to 2 to make cycles longer over time
        eta_min=1e-7
    )

    # Initialize persistent chain with small variance
    persistent_chain = torch.randn(
        config['batch_size'], 
        config['seq_len'], 
        config['embed_dim'],
        device=device
    ).detach() * 0.1
    
    # For tracking and analysis
    energy_history = {'real': [], 'fake': [], 'loss': []}

    try:
        for epoch in range(config['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                # --- Positive Phase ---
                real_data_ids = batch['input_ids'].to(device)
                real_data_embeddings = model.input_head(real_data_ids)
                energy_real = model.forward_from_embeddings(real_data_embeddings).mean()

                # --- Negative Phase ---
                persistent_chain.requires_grad = True
                
                fake_data_embeddings = generate_for_training(
                    model,
                    start_embeddings=persistent_chain,
                    steps=config['pcd_langevin_steps'],
                    step_size=config['langevin_step_size'],
                    noise_scale=config['noise_scale']
                )
                persistent_chain = fake_data_embeddings.detach()
                energy_fake = model.forward_from_embeddings(fake_data_embeddings).mean()
                
                # --- Loss Calculation with Regularization ---
                cd_loss = energy_real - energy_fake
                # Add regularization to keep energy values from exploding
                energy_reg = config['energy_reg_weight'] * (energy_real.pow(2) + energy_fake.pow(2)).mean()
                total_loss = cd_loss + energy_reg
                
                # --- Backward Pass ---
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # --- Logging and Plotting ---
                if plot_queue:
                    plot_queue.put(total_loss.item())
                
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'E_real': f"{energy_real.item():.2f}",
                    'E_fake': f"{energy_fake.item():.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # --- Periodic Chain Reinitialization ---
                # Helps prevent mode collapse by injecting new randomness.
                if batch_idx > 0 and batch_idx % 200 == 0:
                    reinit_size = config['batch_size'] // 4
                    if reinit_size > 0:
                        persistent_chain[:reinit_size] = torch.randn_like(
                            persistent_chain[:reinit_size]
                        ) * 0.1
            

    except KeyboardInterrupt:
        print("\n--- Training Interrupted. ---")
    
    finally:
        # Final save
        torch.save(model.state_dict(), config['save_path'])
        print(f"Final model saved to {config['save_path']}")