import os
import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import multiprocessing as mp

from LNFT import LNFT_EBM
from plotting import plot_worker

@torch.enable_grad()
def generate_for_training(model, start_embeddings, hmc_steps, hmc_step_size):
    """
    Performs a few steps of HMC starting from a given state.
    This is used to generate negative samples during training.
    Returns final embeddings, not token IDs.
    """
    model.eval()
    q = start_embeddings
    p = torch.randn_like(q)
    energy = model.forward_from_embeddings(q).sum()
    grad_S_q, = torch.autograd.grad(energy, q)
    p = p - (hmc_step_size / 2) * grad_S_q
    for _ in range(hmc_steps):
        q.data += hmc_step_size * p
        energy = model.forward_from_embeddings(q).sum()
        grad_S_q, = torch.autograd.grad(energy, q)
        p = p - hmc_step_size * grad_S_q
    p = p - (hmc_step_size / 2) * grad_S_q
    p = -p
    model.train()
    return q


def train_ebm_model(config, train_loader):
    """
    Trains the LNFT_EBM using Persistent Contrastive Divergence (PCD)
    and includes a live plotting process.
    """
    # --- 1. Process and Queue Initialization for Plotting ---
    # Create a queue for communication between training and plotting processes
    plot_queue = mp.Queue()
    # Create the background process, targeting our plot_worker function
    plotter_process = mp.Process(target=plot_worker, args=(plot_queue,config["plt_save_path"],config["plt_update_interval"]))
    plotter_process.start()
    
    # --- 2. Standard Model and Optimizer Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting EBM Training on {device} with PCD ---")

    model = LNFT_EBM(config).to(device)
    optimizer = AdamW(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config.get('weight_decay', 0.01) # Add this
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['num_epochs'], eta_min=1e-8)

    # --- 3. Persistent Contrastive Divergence (PCD) Setup ---
    # Initialize the persistent buffer for the MCMC chain
    persistent_chain = torch.randn(
        config['batch_size'], 
        config['seq_len'], 
        config['embed_dim'],
        device=device
    ).detach()

    # --- 4. Main Training Loop with Graceful Exit ---
    try:
        for epoch in range(config['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in pbar:
                optimizer.zero_grad()
                
                # --- Positive Phase: Energy of real data ---
                real_data_ids = batch['input_ids'].to(device)
                real_data_embeddings = model.input_head(real_data_ids)
                energy_real = model.forward_from_embeddings(real_data_embeddings).mean()

                # --- Negative Phase: Energy of generated data via PCD ---
                # Set requires_grad=True to compute gradients for the HMC update
                persistent_chain.requires_grad = True
                fake_data_embeddings = generate_for_training(
                    model,
                    start_embeddings=persistent_chain,
                    hmc_steps=config['pcd_hmc_steps'],
                    hmc_step_size=config['hmc_step_size']
                )
                
                # Update the persistent chain for the next iteration.
                # Detach to cut the gradient history between training steps.
                persistent_chain = fake_data_embeddings.detach()
                
                energy_fake = model.forward_from_embeddings(fake_data_embeddings).mean()
                
                # --- Contrastive Divergence Loss ---
                loss = energy_real - energy_fake
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() 
                # --- Send Data to Plotter ---
                # Put the loss value into the queue for the background process
                plot_queue.put(loss.item())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'E_real': f"{energy_real.item():.2f}",
                    'E_fake': f"{energy_fake.item():.2f}"
                })
        
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by user (Ctrl+C). ---")
    
    finally:
        # --- 5. Clean Shutdown of Plotter and Final Save ---
        print("Signaling plotter to terminate and saving final model...")
        
        # Send a sentinel value (None) to tell the plotter process to exit its loop
        plot_queue.put(None)
        # Wait for the plotter process to finish cleaning up (e.g., show the final plot)
        plotter_process.join()
        
        torch.save(model.state_dict(), config['save_path'])
        print(f"Model saved to {config['save_path']}")