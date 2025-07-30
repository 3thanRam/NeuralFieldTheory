import os
import torch 
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from LNFT import LNFT_EBM



import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os


def train_ebm_model(config, train_loader):
    """
    Trains the LNFT_EBM using Persistent Contrastive Divergence (PCD).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting EBM Training on {device} with PCD ---")

    # 1. Initialize Model, Optimizer
    model = LNFT_EBM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    # --- PCD INITIALIZATION ---
    # Create the persistent buffer for our MCMC chain.
    # This buffer will hold the generated samples from the previous step.
    # It starts as random noise. We use .detach() so it's not part of the computation graph.
    persistent_chain = torch.randn(
        config['batch_size'], 
        config['seq_len'], 
        config['embed_dim'],
        device=device
    ).detach()
    # --------------------------

    # --- The Contrastive Divergence Training Loop ---
    try:
        for epoch in range(config['num_epochs']):
            print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                optimizer.zero_grad()
                
                # --- Positive Phase: Push energy of real data down ---
                real_data_ids = batch['input_ids'].to(device)
                
                # Convert real token IDs to embeddings for a fair energy comparison later
                real_data_embeddings = model.input_head(real_data_ids)
                
                # Calculate energy of real data
                energy_real = model.forward_from_embeddings(real_data_embeddings).mean()

                # --- Negative Phase with PCD ---
                # Generate new "fake" samples by running a SHORT MCMC chain 
                # starting from our PERSISTENT buffer.
                
                # The generate_for_training function now takes a starting point.
                # It's crucial that requires_grad=True is set on the starting tensor.
                persistent_chain.requires_grad = True
                fake_data_embeddings = generate_for_training(
                    model,
                    start_embeddings=persistent_chain,
                    hmc_steps=config['pcd_hmc_steps'], # Use a small number of steps
                    hmc_step_size=config['hmc_step_size']
                )
                
                # IMPORTANT: Update the persistent chain for the NEXT iteration.
                # We detach the result so that the gradient history of this step
                # does not carry over to the next.
                persistent_chain = fake_data_embeddings.detach()
                
                # Calculate energy of the newly generated fake data
                energy_fake = model.forward_from_embeddings(fake_data_embeddings).mean()
                
                # --- Contrastive Divergence Loss ---
                # The objective is still to lower real energy and raise fake energy.
                loss = energy_real - energy_fake
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping is good practice
                optimizer.step()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'E_real': f"{energy_real.item():.2f}",
                    'E_fake': f"{energy_fake.item():.2f}"
                })
        
        # --- Save Final Model ---
        print("Training finished. Saving final model.")
        torch.save(model.state_dict(), config['save_path'])

    except KeyboardInterrupt:
        print("\n--- Training Interrupted. Saving final model. ---")
        torch.save(model.state_dict(), config['save_path'])

# We need a dedicated generation function for training that returns embeddings
# instead of token IDs, to avoid a non-differentiable step inside the loop.

@torch.enable_grad()
def generate_for_training(model, start_embeddings, hmc_steps, hmc_step_size):
    """
    Performs a few steps of HMC starting from a given state.
    This is used to generate negative samples during training.
    Returns final embeddings, not token IDs.
    """
    model.eval() # Set to eval mode for generation
    
    # q is our position, which is the starting embeddings
    q = start_embeddings
    
    # p is our momentum, sampled randomly each time
    p = torch.randn_like(q)

    # Calculate initial gradient
    energy = model.forward_from_embeddings(q).sum()
    grad_S_q, = torch.autograd.grad(energy, q)
    
    # --- Leapfrog Integrator ---
    p = p - (hmc_step_size / 2) * grad_S_q
    for _ in range(hmc_steps):
        q.data += hmc_step_size * p
        energy = model.forward_from_embeddings(q).sum()
        grad_S_q, = torch.autograd.grad(energy, q)
        p = p - hmc_step_size * grad_S_q
    p = p - (hmc_step_size / 2) * grad_S_q
    p = -p # Make the proposal symmetric
    
    model.train() # Return model to training mode
    return q