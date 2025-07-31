# main.py

import torch
import multiprocessing as mp
import os

# Import all your functions and classes from other files
from LNFT import LNFT_EBM
from training import improved_train_ebm_model
from data_utils import setup_data
from plotting import plot_worker

if __name__ == "__main__":
    # --- 1. DEFINE CONFIGURATION ---
    config = {
        # Model
        "embed_dim": 64,
        "d_ff": 64,
        "num_blocks": 3,
        "dropout": 0.1,
        
        # Data & Tokenizer
        "tokenizer_name": "gpt2",
        "seq_len": 32,
        
        # Training
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "energy_reg_weight": 0.01,
        
        # Langevin Dynamics / PCD
        "pcd_langevin_steps": 5,
        "langevin_step_size": 2e-3,  
        "noise_scale": 0.01,         
        
        # System
        "save_path": os.path.join( os.path.dirname(__file__),"lnft_ebm_final.pth"),
        "plt_save_path":os.path.join( os.path.dirname(__file__),"trainingloss.pdf"),
        "plt_update_interval":20,
    }

    # --- 2. SETUP MULTIPROCESSING CONTEXT FIRST ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # This context can only be set once. If the script is re-run in the
        # same session, this will raise an error, which we can safely ignore.
        pass

    # --- 3. START THE BACKGROUND PLOTTER PROCESS ---
    # Create the communication queue and start the plotter immediately.
    # This isolates it from the DataLoader initialization.
    plot_queue = mp.Queue()
    plotter_process = mp.Process(target=plot_worker, args=(plot_queue,config["plt_save_path"],config["plt_update_interval"]))
    plotter_process.start()
    

    # --- 4. SETUP DATA ---
    # The DataLoader workers are created here, safely after the plotter process.
    train_loader, tokenizer = setup_data(config)

    # --- 5. TRAIN THE MODEL ---
    # Pass the already-created plot_queue to the training function.
    improved_train_ebm_model(config, train_loader, plot_queue)

    # --- 6. CLEANLY SHUT DOWN PLOTTER ---
    print("Main script finished training. Signaling plotter to terminate.")
    plot_queue.put(None) # Send the sentinel value
    plotter_process.join() # Wait for the process to finish
    print("Plotter process has terminated.")

    # --- 7. LOAD FINAL MODEL AND GENERATE ---
    print("\n--- Loading final model for generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = LNFT_EBM(config)
    
    if os.path.exists(config['save_path']):
        final_model.load_state_dict(torch.load(config['save_path'], map_location=device))
        final_model.to(device)

        print("\n--- Generating sample sentences using Langevin Dynamics ---")
        generated_sentences = final_model.generate(
            batch_size=4,
            seq_len=config['seq_len'],
            steps=200,
            step_size=5e-3,
            noise_scale=0.005,
            tokenizer=tokenizer
        )
        for i, sentence in enumerate(generated_sentences):
            print(f"Sample {i+1}: {sentence}")
    else:
        print("Final model file not found. Skipping generation.")