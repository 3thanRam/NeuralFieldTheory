# main.py

import torch
import multiprocessing as mp
import os

from LNFT import LNFT_Noise_Predictor
from training import train_score_model
from data_setup import setup_data
from plotter import plot_worker
from sampler import DPMSolver


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Configuration for Score-Based Model ---
    config = {
        # Model
        "embed_dim": 128,
        "d_ff": 192,
        "num_blocks": 3,
        "dropout": 0.1,
        "gauge_rank": 16,
        "gauge_window_size": 16,
        
        # Data & Tokenizer
        "tokenizer_name": "gpt2",
        "seq_len": 48,
        
        # Training
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4, 
        "weight_decay": 0.01,
        
        # System
         "save_path": os.path.join( os.path.dirname(__file__),"lnft_ebm_final.pth"),
        "plt_save_path":os.path.join( os.path.dirname(__file__),"trainingloss.pdf"),
        "plt_update_interval":20,
    }

    plot_queue = mp.Queue()
    plotter_process = mp.Process(target=plot_worker, args=(plot_queue,config["plt_save_path"],config["plt_update_interval"]))
    plotter_process.start()
    
    train_loader, tokenizer = setup_data(config)
    
    train_score_model(config, train_loader, plot_queue)
    
    plotter_process.join()

    # --- Load final model and generate with DPM-Solver ---
    print("\n--- Loading final model for generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = LNFT_Noise_Predictor(config) 
    
    if os.path.exists(config['save_path']):
        raw_state_dict = torch.load(config['save_path'], map_location=device)
        clean_state_dict = {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in raw_state_dict.items()}
        final_model.load_state_dict(clean_state_dict)
        final_model.to(device)
        
        final_sampler = DPMSolver(final_model, config, device)

        print("\n--- Generating sample sentences using DPM-Solver++ ---")
        generated_sentences = final_sampler.generate(
            batch_size=4,
            seq_len=config['seq_len'],
            num_steps=25,
            tokenizer=tokenizer
        )
        for i, sentence in enumerate(generated_sentences):
            print(f"Sample {i+1}: {sentence}")
    else:
        print("Final model file not found. Skipping generation.")