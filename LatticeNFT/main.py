import os
import torch
from training import train_model,train_ebm_model
from data_utils import setup_data
from tasks import load_model_for_inference,test_model,manual_test_chat
from LNFT import LNFT_EBM


if __name__ == "__main__":
    config = {
        "mode":"train",
        # Model
        "embed_dim": 64,         # D
        "d_ff": 128,             # MLP hidden dim
        "num_blocks": 1,         # Number of GaugeConvolutionBlocks
        "dropout": 0.1,
        
        # Data & Tokenizer
        "tokenizer_name": "gpt2",
        "seq_len": 32,           # Keep this small due to O(N^2) complexity
        
        # Training
        "num_epochs": 10,
        "batch_size": 16,        # Keep this small
        "learning_rate": 1e-3,
        
        # HMC Generation (for training contrastive loss)
        "hmc_steps": 5,
        "pcd_hmc_steps": 1, # Using 1-step PCD is very common and efficient
        "hmc_step_size": 1e-2,
        
        # System
        "save_path": "lnft_ebm_final.pth"
    }

    # 1. Setup Data
    train_loader, tokenizer = setup_data(config)

    if config["mode"]=="train":
        # 2. Train the EBM
        train_ebm_model(config, train_loader)

    

    if config["mode"]=="test":
        # 3. Load final model and generate some samples
        print("\n--- Loading final model for generation ---")
        final_model = LNFT_EBM(config)
        final_model.load_state_dict(torch.load(config['save_path']))
        final_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("\n--- Generating sample sentences ---")
        generated_sentences = final_model.generate(
            batch_size=4,
            seq_len=config['seq_len'],
            hmc_steps=50, # Use more steps for higher quality generation
            hmc_step_size=1e-2,
            tokenizer=tokenizer
        )

        for i, sentence in enumerate(generated_sentences):
            print(f"Sample {i+1}: {sentence}")