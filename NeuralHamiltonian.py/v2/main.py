import torch
import os
import numpy as np
from config import config 
from training import training 
from testing import testing
from network import HamiltonianModel 
from data_handling import prepare_dataset_from_api


def create_optimizer(model, learning_rate=3e-4, weight_decay=1e-2):
    """
    Create an advanced optimizer with:
    - Differential learning rates for different parameter groups
    - Gradient clipping
    - Weight decay
    """
    # Group parameters for differential learning rates
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'input_projection' in n or 'lm_head' in n],
         'lr': learning_rate * 0.5},  # Lower LR for input/output layers
        {'params': [p for n, p in model.named_parameters() if 'momentum_net' in n],
         'lr': learning_rate},  # Default LR for momentum net
        {'params': [p for n, p in model.named_parameters() if 'blocks' in n],
         'lr': learning_rate * 2},  # Higher LR for Hamiltonian blocks
    ]
    
    # AdamW is generally better than Adam for transformers and similar architectures
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),  # More stable than default (0.9, 0.999)
        eps=1e-6
    )
    
    return optimizer

def main():
    model = HamiltonianModel(
            num_blocks=config["num_blocks"],
            input_dim=config["input_dim"],
            d_embedding=config["d_embedding"],
            d_hidden_dim=config["d_hidden_dim"],
            output_dim=config["output_dim"],
            sequence_length=config["sequence_length"],
            timestep=config["timestep"],
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
    start_epoch = 0

    if config["load_model"] and os.path.exists(config["model_save_path"]):
        print(f"Loading model checkpoint from {config['model_save_path']}")
        checkpoint = torch.load(config["model_save_path"], map_location=torch.device(config["device"]))
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and config["mode"] == "train":
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded model from epoch {start_epoch}. Previous Val Loss (if available in future checkpoints): {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        print("No model loaded or path not found, starting from scratch.")


    if config["mode"] == "train":
        if config.get("load_training_data", False) and os.path.exists(config["data_save_path"]):
            print(f"Loading pre-split training data from {config['data_save_path']}...")
            data = np.load(config["data_save_path"])
            X_train, Y_train, X_val, Y_val = data['X_train'], data['Y_train'], data['X_val'], data['Y_val']
            # We need to combine them to pass to the training function, which will re-split
            all_X = np.concatenate([X_train, X_val], axis=0)
            all_Y = np.concatenate([Y_train, Y_val], axis=0)
        else:
            # Prepare the entire dataset from the API
            all_X, all_Y = prepare_dataset_from_api(
                symbols=config["symbols"],
                primary_symbol=config["primary_symbol"]
            )
        
        training(model, optimizer, start_epoch, all_X, all_Y)
    elif config["mode"] == "test":
        if not config["load_model"] or not os.path.exists(config["model_save_path"]):
            print("Error: Testing mode requires a pre-trained model to load. Set load_model=True and provide correct model_save_path.")
            return
        testing(model)
    else:
        print(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    main()