import torch
import os
import numpy as np
from config import config 
from training import training,plot_learning_curves
from testing import testing as dual_testing
from data_handling import prepare_dataset_from_api

from network import HamiltonianModel 
from transformer import TransformerModel

def get_model(model_type):
    """Factory function to create a model instance based on type using shared config."""
    if model_type == "hamiltonian":
        print("Initializing HamiltonianModel with shared parameters...")
        model = HamiltonianModel(
            num_blocks=config["num_layers"],
            d_hidden_dim=config["d_ffn"],
            dropout=config["dropout"],
            input_dim=config["input_dim"],
            d_embedding=config["d_embedding"],
            output_dim=config["output_dim"],
            sequence_length=config["sequence_length"],
            timestep=config["timestep"],
        )
    elif model_type == "transformer":
        print("Initializing TransformerModel with shared parameters...")
        model = TransformerModel(
            num_encoder_layers=config["num_layers"],
            dim_feedforward=config["d_ffn"],
            dropout=config["dropout"],
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            d_embedding=config["d_embedding"],
            nhead=config["nhead"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model

def load_checkpoint(model, optimizer, model_type):
    """
    Loads a checkpoint for a specific model type from its unique file.
    Does NOT overwrite anything.
    """
    # **CRITICAL**: The path is unique for each model type.
    model_path = os.path.join(config["model_save_dir"], f"{model_type}_model.pth")
    start_epoch = 0
    
    if config["load_model"] and os.path.exists(model_path):
        print(f"Loading checkpoint for {model_type} from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(config["device"]))
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and config["mode"] == "train":
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError:
                print("Could not load optimizer state. This is normal if you changed the model architecture.")
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded {model_type} from epoch {start_epoch}.")
    else:
        print(f"No checkpoint found for {model_type} at {model_path}. Starting from scratch.")
    return start_epoch

def main():
    if config["mode"] == "train":
        # Prepare data once for all models
        if config.get("load_training_data", False) and os.path.exists(config["data_save_path"]):
            print(f"Loading shared training data from {config['data_save_path']}...")
            data = np.load(config["data_save_path"])
            all_X = np.concatenate([data['X_train'], data['X_val']], axis=0)
            all_Y = np.concatenate([data['Y_train'], data['Y_val']], axis=0)
        else:
            all_X, all_Y = prepare_dataset_from_api(
                symbols=config["symbols"],
                primary_symbol=config["primary_symbol"]
            )
        
        models_to_train = ["hamiltonian", "transformer"] if config["model_type"] == "dual" else [config["model_type"]]
        
        # This loop trains one model, saves it, then moves to the next. No overwriting.
        for model_type in models_to_train:
            print(f"\n{'='*20} TRAINING: {model_type.upper()} MODEL {'='*20}")
            model = get_model(model_type)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
            start_epoch = load_checkpoint(model, optimizer, model_type)
            
            # The training function saves the model to its unique path inside the training loop.
            training(model, optimizer, start_epoch, all_X, all_Y, model_type=model_type)
            print(f"{'='*20} FINISHED TRAINING: {model_type.upper()} MODEL {'='*20}")

    elif config["mode"] == "test":
        if config["model_type"] == "dual":
            print(f"\n{'='*20} DUAL TESTING MODE {'='*20}")
            # --- Load Hamiltonian Model from its file ---
            model1 = get_model("hamiltonian")
            # Dummy optimizer needed for the load function signature
            optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config["lr"], weight_decay=1e-4) 
            _ = load_checkpoint(model1, optimizer1, "hamiltonian")

            # --- Load Transformer Model from its file ---
            model2 = get_model("transformer")
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config["lr"], weight_decay=1e-4)
            _ = load_checkpoint(model2, optimizer2, "transformer")

            # --- Run Comparison Test with the two loaded models ---
            dual_testing(model1, model2, model1_name="Hamiltonian", model2_name="Transformer")
            plot_learning_curves()

        else: # Single model testing
            model_type = config['model_type']
            print(f"\n{'='*20} SINGLE TESTING MODE: {model_type.upper()} {'='*20}")
            model = get_model(model_type)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
            _ = load_checkpoint(model, optimizer, model_type)
            
            print(f"Generating test plot for single model: {model_type}")
            dual_testing(model, model, model1_name=model_type, model2_name=f"Actual Future (reference)")


    elif config["mode"] == "plot":
        plot_learning_curves()
        
    else:
        print(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    main()