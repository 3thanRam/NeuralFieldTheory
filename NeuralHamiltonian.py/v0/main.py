# main.py
from config import config # Ensure this is at the top
import torch
import torch.optim as optim
import os # For os.path.exists

from training import training, testing
from network import HamiltonianModel

def main():
    device = torch.device(config.get("device", "cpu"))
    print(f"Using device: {device}")

    model = HamiltonianModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_blocks=config["num_blocks"],
        max_seq_len=config["max_seq_len"],
        model_max_order=config["max_order"],
        pad_idx=config["pad_idx"],
        dropout_p=config.get("dropout_p", 0.1)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))
    start_epoch = config.get("start_epoch", 0)

    model_path = config.get("model_save_path")
    if config.get("load_model", False) and model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', start_epoch)
        print(f"Resuming training from epoch {start_epoch + 1}") # epoch in checkpoint is last completed epoch
    elif config.get("load_model", False):
        print(f"Warning: load_model is True but model_save_path '{model_path}' not found or not specified.")


    if config["mode"] == "train":
        training(model, optimizer, start_epoch)
    elif config["mode"] == "test":
        if not (config.get("load_model", False) and model_path and os.path.exists(model_path)):
            print("Warning: Running test mode with a fresh/untrained model as no model was loaded.")
        testing(model)
    else:
        print(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    main()