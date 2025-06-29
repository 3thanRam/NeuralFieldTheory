# In a new main.py or your existing script that calls training()
import torch
import os
from config import config # Your updated config.py
from training import training, testing # Assuming training.py contains these
# Make sure HamiltonianSeq2Seq class is importable
from network import EncoderDecoderHamiltonianModel # Or wherever you save the class

def main():
    # --- Model Initialization ---
    if config["model_type"] == "EncoderDecoderHamiltonianModel": # New model type
        model = EncoderDecoderHamiltonianModel(
            vocab_size=config["vocab_size"],
            d_embedding=config["embed_dim"],
            enc_seq_len=config["enc_seq_len"],
            dec_seq_len=config["dec_seq_len"],
            d_hidden_potential_enc=config["d_hidden_potential_enc"],
            d_hidden_potential_dec=config["d_hidden_potential_dec"],
            num_ham_steps_enc=config["num_ham_steps_enc"],
            num_ham_steps_dec=config["num_ham_steps_dec"],
            h_step_integrator=config["h_step_integrator"],
            delta_t_momentum=config["delta_t_momentum"], # For encoder
            force_clip_value=config.get("force_clip_value"),
            pad_idx=config["pad_idx"],
            num_attn_heads=config.get("num_attn_heads", 4)
        )
        #config["model_save_path"] = os.path.join(PROJECT_ROOT, "data", "enc_dec_hamiltonian_model.pth")
    else:
        raise ValueError(f"Unsupported model_type: {config['model_type']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
    
    start_epoch = 0
    if config["load_model"] and os.path.exists(config["model_save_path"]):
        print(f"Loading model checkpoint from {config['model_save_path']}")
        checkpoint = torch.load(config["model_save_path"], map_location=torch.device(config["device"]))
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and config["mode"] == "train": # Only load optimizer in train mode
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        # You might want to also load config from checkpoint to ensure consistency if resuming
        # loaded_config = checkpoint.get('config', {})
        # config.update(loaded_config) # Be careful with overriding current config
        print(f"Loaded model from epoch {start_epoch}. Previous Val Loss (if available in future checkpoints): {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        print("No model loaded or path not found, starting from scratch.")


    if config["mode"] == "train":
        training(model, optimizer, start_epoch) # Pass start_epoch
    elif config["mode"] == "test":
        if not config["load_model"] or not os.path.exists(config["model_save_path"]):
            print("Error: Testing mode requires a pre-trained model to load. Set load_model=True and provide correct model_save_path.")
            return
        testing(model)
    else:
        print(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    main()