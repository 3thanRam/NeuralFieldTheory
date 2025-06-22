import torch
import os

from config import config
# Import EncoderDecoderModel and its specific loader from network.py
from network import EncoderDecoderModel, load_encoder_decoder_model 
from training import training

def main():
    if config["training"]["device"] == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] != "cpu" else "cpu")
    else:
        DEVICE = torch.device(config["training"]["device"])
    print(f"Using device: {DEVICE}")
    
    model = None
    optimizer_state_dict = None
    start_epoch = 0

    if config["data"].get("try_load_model", False):
        print("Attempting to load EncoderDecoderModel...")
        loaded_model, loaded_optimizer_state, completed_epoch = load_encoder_decoder_model(DEVICE)
        if loaded_model:
            model = loaded_model
            optimizer_state_dict = loaded_optimizer_state
            start_epoch = completed_epoch 
            print(f"EncoderDecoderModel loaded. Resuming training from epoch {start_epoch}.")
        else:
            print("Failed to load EncoderDecoderModel, or no model found. Initializing new model.")

    if model is None:
        print("Initializing a new EncoderDecoderModel.")
        # encoder_input_dim is now pre-calculated and stored in config["model"] by config.py
        encoder_input_dim = config["model"]["encoder_input_dim"]

        model = EncoderDecoderModel(
            encoder_input_dim=encoder_input_dim,
            decoder_embed_dim=config["model"]["decoder_embed_dim"],
            shared_embed_dim=config["model"]["shared_embed_dim"],
            num_blocks_enc=config["model"]["num_blocks_enc"],
            num_blocks_dec=config["model"]["num_blocks_dec"],
            max_order_enc=config["model"]["max_order_enc"],
            max_order_dec=config["model"]["max_order_dec"],
            num_configs_enc=config["model"]["num_configs_enc"],
            num_configs_dec=config["model"]["num_configs_dec"],
            max_seq_len_enc=config["model"]["max_seq_len_enc"],
            max_seq_len_dec=config["model"]["max_seq_len_dec"],
            num_lags_enc=config["model"]["num_lags_enc"],
            num_lags_dec=config["model"]["num_lags_dec"],
            dropout_rate=config["model"]["dropout_rate"],
            lstm_pe_layers_enc=config["model"]["lstm_pe_layers_enc"],
            lstm_pe_bidirectional_enc=config["model"]["lstm_pe_bidirectional_enc"],
            lstm_pe_layers_dec=config["model"]["lstm_pe_layers_dec"],
            lstm_pe_bidirectional_dec=config["model"]["lstm_pe_bidirectional_dec"]
        ).to(DEVICE)

    if config["mode"]=="training":
        training(model, optimizer_state_dict, start_epoch)

if __name__ == "__main__":
    main()
