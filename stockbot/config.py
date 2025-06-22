# config.py
import os
from alpaca.data.timeframe import TimeFrameUnit

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

config = {
    "PROJECT_ROOT":PROJECT_ROOT,
    "mode":"training", #training or testing
    # --- Model Hyperparameters ---
    "model": {
        "shared_embed_dim": 128, 
        # encoder_input_dim will be calculated in main.py based on len(config.data.symbols)
        # and decoder_embed_dim. It's good practice to have it calculated rather than hardcoded
        # if it depends on other config values.
        "decoder_embed_dim": 5,  # For psymbol (timestamp, o,c,h,l) - also features per any symbol

        "num_blocks_enc": 2,
        "num_blocks_dec": 2,
        "max_order_enc": 3,
        "max_order_dec": 3,
        "num_configs_enc": 8,
        "num_configs_dec": 8,
        "max_seq_len_enc": 100, # Matches data.sequence_length for history
        "max_seq_len_dec": 100, # Matches data.sequence_length for prediction horizon
        "num_lags_enc": 50,
        "num_lags_dec": 50,
        "dropout_rate": 0.1,
        "lstm_pe_layers_enc":2,
        "lstm_pe_bidirectional_enc":False,
        "lstm_pe_layers_dec":2,
        "lstm_pe_bidirectional_dec":False
    },

    # --- Data Generation & Handling ---
    "data": {
        "total_samples": 10**3,
        "sequence_length": 100, # Used for both encoder history and decoder prediction length
        "val_split_ratio": 0.2,
        "data_file_path": os.path.join(PROJECT_ROOT, "data", "nft_trainingdata.npz"),
        "model_file_path": os.path.join(PROJECT_ROOT, "data", "nft_model_best.pth"),
        "timedelta_config": {
            "datetime": "days",
            "alpaca": TimeFrameUnit.Day
        },
        # IMPORTANT: Ensure primary_symbol is also in this list for Scenario 3.A
        "symbols": ["GLD"], # Ensure primary is here.
        "primary_symbol": "GLD", 
        "try_load_model":False, # Set to False if you want to force new model for testing this setup
        "try_load_data":True,   # Set to False to regenerate data if needed
        "max_years_lookback": 7,
    },

    # --- Training Hyperparameters ---
    "training": {
        "sampling_mode":"expectation", 
        "learning_rate": 1e-1, 
        "weight_decay": 1e-2,  
        "scheduler_patience": 5, 
        "num_epochs": 50, 
        "batch_size": 32,
        "clip_grad_norm": 0.5,
        "device": "auto",
    },
    "plotting": {
        "num_samples_to_plot": 3
    },
    "api_keys": {
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY")
    }
}

if config["data"]["primary_symbol"] not in config["data"]["symbols"]:
    print(f"Warning: Primary symbol '{config['data']['primary_symbol']}' was not in 'data.symbols'. Adding it.")
    # Make a mutable copy if it's a tuple, then append
    symbols_list = list(config["data"]["symbols"])
    symbols_list.append(config["data"]["primary_symbol"])
    # Remove duplicates if any by converting to set and back to list, preserving order is tricky here
    # A simple way if order doesn't strictly matter for non-primary:
    # config["data"]["symbols"] = list(set(symbols_list))
    # If order matters, and you just want to ensure it's present:
    temp_set = set()
    ordered_unique_symbols = []
    for sym in symbols_list:
        if sym not in temp_set:
            ordered_unique_symbols.append(sym)
            temp_set.add(sym)
    config["data"]["symbols"] = ordered_unique_symbols


# Calculate and store encoder_input_dim in the config for clarity and use in main.py/load_model
# This assumes all symbols will have 'decoder_embed_dim' features.
num_encoder_features_per_symbol = config["model"]["decoder_embed_dim"]
config["model"]["encoder_input_dim"] = len(config["data"]["symbols"]) * num_encoder_features_per_symbol
print(f"Calculated encoder_input_dim: {config['model']['encoder_input_dim']} "
      f"(based on {len(config['data']['symbols'])} symbols * {num_encoder_features_per_symbol} features each)")