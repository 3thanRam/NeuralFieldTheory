import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["GLD", "AAPL", "TSLA", "SPY", "TLT"]
PRIMARY_SYMBOL = "GLD"
# Each symbol provides 4 features (o, c, h, l).
INPUT_DIM = len(SYMBOLS) * 4 
OUTPUT_DIM = 4 # We are predicting 4 values (o, c, h, l) for the primary symbol

config={
    "mode": "test", # "train" or "test"
    "load_model": True, # Set to False for initial training
    "load_training_data": False, # Set to False to generate new data
    "model_save_path": os.path.join(PROJECT_ROOT, "data", "hamiltonian_model.pth"),
    "data_save_path": os.path.join(PROJECT_ROOT, "data", "hamiltonian_trainingdata.npz"),
    "num_blocks":2,
    "d_embedding":512,
    "d_hidden_dim":512,
    "output_dim":OUTPUT_DIM,
    "input_dim":INPUT_DIM,
    "sequence_length":100,
    "timestep":1,
    "symbols": SYMBOLS,
    "primary_symbol": PRIMARY_SYMBOL,
    "api_keys": { # Store your API keys securely, e.g., via environment variables
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY")
    },
    # --- Training ---
    "num_training_samples": 1e4,
    "num_epoch": 75,
    "batch_size": 32,
    "lr": 1e-2,
    "VAL_SPLIT_RATIO": 0.15,
    "max_grad_norm":1,
    # Loss function weights
    "mse_weight": 1.0,
    "direction_weight": 0.3,
    "volatility_weight": 0.1,
    
    # Curriculum learning
    "use_curriculum": True,

    "device": "cuda" if "torch" in globals() and torch.cuda.is_available() else "cpu",
}