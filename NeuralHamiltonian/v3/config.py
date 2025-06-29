# config.py

import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["GLD", "AAPL", "TSLA", "SPY", "TLT","NVDA","AMZN"]
PRIMARY_SYMBOL = "GLD"
INPUT_DIM = len(SYMBOLS) * 4 
OUTPUT_DIM = 4

MODEL_TYPE = "hamiltonian" #"hamiltonian", "transformer","dual"

config={
    "model_type": MODEL_TYPE,
    "mode": "test", #train,test,plot
    "load_model": False,
    "load_training_data": False,

    "model_save_dir": os.path.join(PROJECT_ROOT, "data"),
    "history_save_path": os.path.join(PROJECT_ROOT, "data", "training_history.json"),
    "learning_curve_plot_path": os.path.join(PROJECT_ROOT, "data", "learning_curves.png"),
    "comparison_plot_path": os.path.join(PROJECT_ROOT, "data", "comparison_test_results.png"),
    "data_save_path": os.path.join(PROJECT_ROOT, "data", "trainingdata.npz"),
    
    # --- SHARED ARCHITECTURE PARAMETERS for Fair Comparison ---
    "d_embedding": 256,         # Embedding dimension for both models
    "num_layers": 1,            # SHARED: num_layers (Hamiltonian) and num_encoder_layers (Transformer)
    "d_ffn": 256,               # SHARED: d_hidden_dim (Hamiltonian) and dim_feedforward (Transformer)
    "dropout": 0.1,             # SHARED: Dropout rate for both models

    # --- Model-Specific Parameters ---
    "nhead": 8,                 # Transformer-specific: Number of attention heads
    "timestep": 0.2,              # Hamiltonian-specific
    
    # --- Shared Model Config (Dimensions) ---
    "output_dim": OUTPUT_DIM,
    "input_dim": INPUT_DIM,
    "sequence_length": 100,
    
    # --- Data and API Config ---
    "symbols": SYMBOLS,
    "primary_symbol": PRIMARY_SYMBOL,
    "api_keys": {
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY")
    },
    
    # --- Training Config ---
    "num_epoch": 40,
    "batch_size": 32,
    "lr": 1e-3,
    "VAL_SPLIT_RATIO": 0.15,
    "max_grad_norm": 1,
    
    # Loss function weights
    "mse_weight": 1.0,
    "direction_weight": 0.3,
    "volatility_weight": 0.1,
    
    "use_curriculum": True,

    "device": "cuda" if "torch" in globals() and torch.cuda.is_available() else "cpu",
}