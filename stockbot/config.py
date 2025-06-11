# config.py
import os
from alpaca.data.timeframe import TimeFrameUnit

# Get the absolute path of the directory where config.py is located
# This helps in defining other paths relative to the project root, assuming config.py is at the root or a known location.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

config = {
    "PROJECT_ROOT":PROJECT_ROOT,

    # --- Model Hyperparameters ---
    "model": {
        "embed_dim": 6,          # Number of features per stock (date, o, c, h, l, v)
        "max_order_stats": 2,    # Up to which order statistics (1:mean, 2:+std/var, 3:+skew, 4:+kurt)
        "num_configs": 8,        # Number of configurations in NFT
    },

    # --- Data Generation & Handling ---
    "data": {
        "total_samples": 10**3,    # Total data samples to generate (train + val)
        "sequence_length": 50,   # Sequence length for history and prediction horizon
        "val_split_ratio": 0.2,
        # Path to save/load the generated .npz data file
        "data_file_path": os.path.join(PROJECT_ROOT, "data", "nft_trainingdata.npz"),
        "timedelta_config": {
            "datetime": "days",  # Unit for timedelta calculations ('days', 'minutes', etc.)
            "alpaca": TimeFrameUnit.Day  # Alpaca TimeFrameUnit (Day, Hour, Minute)
        },
        "symbols": ["GLD", "AAPL", "TSLA", "SPY", "TLT"], # Stock symbols for encoder input
        "primary_symbol": "GLD", # Target symbol for prediction (decoder output)
        "force_regenerate_data": True,
        "force_regenerate_model": True,
    },

    # --- Training Hyperparameters ---
    "training": {
        "learning_rate": 1e-2,
        "num_epochs": 50,
        "batch_size": 32,
        "clip_grad_norm": 1.0, # Max norm for gradient clipping, set to None to disable
        # Device: 'cuda', 'cpu', or 'auto' (auto-detects CUDA)
        "device": "auto", # Will be resolved to torch.device in train_main.py
    },

    # --- Plotting Configuration (for train_main.py) ---
    "plotting": {
        "feature_index_to_plot": 2, # 0:date, 1:open, 2:close, 3:high, 4:low, 5:volume
        "num_samples_to_plot": 3,   # How many validation samples to plot after training
    },

    # --- Alpaca API Keys (Ideally use environment variables, but can be placeholders) ---
    # It's STRONGLY recommended to use environment variables for API keys.
    # These are here as a structural example if direct config is absolutely necessary (not advised for sensitive data).
    "api_keys": {
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"), # Fetches from env var
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY") # Fetches from env var
    }
}

# You can also add a helper function here to easily access nested dictionary keys
def get_config_value(key_path, default=None):
    """
    Accesses a value from the nested config dictionary using a 'path.to.key' string.
    Example: get_config_value("model.embed_dim")
    """
    keys = key_path.split('.')
    val = config
    try:
        for key in keys:
            val = val[key]
        return val
    except KeyError:
        return default
    except TypeError: # If a key leads to a non-dictionary intermediate value
        return default

if __name__ == '__main__':
    # Example of using get_config_value
    print(f"Model Embed Dim: {get_config_value('model.embed_dim')}")
    print(f"Learning Rate: {get_config_value('training.learning_rate')}")
    print(f"Data File Path: {get_config_value('data.data_file_path')}")
    print(f"Alpaca Key ID (from env): {get_config_value('api_keys.alpaca_api_key_id')}")