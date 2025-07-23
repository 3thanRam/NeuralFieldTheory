# main.py
import torch
from types import SimpleNamespace
import os
from importlib import import_module

# --- All imports now come from the common directory ---
from common.base_model import BaseModel
from common.trainer import train_model
from common.loss import get_loss_fn
from common.data_loaders import prepare_chatbot_loaders, prepare_stock_loaders
from tasks.testing import test_model
# ==================================================================
#                       MASTER CONFIG PANEL
# ==================================================================
TASK_CONFIG = {
    "project_directory":  os.path.dirname(__file__),
    "tokenizer_directory":os.path.join( os.path.dirname(__file__),"model_data", "chatbot_tokenizer"),
    "task": "stockbot",  # 'stockbot' or 'chatbot'
    "mode": "test",     # 'train' or 'test'
    "load_checkpoint": True,
    "start_epoch": 0,
    "best_val_loss": torch.inf,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # --- Core Model Hyperparameters ---
    "core_model_class": "LNN",
    "reversible": False,          # True for Gauss-Seidel, False for Jacobi update
    "parallel_force": True,       # True for fast Conv1d, False for slow autograd
    "embed_dim": 256,    # Must be even if reversible=True
    "d_hidden_dim": 256,
    "num_blocks": 3,
    "dropout": 0.1,
    "dt": 0.1,
    "lr": 1e-3,
    "batch_size": 32,
    "val_split_ratio":0.1,
    
    
    # --- Data & Task Type Flags ---
    "use_tokenization": False,
    "use_normalization": True,

    "test_type": "stockbot", #chatbot,completetext,stockbot
    
    

    # Stockbot specific params
    "symbols":  ["GLD", "AAPL", "TSLA", "SPY", "TLT","NVDA","AMZN","MSFT","GOOGL","META","AVGO","BRK.B","TSM"],
    "primary_symbol": "GLD",
    "years_of_data": 5,
    "sequence_length": 60,
    
    # Chatbot specific params
    "vocab_size": None,
    "pad_idx": None,

    # --- Loss Weights ---
    "loss_config": {
        # The base loss should always have a weight of 1.0
        "stockbot_base": 1.0,
        "chatbot_base": 0., 
        
        # Auxiliary losses can be added, removed, or have their weights changed
        "norm_constraint": 0.1,
        "force_minimization": 1.,
        "force_decorrelation": 1.,
        "round_trip": 1.,
        "energy_ratio": 1.
    },
}
TASK_CONFIG["num_input_features"]=len(TASK_CONFIG["symbols"]) * 5,
# ==================================================================


def run_task(config_dict):
    config = SimpleNamespace(**config_dict)
    print(f"--- Task: {config.task} | Mode: {config.mode} ---")
    torch.manual_seed(42)
    val_loader,tokenizer=None,None

    # --- 1. Data Loading (Unchanged) ---
    if config.use_tokenization:
        train_loader, val_loader, tokenizer = prepare_chatbot_loaders(config)
        # Update config with data-dependent values
        config.vocab_size = len(tokenizer)
        config.pad_idx = tokenizer.pad_token_id
    else:
        train_loader, val_loader = prepare_stock_loaders(config)
        # Calculate dynamic feature sizes
        config.num_input_features = len(config.symbols) * 5
        config.num_output_predictions = len(config.symbols) * 5
    
    # --- 2. Build Core Model (Unchanged) ---
    CoreModelClass = getattr(import_module(f"models.{config.core_model_class.lower()}"), config.core_model_class)
    core_model = CoreModelClass(
        embed_dim=config.embed_dim,
        d_hidden_dim=config.d_hidden_dim,
        num_blocks=config.num_blocks,
        reversible=config.reversible,
        parallel_force=config.parallel_force,
        dropout=config.dropout,
        dt=config.dt
    )
    # --- 3. Build Wrapper Model (Unchanged) ---
    model = BaseModel(core_network=core_model, config=vars(config))
    
    # --- 4. NEW: Assemble the Loss Functions ---
    loss_functions = []
    # These are kwargs that might be needed by some loss functions
    loss_kwargs = {
        'pad_idx': getattr(config, 'pad_idx', None),
        'primary_symbol_idx': config.symbols.index(config.primary_symbol) if hasattr(config, 'symbols') else None,
        'huber_weight': 0.7, 'direction_weight': 0.3 # Example
    }
    
    for name, weight in config.loss_config.items():
        if weight > 0:
            loss_fn = get_loss_fn(name, **loss_kwargs)
            loss_functions.append((name, weight, loss_fn))
    
    print(f"Using losses: {[name for name, _, _ in loss_functions]}")

    # --- 5. Run Mode ---
    config.ckpt_path = os.path.join("model_data", f"{config.task}_checkpoint.pth.tar")
    if config.load_checkpoint:
        if os.path.exists(config.ckpt_path):
            print(f"--- Loading checkpoint from {config.ckpt_path} ---")
            checkpoint = torch.load(config.ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # We also load training state, which will be used by the trainer if mode is 'train'
            config.start_epoch = checkpoint.get('epoch', 0)
            config.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print(f"Warning: Checkpoint file not found at {config.ckpt_path}. Starting from scratch.")
            config.load_checkpoint = False # Prevent trainer from trying to load optimizer state

    if config.mode == 'train':
        train_model(config, model, loss_functions, train_loader, val_loader)
    elif config.mode == 'test':
        print("Testing Model")
        test_model(config, model,val_loader,tokenizer)

def main():
    run_task(TASK_CONFIG)

if __name__ == "__main__":
    main()