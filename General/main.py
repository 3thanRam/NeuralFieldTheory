# main.py
import torch
from types import SimpleNamespace
import os
from importlib import import_module

from common.base_model import BaseModel
from common.trainer import train_model
from common.loss import get_loss_fn
from common.data_loaders import prepare_chatbot_loaders, prepare_stock_loaders
from tasks.testing import test_model

TASK_CONFIG = {
    "project_directory":  os.path.dirname(__file__),
    "tokenizer_directory":os.path.join( os.path.dirname(__file__),"model_data", "chatbot_tokenizer"),
    "task": "chatbot",  # 'stockbot' or 'chatbot'
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
    "kernel_size": 3,  # Standard 3: 1 left, 1 right. 5 or 7 for a wider view...
    "d_hidden_dim": 512,
    "num_blocks": 2,
    "num_heads":10,
    "dropout": 0.1,
    "dt": 0.05,
    "lr": 1e-3,
    "batch_size": 32,
    "val_split_ratio":0.1,
    
    
    # --- Data & Task Type Flags ---
    "use_tokenization": True,
    "use_normalization": False,
    "normalization_type": "returns",  # Options: "last_point" or "sequence" or "relative" or "returns"

    "test_type": "chatbot", #chatbot,completetext,stockbot
    
    

    # Stockbot specific params
    "symbols":   ["GLD", "AAPL", "TSLA", "SPY", "TLT","NVDA","AMZN","MSFT","GOOGL","META","AVGO","BRK.B","TSM","JPM","WMT","LLY","ORCL","V","MA","NFLX"],
    "primary_symbol": "GLD",
    "years_of_data": 7,
    "sequence_length": 80,
    "test_history_length": 60,   # How many past data points to show on the plot
    "test_prediction_length": 20, # How many future data points to predict and plotc
    
    # Chatbot specific params
    "vocab_size": None,
    "pad_idx": None,

    # --- Loss Weights ---
    "loss_config": {
        # The base loss should always have a weight of 1.0
        "stockbot_base": 0.,
        "chatbot_base": 1.0, 
        
        # Auxiliary losses can be added, removed, or have their weights changed
        "norm_constraint": 100., # chatbot 0.01 stockbot 100
        "force_minimization": 500.,# chatbot 10 stockbot 500
        "force_decorrelation": 500.,# chatbot 10 stockbot 500
        "round_trip": 500.,# chatbot 10 stockbot 500
        "energy_ratio": 500.,# chatbot 20 stockbot 500
        "candle_shape": 0.# chatbot 0 stockbot 500
    },
}
TASK_CONFIG["num_input_features"]=len(TASK_CONFIG["symbols"]) * 5,
# ==================================================================


def run_task(config_dict):
    config = SimpleNamespace(**config_dict)
    print(f"--- Task: {config.task} | Mode: {config.mode} ---")
    torch.manual_seed(42)
    val_loader,tokenizer=None,None

    # --- 1. Data Loading 
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
    
    # --- 2. Build Core Model
    CoreModelClass = getattr(import_module(f"models.{config.core_model_class.lower()}"), config.core_model_class)
    core_model = CoreModelClass(
        embed_dim=config.embed_dim,
        d_hidden_dim=config.d_hidden_dim,
        num_blocks=config.num_blocks,
        reversible=config.reversible,
        parallel_force=config.parallel_force,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        dt=config.dt
    )
    # Wrapper Model  
    model = BaseModel(core_network=core_model, config=vars(config))
    
    #  Assemble the Loss Functions 
    loss_functions = []
    # These are kwargs that might be needed by some loss functions
    loss_kwargs = {
        'pad_idx': getattr(config, 'pad_idx', None),
        'primary_symbol_idx': config.symbols.index(config.primary_symbol) if hasattr(config, 'symbols') else None,
        'huber_weight': 0.7, 'direction_weight': 0.3 
    }
    
    for name, weight in config.loss_config.items():
        if weight > 0:
            loss_fn = get_loss_fn(name, **loss_kwargs)
            loss_functions.append((name, weight, loss_fn))
    
    print(f"Using losses: {[name for name, _, _ in loss_functions]}")

    # Load checkpoint from ckpt_path
    config.ckpt_path = os.path.join(config.project_directory,"model_data", f"{config.task}_checkpoint.pth.tar")
    if config.load_checkpoint:
        if os.path.exists(config.ckpt_path):
            print(f"--- Loading checkpoint from {config.ckpt_path} ---")
            checkpoint = torch.load(config.ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            config.start_epoch = checkpoint.get('epoch', 0)
            config.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print(f"Warning: Checkpoint file not found at {config.ckpt_path}. Starting from scratch.")
            config.load_checkpoint = False 

    if config.mode == 'train':
        train_model(config, model, loss_functions, train_loader, val_loader)
    elif config.mode == 'test':
        print("Testing Model")
        test_model(config, model,val_loader,tokenizer)

def main():
    run_task(TASK_CONFIG)

if __name__ == "__main__":
    main()