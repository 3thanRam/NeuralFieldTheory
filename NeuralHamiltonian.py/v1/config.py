# config.py
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Stock Tokenization Configuration (Option 4: Sequential OHLC Features) ---
TOKENS_PER_DAY = 4

def generate_bin(bin_name,mini,maxi,N):
    bins=[]
    bins.append((-float('inf'),mini,f"{bin_name}_LARGEminus"))
    prev=mini
    step=(maxi-mini)/N
    for i in range(N):
        sup=mini+step*(i+1)
        bins.append((prev,sup,f"{bin_name}_size{i}"))
        prev=sup
    bins.append((maxi,float('inf'),f"{bin_name}_LARGEplus"))
    bins.append((None, None, f"NO_{bin_name}_DATA"))
    return bins

Nbins=10**4

BINS_GAP=generate_bin("GAP",-0.01,0.01,Nbins)
BINS_UPPER_WICK=generate_bin("UWICK",-0.01,0.01,Nbins)
BINS_LOWER_WICK=generate_bin("LWICK",-0.01,0.01,Nbins)
BINS_BODY=generate_bin("BODY",-0.01,0.01,Nbins)

#BINS_GAP = [
#    (-float('inf'), -0.02, "GAP_DOWN_LARGE"), (-0.02, -0.005, "GAP_DOWN_SMALL"),
#    (-0.005, 0.005, "GAP_NONE"),
#    (0.005, 0.02, "GAP_UP_SMALL"), (0.02, float('inf'), "GAP_UP_LARGE"),
#    (None, None, "NO_GAP_DATA")
#]
#
#BINS_UPPER_WICK = [
#    (-float('inf'), 0.001, "UWICK_NONE"),
#    (0.001, 0.01, "UWICK_SMALL"), 
#    (0.01, 0.03, "UWICK_MEDIUM"),
#    (0.03, float('inf'), "UWICK_LARGE"),
#    (None, None, "NO_UWICK_DATA")
#]
#BINS_LOWER_WICK = [
#    (-float('inf'), 0.001, "LWICK_NONE"),
#    (0.001, 0.01, "LWICK_SMALL"),
#    (0.01, 0.03, "LWICK_MEDIUM"),
#    (0.03, float('inf'), "LWICK_LARGE"),
#    (None, None, "NO_LWICK_DATA")
#]
#BINS_BODY = [
#    (-float('inf'), -0.03, "BODY_NEG_LARGE"), (-0.03, -0.01, "BODY_NEG_MEDIUM"),
#    (-0.01, -0.002, "BODY_NEG_SMALL"),
#    (-0.002, 0.002, "BODY_FLAT"),
#    (0.002, 0.01, "BODY_POS_SMALL"), (0.01, 0.03, "BODY_POS_MEDIUM"),
#    (0.03, float('inf'), "BODY_POS_LARGE"),
#    (None, None, "NO_BODY_DATA")
#]
ALL_FEATURE_BINS = {
    "gap": BINS_GAP, "upper_wick": BINS_UPPER_WICK,
    "lower_wick": BINS_LOWER_WICK, "body": BINS_BODY
}
SPECIAL_TOKENS = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
_current_token_id_counter = len(SPECIAL_TOKENS)
STOCK_TOKEN_TO_ID = {}
STOCK_ID_TO_TOKEN = {}
STOCK_FEATURE_TOKEN_RANGES = {}
for feature_type, bins_list in ALL_FEATURE_BINS.items():
    start_id = _current_token_id_counter
    for _, _, token_name in bins_list:
        full_token_name = f"{feature_type.upper()}_{token_name}"
        if full_token_name not in STOCK_TOKEN_TO_ID:
            STOCK_TOKEN_TO_ID[full_token_name] = _current_token_id_counter
            STOCK_ID_TO_TOKEN[_current_token_id_counter] = full_token_name
            _current_token_id_counter += 1
    STOCK_FEATURE_TOKEN_RANGES[feature_type] = (start_id, _current_token_id_counter -1)
VOCAB_SIZE = _current_token_id_counter
REPRESENTATIVE_VALUES_FOR_BINS = {}
for feature_type, bins_list in ALL_FEATURE_BINS.items():
    REPRESENTATIVE_VALUES_FOR_BINS[feature_type] = {}
    no_data_token_name_for_feature = f"{feature_type.upper()}_NO_{feature_type.upper()}_DATA"
    for lower, upper, token_name_suffix in bins_list:
        full_token_name = f"{feature_type.upper()}_{token_name_suffix}"
        token_id = STOCK_TOKEN_TO_ID.get(full_token_name)
        if token_id is None: continue
        if token_name_suffix.startswith("NO_") and token_name_suffix.endswith("_DATA"):
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = 0.0
        elif lower == -float('inf'):
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = upper - 0.005
        elif upper == float('inf'):
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = lower + 0.005
        elif lower is not None and upper is not None:
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = (lower + upper) / 2.0
        else:
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = 0.0

config = {
    # General project config
    "PROJECT_ROOT": PROJECT_ROOT,
    "load_model": False, # Set to False for initial training
    "mode": "train", # "train" or "test"
    "content": "stocks_as_ohlc_feature_tokens",
    "model_save_path": os.path.join(PROJECT_ROOT, "data", "hamiltonian_ohlc_feature_model.pth"),
    "data_save_path": os.path.join(PROJECT_ROOT, "data", "hamiltonian_stock_ohlc_feature_token_trainingdata.npz"),
    "load_training_data": False, # Set to False to generate new data
    "VAL_SPLIT_RATIO": 0.1,
    "device": "cuda" if os.getenv("FORCE_CPU") is None and torch.cuda.is_available() else "cpu",

    # Data generation
    "num_training_samples": 500, # Adjust as needed
    "symbols": ["GLD", "AAPL", "TSLA", "SPY", "TLT"],
    "primary_symbol": "GLD",
    "timeframe_value": 1, # Daily
     "api_keys": { # Store your API keys securely, e.g., via environment variables
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY")
    },

    # Tokenizer and Vocabulary (derived from above)
    "vocab_size": VOCAB_SIZE,
    "pad_idx": SPECIAL_TOKENS["<PAD>"],
    "sos_token_id": SPECIAL_TOKENS["<SOS>"],
    "eos_token_id": SPECIAL_TOKENS["<EOS>"],
    "tokens_per_day": TOKENS_PER_DAY,

    # Core Model Hyperparameters
    "model_type": "EncoderDecoderHamiltonianModel", # For selecting the model
    "embed_dim": 128, # d_embedding for Hamiltonian model
    #"max_seq_len": 30 * TOKENS_PER_DAY, # Must match Hamiltonian model's sequence_length

    # Hamiltonian Model Specific Hyperparameters
    "d_hidden_potential": 128, # Hidden dim for the potential MLP
    "num_hamiltonian_steps": 2,
    "h_step_integrator": 0.1,
    "delta_t_momentum": 1.0,
    "force_clip_value": 1.0, # Or None

    # Training Loop
    "num_epoch": 50,
    "start_epoch": 0, # Will be updated if model is loaded
    "batch_size": 16, # Hamiltonian models can be memory intensive
    "lr": 1e-3, # May need tuning
    "weight_decay": 1e-4,
    "clip_grad_norm": 1.0, # For overall model parameter gradients
    "scheduler_patience": 5,
    "dropout_p": 0.1, # Not directly used by current Hamiltonian model, but good to keep if you add dropout layers

    # These are for reference by data_handling and testing, not direct model params
    "STOCK_TOKEN_TO_ID": STOCK_TOKEN_TO_ID,
    "STOCK_ID_TO_TOKEN": STOCK_ID_TO_TOKEN,
    "ALL_FEATURE_BINS": ALL_FEATURE_BINS,
    "REPRESENTATIVE_VALUES_FOR_BINS": REPRESENTATIVE_VALUES_FOR_BINS,
    "SPECIAL_TOKENS": SPECIAL_TOKENS,

    # Deprecated/Unused by Hamiltonian model (from your original config)
    # "num_blocks": 3,
    # "max_order": 3,
    "enc_seq_len": 30 * TOKENS_PER_DAY, # Or a different length for context
    "dec_seq_len": 30 * TOKENS_PER_DAY, # Length of sequence to predict (must match target data)
                                      # This is your old "max_seq_len"
    "d_hidden_potential_enc": 128,
    "d_hidden_potential_dec": 128, # Can be different
    "num_ham_steps_enc": 2,
    "num_ham_steps_dec": 2, # Can be different
    "num_attn_heads": 4, # For cross-attention
    # ...
    "max_seq_len": 30 * TOKENS_PER_DAY
}

