# config.py
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Stock Tokenization Configuration (Option 4: Sequential OHLC Features) ---
# Number of sub-tokens per day
TOKENS_PER_DAY = 4

# Bins for Overnight Gap: (Open_today - Close_yesterday) / Close_yesterday
BINS_GAP = [
    (-float('inf'), -0.02, "GAP_DOWN_LARGE"), (-0.02, -0.005, "GAP_DOWN_SMALL"),
    (-0.005, 0.005, "GAP_NONE"),
    (0.005, 0.02, "GAP_UP_SMALL"), (0.02, float('inf'), "GAP_UP_LARGE"),
    (None, None, "NO_GAP_DATA")
]

# Bins for Upper Wick: (High - max(Open, Close)) / Open (normalized by Open)
# These values are typically positive or zero.
BINS_UPPER_WICK = [
    (-float('inf'), 0.001, "UWICK_NONE"), # Effectively no upper wick or negative (should be rare if H>=O,C)
    (0.001, 0.01, "UWICK_SMALL"), 
    (0.01, 0.03, "UWICK_MEDIUM"),
    (0.03, float('inf'), "UWICK_LARGE"),
    (None, None, "NO_UWICK_DATA")
]

# Bins for Lower Wick: (min(Open, Close) - Low) / Open (normalized by Open)
# These values are typically positive or zero.
BINS_LOWER_WICK = [
    (-float('inf'), 0.001, "LWICK_NONE"),
    (0.001, 0.01, "LWICK_SMALL"),
    (0.01, 0.03, "LWICK_MEDIUM"),
    (0.03, float('inf'), "LWICK_LARGE"),
    (None, None, "NO_LWICK_DATA")
]

# Bins for Body: (Close - Open) / Open
BINS_BODY = [
    (-float('inf'), -0.03, "BODY_NEG_LARGE"), (-0.03, -0.01, "BODY_NEG_MEDIUM"),
    (-0.01, -0.002, "BODY_NEG_SMALL"),
    (-0.002, 0.002, "BODY_FLAT"),
    (0.002, 0.01, "BODY_POS_SMALL"), (0.01, 0.03, "BODY_POS_MEDIUM"),
    (0.03, float('inf'), "BODY_POS_LARGE"),
    (None, None, "NO_BODY_DATA")
]

ALL_FEATURE_BINS = {
    "gap": BINS_GAP, "upper_wick": BINS_UPPER_WICK,
    "lower_wick": BINS_LOWER_WICK, "body": BINS_BODY
}

# --- Vocabulary and Token ID Management ---
SPECIAL_TOKENS = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
_current_token_id_counter = len(SPECIAL_TOKENS)
STOCK_TOKEN_TO_ID = {}
STOCK_ID_TO_TOKEN = {}
STOCK_FEATURE_TOKEN_RANGES = {} # To know which IDs belong to which feature type

for feature_type, bins_list in ALL_FEATURE_BINS.items():
    start_id = _current_token_id_counter
    for _, _, token_name in bins_list:
        full_token_name = f"{feature_type.upper()}_{token_name}" # Ensure unique names
        if full_token_name not in STOCK_TOKEN_TO_ID:
            STOCK_TOKEN_TO_ID[full_token_name] = _current_token_id_counter
            STOCK_ID_TO_TOKEN[_current_token_id_counter] = full_token_name
            _current_token_id_counter += 1
    STOCK_FEATURE_TOKEN_RANGES[feature_type] = (start_id, _current_token_id_counter -1)


VOCAB_SIZE = _current_token_id_counter # Total number of unique tokens

# Representative values for detokenization (midpoints of bins for each feature type)
REPRESENTATIVE_VALUES_FOR_BINS = {}
for feature_type, bins_list in ALL_FEATURE_BINS.items():
    REPRESENTATIVE_VALUES_FOR_BINS[feature_type] = {}
    no_data_token_name_for_feature = f"{feature_type.upper()}_NO_{feature_type.upper()}_DATA"

    for lower, upper, token_name_suffix in bins_list:
        full_token_name = f"{feature_type.upper()}_{token_name_suffix}"
        token_id = STOCK_TOKEN_TO_ID.get(full_token_name)
        if token_id is None: continue

        if token_name_suffix.startswith("NO_") and token_name_suffix.endswith("_DATA"): # e.g. NO_GAP_DATA
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = 0.0
        elif lower == -float('inf'):
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = upper - 0.005 # Example
        elif upper == float('inf'):
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = lower + 0.005 # Example
        elif lower is not None and upper is not None:
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = (lower + upper) / 2.0
        else: # Should not happen
            REPRESENTATIVE_VALUES_FOR_BINS[feature_type][token_id] = 0.0


config = {
    "load_model": True, "mode": "test", "content": "stocks_as_ohlc_feature_tokens",
    "num_training_samples": 500, # Might need more as sequences are effectively shorter in days
    "model_save_path": os.path.join(PROJECT_ROOT, "data", "unified_ohlc_feature_model.pth"),
    "data_save_path": os.path.join(PROJECT_ROOT, "data", "stock_ohlc_feature_token_trainingdata.npz"),
    "load_training_data": False, "VAL_SPLIT_RATIO": 0.1,

    "vocab_size": VOCAB_SIZE, "embed_dim": 256, # Possibly larger embed_dim
    "num_blocks": 3,
    # max_seq_len is now in terms of sub-day tokens.
    # If you want to model 20 days, max_seq_len = 20 * TOKENS_PER_DAY
    "max_seq_len": 30 * TOKENS_PER_DAY, # e.g., 20 days * 4 tokens/day = 80 total tokens
    "tokens_per_day": TOKENS_PER_DAY,
    "max_order": 3,
    "pad_idx": SPECIAL_TOKENS["<PAD>"], "sos_token_id": SPECIAL_TOKENS["<SOS>"],
    "eos_token_id": SPECIAL_TOKENS["<EOS>"], "dropout_p": 0.1,

    "num_epoch": 80, "start_epoch": 0, "batch_size": 32,
    "lr": 3e-4, "weight_decay": 1e-5, "clip_grad_norm": 1.0, "scheduler_patience": 5,
    "symbols": ["GLD", "AAPL", "TSLA", "SPY", "TLT"], "primary_symbol": "GLD", "timeframe_value": 1,
    "api_keys": {
        "alpaca_api_key_id": os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("APCA_API_SECRET_KEY")
    },
    "device": "cuda" if os.getenv("FORCE_CPU") is None and torch.cuda.is_available() else "cpu",

    "STOCK_TOKEN_TO_ID": STOCK_TOKEN_TO_ID,
    "STOCK_ID_TO_TOKEN": STOCK_ID_TO_TOKEN,
    "ALL_FEATURE_BINS": ALL_FEATURE_BINS, # Store all bin definitions
    "REPRESENTATIVE_VALUES_FOR_BINS": REPRESENTATIVE_VALUES_FOR_BINS,
    "SPECIAL_TOKENS": SPECIAL_TOKENS,
    "PROJECT_ROOT": PROJECT_ROOT,
}