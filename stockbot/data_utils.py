# data_utils.py
import numpy as np
import os
import torch
from datetime import timedelta, datetime
import matplotlib.dates as mdates

from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest

from config import config # Import the global config dictionary

# ... (time_fct, validconfig, savedata, gen_data, process_bars_for_symbol, get_stock_data_batch remain the same) ...
def time_fct(start_dt, Nsteps, step_unit_str):
    return start_dt + timedelta(**{f"{step_unit_str}": int(Nsteps)})

def validconfig(time_start, time_mid, time_end, time_now):
    return (time_start < time_mid) and (time_mid < time_end) and (time_end < time_now)

def savedata(Xtrain_enc,Xtrain_dec_in,Ytrain_tgt,Xval_enc,Xval_dec_in,Yval_tgt,TRAINING_DATA_PATH):
    try:
        data_dir = os.path.dirname(TRAINING_DATA_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        print(f"Saving generated training data to: {TRAINING_DATA_PATH}")
        np.savez_compressed(TRAINING_DATA_PATH,
                 Xtrain_enc=Xtrain_enc, Xtrain_dec_in=Xtrain_dec_in, Ytrain_tgt=Ytrain_tgt,
                 Xval_enc=Xval_enc, Xval_dec_in=Xval_dec_in, Yval_tgt=Yval_tgt)
    except Exception as e:
        print(f"Error saving data to {TRAINING_DATA_PATH}: {e}")

def gen_data(hist_data_list, expected_data_arr):
    enc = np.concatenate(hist_data_list, axis=1)
    tgt = expected_data_arr
    num_features_psymbol = expected_data_arr.shape[1]
    sos_token = np.zeros((1, num_features_psymbol), dtype=expected_data_arr.dtype)
    shifted_target = expected_data_arr[:-1, :]
    dec_in = np.vstack((sos_token, shifted_target))
    return enc, dec_in, tgt

def process_bars_for_symbol(bars_for_symbol_list, target_sequence_length):
    raw_symbol_data_list = []
    for bar in bars_for_symbol_list:
        ts = bar.timestamp.replace(tzinfo=None) if bar.timestamp.tzinfo else bar.timestamp
        raw_symbol_data_list.append([
            mdates.date2num(ts), bar.open, bar.close,
            bar.high, bar.low, bar.volume
        ])
    if not raw_symbol_data_list: return np.full((target_sequence_length, 6), np.nan, dtype=float)
    raw_symbol_data = np.array(raw_symbol_data_list, dtype=float)
    current_len = len(raw_symbol_data)
    if current_len == target_sequence_length: return raw_symbol_data
    if current_len == 0: return np.full((target_sequence_length, 6), np.nan, dtype=float)
    if current_len == 1: return np.tile(raw_symbol_data[0], (target_sequence_length, 1))
    new_indices_float = np.linspace(0, current_len - 1, target_sequence_length)
    original_indices = np.arange(current_len)
    interpolated_resampled_data = np.zeros((target_sequence_length, raw_symbol_data.shape[1]), dtype=float)
    for col_idx in range(raw_symbol_data.shape[1]):
        original_values_col = raw_symbol_data[:, col_idx]
        interpolated_resampled_values_col = np.interp(new_indices_float, original_indices, original_values_col)
        interpolated_resampled_data[:, col_idx] = interpolated_resampled_values_col
    return interpolated_resampled_data

def get_stock_data_batch(client, symbols_list, start_dt, end_dt, target_sequence_length, timeframe_unit):
    req = StockBarsRequest(
        symbol_or_symbols=symbols_list, timeframe=TimeFrame(1, timeframe_unit),
        start=start_dt, end=end_dt, adjustment='split'
    )
    stockbars_multisymbol = client.get_stock_bars(req)
    processed_data_map = {}
    for symbol in symbols_list:
        bars_for_current_symbol = stockbars_multisymbol.data.get(symbol, [])
        processed_data = process_bars_for_symbol(bars_for_current_symbol, target_sequence_length)
        processed_data_map[symbol] = processed_data
    return processed_data_map


def get_training_data():
    # Access config values
    data_cfg = config["data"] # Convenience
    N_total_samples = data_cfg["total_samples"]
    sequence_length = data_cfg["sequence_length"]
    Timedelta_config_datetime = data_cfg["timedelta_config"]["datetime"]
    Timedelta_config_alpaca = data_cfg["timedelta_config"]["alpaca"]
    VAL_SPLIT_RATIO = data_cfg["val_split_ratio"]
    data_save_path = data_cfg["data_file_path"]
    symbols_for_encoder = data_cfg["symbols"]
    psymbol = data_cfg["primary_symbol"]
    force_regenerate = data_cfg.get("force_regenerate_data", False) # Use .get for graceful fallback

    # --- Attempt to load data first ---
    if not force_regenerate and os.path.exists(data_save_path):
        try:
            print(f"Loading training data from: {data_save_path}")
            data_loaded = np.load(data_save_path)
            Xtrain_enc_t = torch.tensor(data_loaded['Xtrain_enc'], dtype=torch.float32)
            Xtrain_dec_in_t = torch.tensor(data_loaded['Xtrain_dec_in'], dtype=torch.float32)
            Ytrain_tgt_t = torch.tensor(data_loaded['Ytrain_tgt'], dtype=torch.float32)
            Xval_enc_t = torch.tensor(data_loaded['Xval_enc'], dtype=torch.float32)
            Xval_dec_in_t = torch.tensor(data_loaded['Xval_dec_in'], dtype=torch.float32)
            Yval_tgt_t = torch.tensor(data_loaded['Yval_tgt'], dtype=torch.float32)
            
            # Basic check: verify if loaded data dimensions match current config (optional but good)
            # This check is for total_samples approximately. Split ratio might differ slightly.
            # More robust would be to save config with data and compare.
            # For now, let's check if the sequence length matches.
            expected_total_samples_in_file = Xtrain_enc_t.shape[0] + Xval_enc_t.shape[0]
            if Xtrain_enc_t.shape[1] == sequence_length and Xval_enc_t.shape[1] == sequence_length :
                 print(f"Data successfully loaded. Shapes: Train Enc {Xtrain_enc_t.shape}, Val Enc {Xval_enc_t.shape}")
                 # You might also want to check if N_total_samples matches approximately,
                 # but this can be tricky if the saved file was generated with a different total.
                 # For simplicity, we assume if it loads and seq_len matches, it's usable.
                 return (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), \
                        (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)
            else:
                print(f"Loaded data sequence length ({Xtrain_enc_t.shape[1]}) does not match config ({sequence_length}). Regenerating data.")

        except Exception as e:
            print(f"Error loading data from {data_save_path}: {e}. Regenerating data.")
    
    if force_regenerate:
        print("Forcing data regeneration as per config.")
    else:
        print(f"Data file not found at {data_save_path} or regeneration forced. Generating new data...")

    # --- Data Generation (if not loaded or forced) ---
    api_key = config["api_keys"]["alpaca_api_key_id"]
    secret_key = config["api_keys"]["alpaca_api_secret_key"]

    if not api_key or not secret_key:
        print("Error: Alpaca API credentials not found.")
        empty_tensor = torch.empty(0)
        return (empty_tensor, empty_tensor, empty_tensor), (empty_tensor, empty_tensor, empty_tensor)
    
    client = StockHistoricalDataClient(api_key, secret_key, raw_data=False)

    all_sequences_encoder = []
    all_sequences_decoder_input = []
    all_sequences_target = []
    
    now = datetime.now()
    skipped_due_to_nans = 0
    skipped_due_to_invalid_config = 0
    
    print(f"Attempting to generate {N_total_samples} valid samples...")
    generated_count = 0
    attempt_count = 0
    max_attempts = N_total_samples * 10 

    while generated_count < N_total_samples and attempt_count < max_attempts:
        attempt_count += 1
        offset_val = np.random.randint(2, 61) 
        if offset_val <= 1: duration_val = 1
        else: duration_val = np.random.randint(1, offset_val)

        seq_end_dt = now - timedelta(**{Timedelta_config_datetime: int(offset_val)})
        hist_encoder_len_units = sequence_length
        seq_start_dt = seq_end_dt - timedelta(**{Timedelta_config_datetime: int(hist_encoder_len_units)})
        pred_end_dt_for_fetch = seq_end_dt + timedelta(**{Timedelta_config_datetime: int(duration_val)})

        MAX_YEARS_LOOKBACK = data_cfg.get("max_years_lookback", 7)
        if (now - seq_start_dt).days > MAX_YEARS_LOOKBACK * 365.25:
            skipped_due_to_invalid_config +=1
            continue
            
        if not validconfig(seq_start_dt, seq_end_dt, pred_end_dt_for_fetch, now):
            skipped_due_to_invalid_config += 1
            continue

        hist_data_map_raw = get_stock_data_batch(client, symbols_for_encoder,
                                                 start_dt=seq_start_dt, end_dt=seq_end_dt,
                                                 target_sequence_length=sequence_length,
                                                 timeframe_unit=Timedelta_config_alpaca)
        
        hist_data_list_ordered = []
        possible_nan_in_hist = False
        for symbol_item in symbols_for_encoder:
            h_data = hist_data_map_raw.get(symbol_item)
            if h_data is None or np.isnan(h_data).any():
                possible_nan_in_hist = True
                break
            hist_data_list_ordered.append(h_data)
        
        if possible_nan_in_hist:
            skipped_due_to_nans += 1
            continue

        psymbol_bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=[psymbol], timeframe=TimeFrame(1, Timedelta_config_alpaca),
            start=seq_end_dt, end=pred_end_dt_for_fetch, adjustment='split'
        )).data.get(psymbol, [])
        expected_data_raw = process_bars_for_symbol(psymbol_bars, sequence_length)

        if np.isnan(expected_data_raw).any():
            skipped_due_to_nans += 1
            continue
        
        enc, dec_in, tgt = gen_data(hist_data_list_ordered, expected_data_raw)
        all_sequences_encoder.append(enc)
        all_sequences_decoder_input.append(dec_in)
        all_sequences_target.append(tgt)
        generated_count += 1
        if generated_count % (N_total_samples // 10 if N_total_samples >=10 else 1) == 0 and generated_count > 0 :
            print(f"Generated {generated_count}/{N_total_samples} samples...")
            
    if generated_count == 0:
        print("Could not generate any valid training samples.")
        empty_tensor = torch.empty(0)
        return (empty_tensor, empty_tensor, empty_tensor), (empty_tensor, empty_tensor, empty_tensor)

    print(f"\n--- Data Generation Summary ---")
    print(f"Total samples generated: {generated_count} / {N_total_samples} requested")
    print(f"Skipped due to NaNs: {skipped_due_to_nans}")
    print(f"Skipped due to invalid date config: {skipped_due_to_invalid_config}")
    print(f"Total attempts made: {attempt_count} / {max_attempts}")
    print(f"-----------------------------\n")

    X_enc_np = np.array(all_sequences_encoder)
    X_dec_in_np = np.array(all_sequences_decoder_input)
    Y_tgt_np = np.array(all_sequences_target)
    
    indices = np.arange(len(X_enc_np))
    np.random.shuffle(indices)
    X_enc_np, X_dec_in_np, Y_tgt_np = X_enc_np[indices], X_dec_in_np[indices], Y_tgt_np[indices]

    valsplit_index = int(len(X_enc_np) * (1 - VAL_SPLIT_RATIO))
    Xtrain_enc, Xval_enc = X_enc_np[:valsplit_index], X_enc_np[valsplit_index:]
    Xtrain_dec_in, Xval_dec_in = X_dec_in_np[:valsplit_index], X_dec_in_np[valsplit_index:]
    Ytrain_tgt, Yval_tgt = Y_tgt_np[:valsplit_index], Y_tgt_np[valsplit_index:]

    if data_save_path:
        savedata(Xtrain_enc, Xtrain_dec_in, Ytrain_tgt, Xval_enc, Xval_dec_in, Yval_tgt, TRAINING_DATA_PATH=data_save_path)

    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.float32)
    Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.float32)
    Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.float32)
    Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.float32)
    Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.float32)
    Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.float32)

    return (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), \
           (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)