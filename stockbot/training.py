import os
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest

from config import config
# Assuming EncoderDecoderModel is imported from network by main.py

def save_encoder_decoder_model(model, optimizer, epoch, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config_model_params': config["model"] 
    }
    torch.save(checkpoint, model_path)
    print(f"EncoderDecoderModel checkpoint saved to {model_path}")


def validconfig(time_start, time_mid, time_end, time_now):
    return (time_start < time_mid) and (time_mid < time_end) and (time_end < time_now)

def savedata(Xtrain_enc,Xtrain_dec_in,Ytrain_tgt,Xval_enc,Xval_dec_in,Yval_tgt,data_file_path_arg):
    try:
        data_dir = os.path.dirname(data_file_path_arg)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        print(f"Saving generated training data to: {data_file_path_arg}")
        np.savez_compressed(data_file_path_arg,
                 Xtrain_enc=Xtrain_enc, Xtrain_dec_in=Xtrain_dec_in, Ytrain_tgt=Ytrain_tgt,
                 Xval_enc=Xval_enc, Xval_dec_in=Xval_dec_in, Yval_tgt=Yval_tgt)
    except Exception as e:
        print(f"Error saving data to {data_file_path_arg}: {e}")

def gen_data(hist_data_list, expected_data_arr): 
    enc = np.concatenate(hist_data_list, axis=1) 
    tgt = expected_data_arr 
    num_features_psymbol = expected_data_arr.shape[1]
    sos_token = np.zeros((1, num_features_psymbol), dtype=expected_data_arr.dtype)
    shifted_target = expected_data_arr[:-1, :]
    dec_in = np.vstack((sos_token, shifted_target)) 
    return enc, dec_in, tgt

def process_bars_for_symbol(bars_for_symbol_list, target_sequence_length):
    num_features_per_symbol = config["model"]["decoder_embed_dim"] # Features for one symbol
    raw_symbol_data_list = []
    for bar in bars_for_symbol_list:
        ts = bar.timestamp.replace(tzinfo=None) if bar.timestamp.tzinfo else bar.timestamp
        raw_symbol_data_list.append([
            mdates.date2num(ts), bar.open, bar.close, bar.high, bar.low
        ])
    if not raw_symbol_data_list: return np.full((target_sequence_length, num_features_per_symbol), np.nan, dtype=float)
    raw_symbol_data = np.array(raw_symbol_data_list, dtype=float)
    if raw_symbol_data.size > 0 and raw_symbol_data.shape[1] != num_features_per_symbol:
        print(f"Warning: Raw symbol data feature count {raw_symbol_data.shape[1]} != decoder_embed_dim {num_features_per_symbol}")
        return np.full((target_sequence_length, num_features_per_symbol), np.nan, dtype=float)
    current_len = len(raw_symbol_data)
    if current_len == target_sequence_length: return raw_symbol_data
    if current_len == 0: return np.full((target_sequence_length, num_features_per_symbol), np.nan, dtype=float)
    if current_len == 1: return np.tile(raw_symbol_data[0], (target_sequence_length, 1))
    if current_len < target_sequence_length:
        padding_needed = target_sequence_length - current_len
        last_valid_row = raw_symbol_data[-1, :]
        padding_array = np.tile(last_valid_row, (padding_needed, 1))
        raw_symbol_data = np.vstack([raw_symbol_data, padding_array])
    return raw_symbol_data

def get_stock_data_batch(client, symbols_list, start_dt, end_dt, target_sequence_length, timeframe_unit):
    req = StockBarsRequest(symbol_or_symbols=symbols_list, timeframe=TimeFrame(1, timeframe_unit), start=start_dt, end=end_dt, adjustment='split')
    stockbars_multisymbol = client.get_stock_bars(req)
    processed_data_map = {}
    for symbol in symbols_list:
        bars_for_current_symbol = stockbars_multisymbol.data.get(symbol, [])
        bars_for_current_symbol.sort(key=lambda bar: bar.timestamp)
        processed_data = process_bars_for_symbol(bars_for_current_symbol, target_sequence_length)
        processed_data_map[symbol] = processed_data
    return processed_data_map

def get_training_data():
    sequence_length = config["data"]["sequence_length"]
    data_save_path = config["data"]["data_file_path"]
    # Ensure primary_symbol is in symbols_for_encoder for Scenario 3.A
    symbols_for_encoder = config["data"]["symbols"] # This list is now guaranteed by config.py to include primary
    psymbol = config["data"]["primary_symbol"]
    # ... (rest of variable loading from config) ...
    Timedelta_config_datetime = config["data"]["timedelta_config"]["datetime"]
    Timedelta_config_alpaca = config["data"]["timedelta_config"]["alpaca"]
    VAL_SPLIT_RATIO = config["data"]["val_split_ratio"]
    N_total_samples = config["data"]["total_samples"]
    force_regenerate = config["data"].get("force_regenerate_data", False)
    try_load_data_flag = config["data"].get("try_load_data", True)

    if try_load_data_flag and not force_regenerate and os.path.exists(data_save_path):
        try:
            print(f"Loading training data from: {data_save_path}")
            data_loaded = np.load(data_save_path)
            Xtrain_enc_t = torch.tensor(data_loaded['Xtrain_enc'], dtype=torch.float32)
            Xtrain_dec_in_t = torch.tensor(data_loaded['Xtrain_dec_in'], dtype=torch.float32)
            Ytrain_tgt_t = torch.tensor(data_loaded['Ytrain_tgt'], dtype=torch.float32)
            Xval_enc_t = torch.tensor(data_loaded['Xval_enc'], dtype=torch.float32)
            Xval_dec_in_t = torch.tensor(data_loaded['Xval_dec_in'], dtype=torch.float32)
            Yval_tgt_t = torch.tensor(data_loaded['Yval_tgt'], dtype=torch.float32)
            valid_train = (Xtrain_enc_t.shape[0] > 0 and Xtrain_enc_t.shape[1] == sequence_length and Xtrain_dec_in_t.shape[1] == sequence_length and Ytrain_tgt_t.shape[1] == sequence_length)
            valid_val = (Xval_enc_t.shape[0] == 0 or (Xval_enc_t.shape[0] > 0 and Xval_enc_t.shape[1] == sequence_length and Xval_dec_in_t.shape[1] == sequence_length and Yval_tgt_t.shape[1] == sequence_length))
            if valid_train and valid_val:
                 print(f"Data successfully loaded. Shapes: Train Enc {Xtrain_enc_t.shape}, Train Dec_In {Xtrain_dec_in_t.shape}, Val Enc {Xval_enc_t.shape}, Val Dec_In {Xval_dec_in_t.shape}")
                 return (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)
            else: # ... (error/regen message)
                if Xtrain_enc_t.shape[0] == 0: print("Loaded data is empty.")
                else: print(f"Loaded data sequence length or structure mismatch. Regenerating data.")
        except Exception as e: print(f"Error loading data from {data_save_path}: {e}. Regenerating data.")
    
    # ... (rest of data generation preamble: api keys, client, counters) ...
    if force_regenerate: print("Forcing data regeneration as per config.")
    elif not try_load_data_flag: print("`try_load_data` is false. Generating new data...")
    else: print(f"Data file not found at {data_save_path}. Generating new data...")

    api_key = config["api_keys"]["alpaca_api_key_id"]
    secret_key = config["api_keys"]["alpaca_api_secret_key"]
    if not api_key or not secret_key: 
        empty_tensor = torch.empty(0)
        return (empty_tensor, empty_tensor, empty_tensor), (empty_tensor, empty_tensor, empty_tensor)

    client = StockHistoricalDataClient(api_key, secret_key, raw_data=False)
    all_sequences_encoder_input, all_sequences_decoder_input, all_sequences_decoder_target = [], [], []
    now = datetime.now()
    skipped_due_to_nans, skipped_due_to_invalid_config = 0,0
    print(f"Attempting to generate {N_total_samples} valid samples...")
    generated_count, attempt_count, max_attempts = 0,0, N_total_samples * 10

    while generated_count < N_total_samples and attempt_count < max_attempts:
        attempt_count += 1
        hist_encoder_len_units = sequence_length 
        pred_decoder_len_units = sequence_length 
        offset_val = np.random.randint(pred_decoder_len_units + 5, pred_decoder_len_units + 300)
        pred_end_dt = now - timedelta(**{Timedelta_config_datetime: int(offset_val)})
        pred_start_dt = pred_end_dt - timedelta(**{Timedelta_config_datetime: int(pred_decoder_len_units)})
        hist_start_dt = pred_start_dt - timedelta(**{Timedelta_config_datetime: int(hist_encoder_len_units)})
        
        MAX_YEARS_LOOKBACK = config["data"].get("max_years_lookback", 7)
        if (now - hist_start_dt).days > MAX_YEARS_LOOKBACK * 365.25: skipped_due_to_invalid_config +=1; continue
        if not validconfig(hist_start_dt, pred_start_dt, pred_end_dt, now): skipped_due_to_invalid_config += 1; continue

        hist_data_map_all_symbols = get_stock_data_batch(client, symbols_for_encoder, hist_start_dt, pred_start_dt, sequence_length, Timedelta_config_alpaca)
        hist_data_list_ordered_for_enc = []
        possible_nan_in_hist = False
        for symbol_item in symbols_for_encoder: # symbols_for_encoder now includes psymbol
            h_data = hist_data_map_all_symbols.get(symbol_item)
            if h_data is None or np.isnan(h_data).any(): possible_nan_in_hist = True; break
            hist_data_list_ordered_for_enc.append(h_data)
        if possible_nan_in_hist: skipped_due_to_nans += 1; continue

        psymbol_bars_future = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=[psymbol], timeframe=TimeFrame(1, Timedelta_config_alpaca), start=pred_start_dt, end=pred_end_dt, adjustment='split')).data.get(psymbol, [])
        psymbol_bars_future.sort(key=lambda bar: bar.timestamp)
        expected_data_raw_psymbol = process_bars_for_symbol(psymbol_bars_future, sequence_length)
        if np.isnan(expected_data_raw_psymbol).any(): skipped_due_to_nans += 1; continue
        
        enc_input, dec_in_input, dec_tgt_output = gen_data(hist_data_list_ordered_for_enc, expected_data_raw_psymbol)
        all_sequences_encoder_input.append(enc_input); all_sequences_decoder_input.append(dec_in_input); all_sequences_decoder_target.append(dec_tgt_output)
        generated_count += 1
        if generated_count % (N_total_samples // 10 if N_total_samples >=10 else 1) == 0 and generated_count > 0: print(f"Generated {generated_count}/{N_total_samples} samples...")
            
    if generated_count == 0: # ... (error handling)
        empty_tensor = torch.empty(0)
        return (empty_tensor, empty_tensor, empty_tensor), (empty_tensor, empty_tensor, empty_tensor)

    print(f"\n--- Data Generation Summary ---") # ... (summary print)
    print(f"Total samples generated: {generated_count} / {N_total_samples} requested") # ...
    # ... (numpy conversion, shuffle, split, save) ...
    X_enc_np = np.array(all_sequences_encoder_input); X_dec_in_np = np.array(all_sequences_decoder_input); Y_tgt_np = np.array(all_sequences_decoder_target)
    indices = np.arange(len(X_enc_np)); np.random.shuffle(indices)
    X_enc_np, X_dec_in_np, Y_tgt_np = X_enc_np[indices], X_dec_in_np[indices], Y_tgt_np[indices]
    valsplit_index = int(len(X_enc_np) * (1 - VAL_SPLIT_RATIO))
    Xtrain_enc, Xval_enc = X_enc_np[:valsplit_index], X_enc_np[valsplit_index:]
    Xtrain_dec_in, Xval_dec_in = X_dec_in_np[:valsplit_index], X_dec_in_np[valsplit_index:]
    Ytrain_tgt, Yval_tgt = Y_tgt_np[:valsplit_index], Y_tgt_np[valsplit_index:]
    savedata(Xtrain_enc, Xtrain_dec_in, Ytrain_tgt, Xval_enc, Xval_dec_in, Yval_tgt, data_file_path_arg=data_save_path)
    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.float32); Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.float32); Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.float32)
    Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.float32); Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.float32); Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.float32)
    return (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)


def plot_predictions(model, device, data_for_plot, psymbol_name, num_samples_to_plot): # Renamed arg
    if num_samples_to_plot == 0: print("Number of samples to plot is 0. Skipping plotting."); return
    model.eval()
    features_to_plot_info = { "Open": {"index": 1, "color": "blue"}, "Close": {"index": 2, "color": "green"}, "High": {"index": 3, "color": "red"}, "Low": {"index": 4, "color": "purple"}}
    plotted_count = 0
    X_encoder_input_batch, Y_actual_targets_batch = data_for_plot # Unpack

    if X_encoder_input_batch.ndim == 2: X_encoder_input_batch = X_encoder_input_batch.unsqueeze(0)
    if Y_actual_targets_batch.ndim == 2: Y_actual_targets_batch = Y_actual_targets_batch.unsqueeze(0)
    X_encoder_input_batch = X_encoder_input_batch.to(device)
    Y_actual_targets_batch = Y_actual_targets_batch.to(device)
    num_samples_in_batch = X_encoder_input_batch.size(0)

    with torch.no_grad():
        all_predicted_sequences = model.predict_autoregressive(X_encoder_input_batch, sos_token_val=0.0, device=device)

    for i in range(num_samples_in_batch):
        if plotted_count >= num_samples_to_plot: break
        plt.figure(figsize=(15, 8))
        actual_targets_single = Y_actual_targets_batch[i, :, :]; predicted_sequence_single = all_predicted_sequences[i, :, :]
        timestamps_mdates = actual_targets_single[:, 0].cpu().numpy(); timestamps_datetime = [mdates.num2date(ts) for ts in timestamps_mdates]
        for feature_name, info in features_to_plot_info.items():
            feature_idx, color = info["index"], info["color"]
            if feature_idx >= actual_targets_single.shape[-1] or feature_idx >= predicted_sequence_single.shape[-1]:
                print(f"Warning: Feature index {feature_idx} for {feature_name} out of bounds. Skipping."); continue
            actual_vals = actual_targets_single[:, feature_idx].cpu().numpy(); predicted_vals = predicted_sequence_single[:, feature_idx].cpu().numpy()
            if np.isnan(predicted_vals).any() or np.isinf(predicted_vals).any():
                print(f"WARNING: NaN/Inf in predicted {feature_name} for sample {i+1}. Plotting may be incomplete.")
                predicted_vals = np.nan_to_num(predicted_vals, nan=np.nanmean(actual_vals), posinf=np.nanmax(actual_vals), neginf=np.nanmin(actual_vals)) # Use nan-aware fallbacks
            plt.plot(timestamps_datetime, actual_vals, label=f'Actual {feature_name}', color=color, linestyle='-', marker='.', markersize=3)
            plt.plot(timestamps_datetime, predicted_vals, label=f'Predicted {feature_name} (Autoregressive)', color=color, linestyle='--')
        plt.title(f'{psymbol_name} - Autoregressive Predictions vs Actual (Test Sample {plotted_count+1})'); plt.xlabel('Time'); plt.ylabel('Price Value')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()
        plotted_count += 1

def training(model, optimizer_state_dict, start_epoch_arg):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] != "cpu" else "cpu") if config["training"]["device"] == "auto" else torch.device(config["training"]["device"])
    print(f"Using device: {DEVICE}"); model.to(DEVICE)
    print("Getting training data...")
    (Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t), (Xval_enc_t, Xval_dec_in_t, Yval_tgt_t) = get_training_data()
    if Xtrain_enc_t.nelement() == 0 or Xtrain_dec_in_t.nelement() == 0: print("No training data. Exiting."); return
    Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t = Xtrain_enc_t.to(DEVICE), Xtrain_dec_in_t.to(DEVICE), Ytrain_tgt_t.to(DEVICE)
    Xval_enc_t, Xval_dec_in_t, Yval_tgt_t = Xval_enc_t.to(DEVICE), Xval_dec_in_t.to(DEVICE), Yval_tgt_t.to(DEVICE)
    print(f"Train shapes: Enc:{Xtrain_enc_t.shape}, DecIn:{Xtrain_dec_in_t.shape}, Tgt:{Ytrain_tgt_t.shape}")
    print(f"Val shapes: Enc:{Xval_enc_t.shape}, DecIn:{Xval_dec_in_t.shape}, Tgt:{Yval_tgt_t.shape}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"].get("weight_decay", 1e-2))
    if optimizer_state_dict:
        try: optimizer.load_state_dict(optimizer_state_dict); print("Optimizer state loaded.")
        except Exception as e: print(f"Could not load optimizer state: {e}. Initializing new optimizer.")
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["training"].get("scheduler_patience", 5))
    print("\nStarting training..."); NUM_EPOCHS, BATCH_SIZE_TRAIN, CLIP_GRAD_NORM = config["training"]["num_epochs"], config["training"]["batch_size"], config["training"]["clip_grad_norm"]
    best_val_loss = float('inf'); model_save_path = config["data"]["model_file_path"]

    for epoch in range(start_epoch_arg, NUM_EPOCHS):
        model.train(); epoch_loss = 0.0
        permutation = torch.randperm(Xtrain_enc_t.size(0))
        for i in range(0, Xtrain_enc_t.size(0), BATCH_SIZE_TRAIN):
            optimizer.zero_grad(); indices = permutation[i : i + BATCH_SIZE_TRAIN]
            batch_x_enc, batch_x_dec_in, batch_y_tgt = Xtrain_enc_t[indices], Xtrain_dec_in_t[indices], Ytrain_tgt_t[indices]
            predictions = model(batch_x_enc, batch_x_dec_in) # Teacher forcing for training
            loss = criterion(predictions, batch_y_tgt); loss.backward()
            if CLIP_GRAD_NORM is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            optimizer.step(); epoch_loss += loss.item() * batch_x_enc.size(0)
        epoch_loss /= Xtrain_enc_t.size(0)

        val_loss = 0.0
        if Xval_enc_t.size(0) > 0:
            model.eval()
            with torch.no_grad():
                val_perm = torch.randperm(Xval_enc_t.size(0))
                for i_v in range(0, Xval_enc_t.size(0), BATCH_SIZE_TRAIN):
                    val_idx = val_perm[i_v : i_v + BATCH_SIZE_TRAIN]
                    b_x_v_enc, b_x_v_dec_in, b_y_v_tgt = Xval_enc_t[val_idx], Xval_dec_in_t[val_idx], Yval_tgt_t[val_idx]
                    val_preds = model(b_x_v_enc, b_x_v_dec_in) # Teacher forcing for val loss
                    v_loss = criterion(val_preds, b_y_v_tgt); val_loss += v_loss.item() * b_x_v_enc.size(0)
            val_loss /= Xval_enc_t.size(0)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss; save_encoder_decoder_model(model, optimizer, epoch + 1, model_save_path)
                print(f"New best model saved with val_loss: {best_val_loss:.6f}")
            scheduler.step(val_loss)
        else: # No validation
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {epoch_loss:.6f} (No Val)")
            if epoch_loss != float('inf') and epoch_loss != float('nan'): scheduler.step(epoch_loss)
    print("Training finished."); save_encoder_decoder_model(model, optimizer, NUM_EPOCHS, model_save_path)

    if Xval_enc_t.size(0) > 0:
        print("\nPlotting autoregressive predictions..."); model_to_plot = model
        num_plot = min(config["plotting"]["num_samples_to_plot"], Xval_enc_t.size(0))
        if num_plot > 0:
            plot_predictions(model_to_plot, DEVICE, (Xval_enc_t[:num_plot], Yval_tgt_t[:num_plot]), config["data"]["primary_symbol"], num_plot)
    else: print("No validation data to plot.")
    # Diagnostics section is still commented out - requires EncoderDecoderModel to return them
