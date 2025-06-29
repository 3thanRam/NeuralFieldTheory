import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # For softmax in testing
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os # For saving plot

from config import config # Ensure config has enc_seq_len, dec_seq_len etc.

# Assuming these functions are in data_handling.py and are compatible
from data_handling import (
    savedata,
    gen_data_for_model, # Must return (enc_ids, dec_in_ids, dec_tgt_ids)
                        # where enc_ids is enc_seq_len, dec_in/tgt are dec_seq_len
    gen_sequential_ohlc_token_sequence_with_bars,
    save_model,
    tokenize_ohlc_bar_sequentially # Used in testing
)
from datetime import datetime, timedelta

def get_training_data_loader():
    # This function should remain largely the same, but ensure gen_data_for_model
    # produces enc_ids of length config["enc_seq_len"] and
    # dec_in_ids, dec_tgt_ids of length config["dec_seq_len"].
    # Your current gen_data_for_model uses config["max_seq_len"] for both,
    # so if enc_seq_len and dec_seq_len are different from that, it needs adjustment.
    # For now, let's assume config["max_seq_len"] in data_handling.py matches config["enc_seq_len"]
    # AND config["dec_seq_len"] if they are meant to be the same.
    # If they are different, gen_data_for_model needs to be more sophisticated.
    # A common setup: enc_seq_len for context, dec_seq_len for generation.
    # Your current gen_data_for_model where L_tokens = config["max_seq_len"]
    # implies enc_ids and dec_in_ids/dec_tgt_ids are all config["max_seq_len"] long.
    # So set enc_seq_len = config["max_seq_len"] and dec_seq_len = config["max_seq_len"]
    # in config.py if using current gen_data_for_model.

    all_sequences_encoder_input, all_sequences_decoder_input, all_sequences_decoder_target = [], [], []
    print(f"Generating {config['num_training_samples']} tokenized stock samples...")
    generated_samples = 0
    attempts = 0
    max_attempts = config["num_training_samples"] * 5

    while generated_samples < config["num_training_samples"] and attempts < max_attempts:
        attempts += 1
        # gen_data_for_model should return (encoder_tokens, decoder_input_tokens, decoder_target_tokens)
        # Lengths should match config["enc_seq_len"] and config["dec_seq_len"] respectively
        enc_ids, dec_in_ids, dec_tgt_ids = gen_data_for_model()
        if enc_ids is not None:
            # Ensure lengths match config before appending
            if len(enc_ids) == config["enc_seq_len"] and \
               len(dec_in_ids) == config["dec_seq_len"] and \
               len(dec_tgt_ids) == config["dec_seq_len"]:
                all_sequences_encoder_input.append(enc_ids)
                all_sequences_decoder_input.append(dec_in_ids)
                all_sequences_decoder_target.append(dec_tgt_ids)
                generated_samples += 1
            else:
                print(f"Warning: gen_data_for_model returned mismatched lengths. Skipping sample.")
                print(f"Enc: {len(enc_ids) if enc_ids is not None else 'None'} vs {config['enc_seq_len']}, Dec_in: {len(dec_in_ids) if dec_in_ids is not None else 'None'} vs {config['dec_seq_len']}")


        if attempts > 0 and attempts % (max_attempts // 10 if max_attempts > 10 else 1) == 0 :
            print(f"Attempted {attempts}, Generated {generated_samples} valid samples...")
    
    if generated_samples < config["num_training_samples"]:
        print(f"Warning: Could only generate {generated_samples} out of {config['num_training_samples']} desired samples.")
        if generated_samples == 0:
             raise RuntimeError("Failed to generate ANY valid training samples. Check gen_data_for_model and enc/dec_seq_len in config.")
        if generated_samples < config["batch_size"]:
            print("Warning: Generated fewer samples than batch_size.")


    X_enc_np = np.array(all_sequences_encoder_input)
    X_dec_in_np = np.array(all_sequences_decoder_input)
    Y_tgt_np = np.array(all_sequences_decoder_target)

    # ... (rest of shuffling and splitting logic is fine) ...
    indices = np.arange(len(X_enc_np))
    np.random.shuffle(indices)
    X_enc_np, X_dec_in_np, Y_tgt_np = X_enc_np[indices], X_dec_in_np[indices], Y_tgt_np[indices]

    valsplit_index = int(len(X_enc_np) * (1 - config["VAL_SPLIT_RATIO"]))
    if len(X_enc_np) > 0: # Ensure valsplit_index is valid
        if valsplit_index == 0 : valsplit_index = 1 if len(X_enc_np) > 1 else 0
        if valsplit_index == len(X_enc_np) : valsplit_index = len(X_enc_np) -1 if len(X_enc_np) > 1 else len(X_enc_np)
    else: # No data generated
        return (np.array([]), np.array([]), np.array([])), (np.array([]), np.array([]), np.array([]))


    Xtrain_enc, Xval_enc = X_enc_np[:valsplit_index], X_enc_np[valsplit_index:]
    Xtrain_dec_in, Xval_dec_in = X_dec_in_np[:valsplit_index], X_dec_in_np[valsplit_index:]
    Ytrain_tgt, Yval_tgt = Y_tgt_np[:valsplit_index], Y_tgt_np[valsplit_index:]

    if len(Xtrain_enc) > 0 :
         savedata(Xtrain_enc, Xtrain_dec_in, Ytrain_tgt, Xval_enc, Xval_dec_in, Yval_tgt, data_file_path_arg=config["data_save_path"])
    else:
        print("Warning: No training data to save after splitting.")
        if generated_samples > 0 : # If samples were generated but all went to val or were too few
             raise RuntimeError("No training data available after split. Check VAL_SPLIT_RATIO or increase num_training_samples.")
    
    if len(Xval_enc) == 0 and len(Xtrain_enc) > 0 :
        print("Warning: No validation data after split. Using a small part of training data for validation.")
        num_val_fallback = max(1, int(0.05 * len(Xtrain_enc))) 
        if len(Xtrain_enc) > num_val_fallback:
            Xval_enc, Xval_dec_in, Yval_tgt = Xtrain_enc[-num_val_fallback:], Xtrain_dec_in[-num_val_fallback:], Ytrain_tgt[-num_val_fallback:]
            Xtrain_enc, Xtrain_dec_in, Ytrain_tgt = Xtrain_enc[:-num_val_fallback], Xtrain_dec_in[:-num_val_fallback], Ytrain_tgt[:-num_val_fallback]
    return (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt)


def training(model, optimizer, start_epoch):
    device = torch.device(config.get("device", "cpu"))
    model.to(device)
    print(f"Training on device: {device}")

    print("Preparing training data...")
    if config.get("load_training_data", False) and os.path.exists(config["data_save_path"]):
        try:
            data_loaded = np.load(config["data_save_path"])
            Xtrain_enc, Xtrain_dec_in, Ytrain_tgt = data_loaded['Xtrain_enc'], data_loaded['Xtrain_dec_in'], data_loaded['Ytrain_tgt']
            Xval_enc, Xval_dec_in, Yval_tgt = data_loaded['Xval_enc'], data_loaded['Xval_dec_in'], data_loaded['Yval_tgt']
            print("Loaded training data from file.")
            if Xtrain_enc.size == 0 or Xtrain_dec_in.size == 0:
                raise ValueError("Loaded training data arrays are empty.")
            # Verify sequence lengths
            if Xtrain_enc.shape[1] != config["enc_seq_len"] or \
               Xtrain_dec_in.shape[1] != config["dec_seq_len"] or \
               Ytrain_tgt.shape[1] != config["dec_seq_len"]:
                print("Warning: Loaded data sequence lengths mismatch config. Regenerating.")
                raise ValueError("Sequence length mismatch.")
        except (FileNotFoundError, ValueError, KeyError) as e: # Added KeyError for missing keys in npz
            print(f"Data file issue ({e}) at {config['data_save_path']}. Generating new data.")
            (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt) = get_training_data_loader()
    else:
        (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt) = get_training_data_loader()

    if Xtrain_enc.size == 0 or Xtrain_dec_in.size == 0:
        raise RuntimeError("Failed to obtain any training data. Aborting.")

    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.long)
    Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.long)
    Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.long)
    
    train_dataset = TensorDataset(Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    val_loader = None
    if Xval_enc.size > 0 and Xval_dec_in.size > 0 : # Check if val data actually exists
        Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.long)
        Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.long)
        Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.long)
        val_dataset = TensorDataset(Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    else:
        print("Warning: No validation data available. Skipping validation loop.")

    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_idx"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get("scheduler_patience", 3))
    
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config["num_epoch"]):
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (batch_enc_in, batch_dec_in, batch_tgt) in enumerate(train_loader):
            batch_enc_in = batch_enc_in.to(device)
            batch_dec_in = batch_dec_in.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            
            # Model now takes both encoder and decoder inputs
            predictions_logits = model(enc_input_ids=batch_enc_in, dec_input_ids=batch_dec_in)
            
            loss = criterion(
                predictions_logits.reshape(-1, config["vocab_size"]),
                batch_tgt.reshape(-1)
            )
            
            loss.backward()
            if config.get("clip_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_grad_norm"])
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        avg_epoch_val_loss = float('nan')

        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_enc_in_val, batch_dec_in_val, batch_tgt_val in val_loader:
                    batch_enc_in_val = batch_enc_in_val.to(device)
                    batch_dec_in_val = batch_dec_in_val.to(device)
                    batch_tgt_val = batch_tgt_val.to(device)

                    val_preds_logits = model(enc_input_ids=batch_enc_in_val, dec_input_ids=batch_dec_in_val)
                    
                    v_loss = criterion(
                        val_preds_logits.reshape(-1, config["vocab_size"]),
                        batch_tgt_val.reshape(-1)
                    )
                    epoch_val_loss += v_loss.item()
            
            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            scheduler.step(avg_epoch_val_loss)

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                save_model(model, optimizer, epoch + 1, config["model_save_path"]) # Consider passing best_val_loss

        print(f"Epoch [{epoch+1}/{config['num_epoch']}], LR: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}")
    print("Training finished.")


def testing(model_instance):
    device = torch.device(config.get("device", "cpu"))
    model_instance.to(device); model_instance.eval()
    print("Generating a test sample for plotting with sequential OHLC feature tokens...")

    tokens_per_day = config["tokens_per_day"]
    
    # Encoder part uses enc_seq_len for days, Decoder part uses dec_seq_len for days
    # The prompt should be based on enc_seq_len
    enc_days = config["enc_seq_len"] // tokens_per_day
    # The prediction length should be based on dec_seq_len
    predict_days = config["dec_seq_len"] // tokens_per_day

    total_days_to_fetch = enc_days # We only need enc_days for the prompt for the encoder
                                   # And predict_days for the target raw data comparison
    
    now = datetime.now()
    # Fetch enough data to cover the encoder input period and the period we want to predict and compare against
    end_dt_for_series = now - timedelta(days=np.random.randint(enc_days + predict_days + 90, enc_days + predict_days + 730))

    # Get tokens for encoder input period AND raw bars for the target prediction period
    encoder_input_tokens_flat, ohlc_bars_for_encoder_period, bar_before_encoder_period = \
        gen_sequential_ohlc_token_sequence_with_bars(
            config["primary_symbol"], enc_days, end_dt_for_series # Fetch only enc_days for prompt
        )

    if encoder_input_tokens_flat is None:
        print("Failed to generate encoder prompt data for testing plot."); return

    # For target comparison, fetch raw bars for the prediction period
    # The prediction period starts right after the encoder period ends
    target_period_start_dt = ohlc_bars_for_encoder_period[-1]['timestamp'].replace(tzinfo=None) + timedelta(days=1) \
                             if ohlc_bars_for_encoder_period else end_dt_for_series + timedelta(days=1)
    
    # We need the bar right before the target period starts for tokenizing its first day's gap
    bar_before_target_period = ohlc_bars_for_encoder_period[-1] if ohlc_bars_for_encoder_period else bar_before_encoder_period

    _, ohlc_bars_target_period, _ = \
        gen_sequential_ohlc_token_sequence_with_bars(
            config["primary_symbol"], predict_days, target_period_start_dt + timedelta(days=predict_days -1),
            # Need to pass the bar before this series for correct first gap
            # This requires modification of gen_sequential_ohlc_token_sequence_with_bars
            # or careful handling of prev_bar if fetching separately.
            # Simpler: just use the last bar of encoder period as prev bar for 1st target day.
        )
    
    if ohlc_bars_target_period is None:
        print("Failed to generate target raw data for testing plot comparison."); return
        
    # Prepare encoder input tensor
    enc_input_ids_tensor = torch.tensor([encoder_input_tokens_flat], dtype=torch.long).to(device)
    # Ensure it's the correct length for the encoder
    if enc_input_ids_tensor.size(1) < config["enc_seq_len"]:
        padding = torch.full((1, config["enc_seq_len"] - enc_input_ids_tensor.size(1)), config["pad_idx"], dtype=torch.long, device=device)
        enc_input_ids_tensor = torch.cat([padding, enc_input_ids_tensor], dim=1) # Left pad
    elif enc_input_ids_tensor.size(1) > config["enc_seq_len"]:
        enc_input_ids_tensor = enc_input_ids_tensor[:, -config["enc_seq_len"]:]


    # --- Autoregressive Generation (Decoder) ---
    generated_tokens_predicted_period_flat = []
    # Decoder starts with SOS token
    current_decoder_input_tokens = [config["sos_token_id"]]
    
    # The number of tokens to predict is predict_days * tokens_per_day, up to dec_seq_len
    num_tokens_to_predict_total = predict_days * tokens_per_day

    last_close_of_prompt = float(ohlc_bars_for_encoder_period[-1]["c"]) if ohlc_bars_for_encoder_period else \
                           (float(bar_before_encoder_period["c"]) if bar_before_encoder_period else 100.0) # Fallback

    with torch.no_grad():
        for i in range(num_tokens_to_predict_total):
            # Prepare decoder input: current generated sequence, padded to dec_seq_len
            dec_input_tensor_unpadded = torch.tensor([current_decoder_input_tokens], dtype=torch.long).to(device)
            
            # Right-pad decoder input to dec_seq_len
            current_dec_len = dec_input_tensor_unpadded.size(1)
            if current_dec_len < config["dec_seq_len"]:
                padding_needed = config["dec_seq_len"] - current_dec_len
                padding_tensor = torch.full((1, padding_needed), config["pad_idx"], dtype=torch.long, device=device)
                dec_input_for_model = torch.cat([dec_input_tensor_unpadded, padding_tensor], dim=1)
            else: # Should only happen if current_dec_len == dec_seq_len
                dec_input_for_model = dec_input_tensor_unpadded[:, :config["dec_seq_len"]] # Ensure it's not longer

            # Model expects (enc_input_ids, dec_input_ids)
            output_logits = model_instance(enc_input_ids=enc_input_ids_tensor,
                                           dec_input_ids=dec_input_for_model)
            
            # Logits for the *next* token are at the position of the *last actual input token*
            # output_logits is (batch, dec_seq_len, vocab_size)
            # The last actual input token was at index current_dec_len - 1
            next_token_logits = output_logits[:, current_dec_len - 1, :] 
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens_predicted_period_flat.append(next_token_id)
            
            if next_token_id == config["eos_token_id"] and (i + 1) % tokens_per_day == 0:
                print(f"Generated <EOS> at end of day { (i+1)//tokens_per_day } features. Stopping generation for this day's features or prematurely.");
                # If you want to stop full generation, break here.
                # If only stopping for this day's set of 4 tokens, you might pad rest of day's tokens and continue for next day.
                # For simplicity, if EOS is generated, we might just append it and continue,
                # or handle it by padding the rest of the day's tokens with PAD and then breaking if total tokens reached.
                # Here, we just append and let the loop continue up to num_tokens_to_predict_total.
            
            current_decoder_input_tokens.append(next_token_id)

            # If current_decoder_input_tokens exceeds dec_seq_len, it means we predicted more than
            # the decoder was trained to handle in one go. This shouldn't happen if num_tokens_to_predict_total <= dec_seq_len.
            # If we want to predict longer than dec_seq_len, we'd need a sliding window for decoder input too.
            # For now, assume num_tokens_to_predict_total <= config["dec_seq_len"]

    # --- Detokenization of Model's Predicted Tokens ---
    rep_vals = config["REPRESENTATIVE_VALUES_FOR_BINS"]
    predicted_ohlc_bars = [] 
    num_predicted_full_days = len(generated_tokens_predicted_period_flat) // tokens_per_day
    current_open_for_model_pred = last_close_of_prompt 
    for day_idx in range(num_predicted_full_days):
        day_tokens = generated_tokens_predicted_period_flat[day_idx*tokens_per_day : (day_idx+1)*tokens_per_day]
        if len(day_tokens) < tokens_per_day: continue
        val_gap = rep_vals["gap"].get(day_tokens[0], 0.0)
        val_uw = rep_vals["upper_wick"].get(day_tokens[1], 0.0)
        val_lw = rep_vals["lower_wick"].get(day_tokens[2], 0.0)
        val_body = rep_vals["body"].get(day_tokens[3], 0.0)
        pred_o = current_open_for_model_pred * (1 + val_gap)
        pred_c = pred_o * (1 + val_body)
        pred_h = val_uw * pred_o + max(pred_o, pred_c)
        pred_l = min(pred_o, pred_c) - val_lw * pred_o
        h_temp = max(pred_o, pred_c, pred_h); l_temp = min(pred_o, pred_c, pred_l)
        predicted_ohlc_bars.append({'open': pred_o, 'high': h_temp, 'low': l_temp, 'close': pred_c})
        current_open_for_model_pred = pred_c

    # --- Tokenize and Detokenize Actual Target Data for Quantized Ground Truth ---
    quantized_ground_truth_tokens_flat = []
    prev_bar_for_gt_tokenization = bar_before_target_period
    for day_idx in range(min(len(ohlc_bars_target_period), predict_days)): # Iterate up to predict_days
        current_bar = ohlc_bars_target_period[day_idx]
        daily_tokens = tokenize_ohlc_bar_sequentially(prev_bar_for_gt_tokenization, current_bar)
        quantized_ground_truth_tokens_flat.extend(daily_tokens)
        prev_bar_for_gt_tokenization = current_bar

    quantized_ground_truth_ohlc_bars = []
    num_gt_full_days = len(quantized_ground_truth_tokens_flat) // tokens_per_day
    current_open_for_gt_recon = float(bar_before_target_period["c"]) if bar_before_target_period else last_close_of_prompt

    for day_idx in range(num_gt_full_days):
        day_tokens = quantized_ground_truth_tokens_flat[day_idx*tokens_per_day : (day_idx+1)*tokens_per_day]
        if len(day_tokens) < tokens_per_day: continue
        val_gap = rep_vals["gap"].get(day_tokens[0], 0.0)
        val_uw = rep_vals["upper_wick"].get(day_tokens[1], 0.0)
        val_lw = rep_vals["lower_wick"].get(day_tokens[2], 0.0)
        val_body = rep_vals["body"].get(day_tokens[3], 0.0)
        gt_o = current_open_for_gt_recon * (1 + val_gap)
        gt_c = gt_o * (1 + val_body)
        gt_h = val_uw * gt_o + max(gt_o, gt_c); gt_l = min(gt_o, gt_c) - val_lw * gt_o
        h_temp = max(gt_o, gt_c, gt_h); l_temp = min(gt_o, gt_c, gt_l)
        quantized_ground_truth_ohlc_bars.append({'open': gt_o, 'high': h_temp, 'low': l_temp, 'close': gt_c})
        current_open_for_gt_recon = gt_c

    # --- Plotting ---
    num_plotted_days = min(len(predicted_ohlc_bars), len(ohlc_bars_target_period), len(quantized_ground_truth_ohlc_bars), predict_days)
    if num_plotted_days == 0: print("No full predicted days to plot."); return
    fig, ax = plt.subplots(figsize=(15, 8))
    # ... (Plotting logic from your previous version is fine, ensure legend and titles are updated) ...
    for i in range(num_plotted_days): # Actual Raw
        bar = ohlc_bars_target_period[i]
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
        color = 'darkgreen' if c >= o else 'darkred'
        ax.plot([i - 0.05, i - 0.05], [l, h], color=color, linewidth=1, label='_nolegend_')
        ax.add_patch(plt.Rectangle((i - 0.3 - 0.05, min(o, c)), 0.6, abs(c - o), facecolor=color, edgecolor='black', alpha=0.7, label='_nolegend_'))
    for i in range(num_plotted_days): # Quantized GT
        q_gt_bar = quantized_ground_truth_ohlc_bars[i]
        qo, qh, ql, qc = q_gt_bar['open'], q_gt_bar['high'], q_gt_bar['low'], q_gt_bar['close']
        q_color = 'royalblue' if qc >= qo else 'orangered'
        ax.plot([i, i], [ql, qh], color=q_color, linewidth=1, linestyle='--', label='_nolegend_')
        ax.add_patch(plt.Rectangle((i - 0.3, min(qo, qc)), 0.6, abs(qc - qo), facecolor=q_color, edgecolor='dimgrey', alpha=0.6, label='_nolegend_'))
    for i in range(num_plotted_days): # Model Predicted
        p_bar = predicted_ohlc_bars[i]
        po, ph, pl, pc = p_bar['open'], p_bar['high'], p_bar['low'], p_bar['close']
        p_color = 'lime' if pc >= po else 'fuchsia'
        ax.plot([i + 0.05, i + 0.05], [pl, ph], color=p_color, linewidth=1, linestyle=':', label='_nolegend_')
        ax.add_patch(plt.Rectangle((i - 0.3 + 0.05, min(po, pc)), 0.6, abs(pc - po), facecolor=p_color, edgecolor='darkgrey', alpha=0.5, label='_nolegend_'))
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkgreen', lw=2, label='Actual Raw OHLC'),
        Line2D([0], [0], color='royalblue', lw=2, linestyle='--', label='Actual Quantized OHLC'),
        Line2D([0], [0], color='lime', lw=2, linestyle=':', label='Model Predicted OHLC')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title(f"Actual vs. Predicted vs. Quantized OHLC for {config['primary_symbol']}")
    ax.set_xlabel(f"Trading Days into Prediction Period (Prompt: {enc_days} days)")
    ax.set_ylabel("Price"); ax.grid(True); plt.tight_layout()
    plot_save_path = os.path.join(config.get("PROJECT_ROOT", "."), "data", f"encdec_test_plot_{config['primary_symbol']}.png")
    plt.savefig(plot_save_path); print(f"Plot saved to {plot_save_path}")
    plt.show()