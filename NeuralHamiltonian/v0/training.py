# training.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # For softmax in testing
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os # For saving plot

from config import config

from data_handling import savedata, gen_data_for_model, gen_sequential_ohlc_token_sequence_with_bars, save_model
from datetime import datetime, timedelta # For testing data generation date logic

def get_training_data_loader(): # Renamed
    all_sequences_encoder_input, all_sequences_decoder_input, all_sequences_decoder_target = [], [], []
    print(f"Generating {config['num_training_samples']} tokenized stock samples...")
    generated_samples = 0
    attempts = 0
    # Increase max_attempts if data generation is often sparse/failing
    max_attempts = config["num_training_samples"] * 5

    while generated_samples < config["num_training_samples"] and attempts < max_attempts:
        attempts += 1
        enc_ids, dec_in_ids, dec_tgt_ids = gen_data_for_model()
        if enc_ids is not None:
            all_sequences_encoder_input.append(enc_ids)
            all_sequences_decoder_input.append(dec_in_ids)
            all_sequences_decoder_target.append(dec_tgt_ids)
            generated_samples += 1
        if attempts > 0 and attempts % (max_attempts // 10 if max_attempts > 10 else 1) == 0 : # Print progress
            print(f"Attempted {attempts}, Generated {generated_samples} valid samples...")
    
    if generated_samples < config["num_training_samples"]:
        print(f"Warning: Could only generate {generated_samples} out of {config['num_training_samples']} desired samples.")
        if generated_samples < config["batch_size"]: # Not even enough for one batch
            raise RuntimeError("Failed to generate enough training samples. Check data fetching, tokenization, or increase num_training_samples/attempts.")

    X_enc_np = np.array(all_sequences_encoder_input)
    X_dec_in_np = np.array(all_sequences_decoder_input)
    Y_tgt_np = np.array(all_sequences_decoder_target)

    indices = np.arange(len(X_enc_np))
    np.random.shuffle(indices)
    X_enc_np, X_dec_in_np, Y_tgt_np = X_enc_np[indices], X_dec_in_np[indices], Y_tgt_np[indices]

    valsplit_index = int(len(X_enc_np) * (1 - config["VAL_SPLIT_RATIO"]))
    if valsplit_index == 0 and len(X_enc_np) > 0: # Ensure val set exists if possible
        valsplit_index = 1 if len(X_enc_np) > 1 else 0
    if valsplit_index == len(X_enc_np) and len(X_enc_np) > 0: # Ensure train set exists
        valsplit_index = len(X_enc_np) -1 if len(X_enc_np) > 1 else len(X_enc_np)


    Xtrain_enc, Xval_enc = X_enc_np[:valsplit_index], X_enc_np[valsplit_index:]
    Xtrain_dec_in, Xval_dec_in = X_dec_in_np[:valsplit_index], X_dec_in_np[valsplit_index:]
    Ytrain_tgt, Yval_tgt = Y_tgt_np[:valsplit_index], Y_tgt_np[valsplit_index:]

    if len(Xtrain_enc) > 0 : # Only save if there's training data
         savedata(Xtrain_enc, Xtrain_dec_in, Ytrain_tgt, Xval_enc, Xval_dec_in, Yval_tgt, data_file_path_arg=config["data_save_path"])
    else:
        print("Warning: No training data to save after splitting.")
        raise RuntimeError("No training data generated after split. Increase num_training_samples or check data generation.")
    
    if len(Xval_enc) == 0 and len(Xtrain_enc) > 0 :
        print("Warning: No validation data after split. Using a small part of training data for validation.")
        # Fallback: use last few samples of training for validation if val set is empty
        num_val_fallback = max(1, int(0.05 * len(Xtrain_enc))) # 5% or at least 1
        if len(Xtrain_enc) > num_val_fallback:
            Xval_enc, Xval_dec_in, Yval_tgt = Xtrain_enc[-num_val_fallback:], Xtrain_dec_in[-num_val_fallback:], Ytrain_tgt[-num_val_fallback:]
            Xtrain_enc, Xtrain_dec_in, Ytrain_tgt = Xtrain_enc[:-num_val_fallback], Xtrain_dec_in[:-num_val_fallback], Ytrain_tgt[:-num_val_fallback]


    return (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt)

def training(model, optimizer, start_epoch):
    device = torch.device(config.get("device", "cpu"))
    model.to(device)
    print(f"Training on device: {device}")

    print("Preparing training data...")
    if config.get("load_training_data", False):
        try:
            data_loaded = np.load(config["data_save_path"])
            Xtrain_enc = data_loaded['Xtrain_enc']; Xtrain_dec_in = data_loaded['Xtrain_dec_in']; Ytrain_tgt = data_loaded['Ytrain_tgt']
            Xval_enc = data_loaded['Xval_enc']; Xval_dec_in = data_loaded['Xval_dec_in']; Yval_tgt = data_loaded['Yval_tgt']
            print("Loaded training data from file.")
            if Xtrain_enc.size == 0: # Check if loaded arrays are empty
                raise ValueError("Loaded training data is empty.")
        except (FileNotFoundError, ValueError) as e:
            print(f"Data file issue ({e}) at {config['data_save_path']}. Generating new data.")
            (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt) = get_training_data_loader()
    else:
        (Xtrain_enc, Xtrain_dec_in, Ytrain_tgt), (Xval_enc, Xval_dec_in, Yval_tgt) = get_training_data_loader()

    # Convert to PyTorch tensors
    Xtrain_enc_t = torch.tensor(Xtrain_enc, dtype=torch.long)
    Xtrain_dec_in_t = torch.tensor(Xtrain_dec_in, dtype=torch.long)
    Ytrain_tgt_t = torch.tensor(Ytrain_tgt, dtype=torch.long)
    Xval_enc_t = torch.tensor(Xval_enc, dtype=torch.long)
    Xval_dec_in_t = torch.tensor(Xval_dec_in, dtype=torch.long)
    Yval_tgt_t = torch.tensor(Yval_tgt, dtype=torch.long)

    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_idx"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get("scheduler_patience", 3)) # Adjusted patience
    
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    BATCH_SIZE = config["batch_size"]
    CLIP_GRAD_NORM = config.get("clip_grad_norm")
    NUM_EPOCHS = config["num_epoch"]

    train_dataset = TensorDataset(Xtrain_enc_t, Xtrain_dec_in_t, Ytrain_tgt_t) # Xtrain_enc_t not directly used by current model
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if len(Xval_enc_t) > 0:
        val_dataset = TensorDataset(Xval_enc_t, Xval_dec_in_t, Yval_tgt_t)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None
        print("Warning: No validation data available. Skipping validation loop.")


    best_val_loss = float('inf')

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        # The first element of train_loader tuple is Xtrain_enc_t, which is currently unused as model input
        for batch_idx, (_, batch_dec_in, batch_tgt) in enumerate(train_loader):
            batch_dec_in = batch_dec_in.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            
            dec_padding_mask = (batch_dec_in != config["pad_idx"]).to(device)
            predictions_logits = model(input_ids=batch_dec_in, padding_mask=dec_padding_mask)
            
            loss = criterion(
                predictions_logits.reshape(-1, config["vocab_size"]),
                batch_tgt.reshape(-1)
            )
            
            loss.backward()
            if CLIP_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            optimizer.step()
            epoch_train_loss += loss.item() # Accumulate loss directly

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)

        avg_epoch_val_loss = float('nan') # Default if no validation
        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for _, batch_val_dec_in, batch_val_tgt in val_loader:
                    batch_val_dec_in = batch_val_dec_in.to(device)
                    batch_val_tgt = batch_val_tgt.to(device)

                    val_dec_padding_mask = (batch_val_dec_in != config["pad_idx"]).to(device)
                    val_preds_logits = model(input_ids=batch_val_dec_in, padding_mask=val_dec_padding_mask)
                    
                    v_loss = criterion(
                        val_preds_logits.reshape(-1, config["vocab_size"]),
                        batch_val_tgt.reshape(-1)
                    )
                    epoch_val_loss += v_loss.item()
            
            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            scheduler.step(avg_epoch_val_loss)

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                save_model(model, optimizer, epoch + 1, config["model_save_path"])

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}")

    print("Training finished.")



def testing(model_instance):
    device = torch.device(config.get("device", "cpu"))
    model_instance.to(device); model_instance.eval()
    print("Generating a test sample for plotting with sequential OHLC feature tokens...")

    tokens_per_day = config["tokens_per_day"]
    # max_seq_len is in terms of total tokens. Calculate days.
    max_days_in_model_seq = config["max_seq_len"] // tokens_per_day

    L_prompt_days = max_days_in_model_seq // 2
    L_predict_days = max_days_in_model_seq - L_prompt_days
    
    total_days_to_fetch = L_prompt_days + L_predict_days
    now = datetime.now()
    end_dt_for_series = now - timedelta(days=np.random.randint(total_days_to_fetch + 90, total_days_to_fetch + 730))

    # Get prompt tokens, actual OHLC Bar objects for the whole period, and the bar before the series
    prompt_and_target_tokens_flat, ohlc_bars_for_period, bar_before_series = \
        gen_sequential_ohlc_token_sequence_with_bars(
            config["primary_symbol"], total_days_to_fetch, end_dt_for_series
        )

    if prompt_and_target_tokens_flat is None:
        print("Failed to generate data for testing plot."); return

    # Prompt tokens are the first L_prompt_days * tokens_per_day
    num_prompt_tokens = L_prompt_days * tokens_per_day
    prompt_token_ids = prompt_and_target_tokens_flat[:num_prompt_tokens]
    
    # Actual OHLC bars for the period we will predict tokens for
    ohlc_bars_target_period = ohlc_bars_for_period[L_prompt_days:]

    # Last close price of the prompt period, or from bar_before_series if prompt is empty
    last_close_of_prompt = float(ohlc_bars_for_period[L_prompt_days-1]["c"]) if L_prompt_days > 0 else float(bar_before_series["c"])
    
    print(f"Last close of prompt period: {last_close_of_prompt:.2f}")

    # --- Autoregressive Generation (token by token) ---
    generated_tokens_predicted_period_flat = []
    current_input_token_seq = torch.tensor([prompt_token_ids], dtype=torch.long).to(device) # (1, num_prompt_tokens)

    num_tokens_to_predict = L_predict_days * tokens_per_day

    with torch.no_grad():
        for _ in range(num_tokens_to_predict):
            input_for_model = current_input_token_seq
            if input_for_model.size(1) > config["max_seq_len"]: # Prune if too long
                input_for_model = input_for_model[:, -config["max_seq_len"]:]
            
            padding_mask = (input_for_model != config["pad_idx"]).to(device)
            output_logits = model_instance(input_ids=input_for_model, padding_mask=padding_mask)
            next_token_logits = output_logits[:, -1, :] # Logits for the very next token
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens_predicted_period_flat.append(next_token_id)
            
            if next_token_id == config["eos_token_id"] and _ % tokens_per_day == (tokens_per_day -1) : # EOS at end of a day's tokens
                print("Generated <EOS> token at end of a day's features."); break 
            
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            current_input_token_seq = torch.cat([current_input_token_seq, next_token_tensor], dim=1)
            # No need to prune inside loop if input_for_model handles it, but good for very long generations
            # if current_input_token_seq.size(1) > config["max_seq_len"]:
            #      current_input_token_seq = current_input_token_seq[:, 1:]


    # --- Detokenization and OHLC Path Reconstruction ---
    rep_vals = config["REPRESENTATIVE_VALUES_FOR_BINS"]
    predicted_ohlc_bars = [] # List of dicts: {'open': o, 'high': h, 'low': l, 'close': c}
    
    # Group generated tokens by day
    num_predicted_full_days = len(generated_tokens_predicted_period_flat) // tokens_per_day
    
    current_open = last_close_of_prompt # Overnight gap is relative to this for the first predicted day

    for day_idx in range(num_predicted_full_days):
        day_tokens = generated_tokens_predicted_period_flat[day_idx*tokens_per_day : (day_idx+1)*tokens_per_day]
        if len(day_tokens) < tokens_per_day: continue # Not a full day's worth of tokens

        # Detokenize each feature for the day
        val_gap = rep_vals["gap"].get(day_tokens[0], 0.0)
        val_uw = rep_vals["upper_wick"].get(day_tokens[1], 0.0)
        val_lw = rep_vals["lower_wick"].get(day_tokens[2], 0.0)
        val_body = rep_vals["body"].get(day_tokens[3], 0.0)

        # Reconstruct OHLC
        # 1. Predicted Open for current day using gap from previous day's *actual* or *predicted* close
        pred_o = current_open * (1 + val_gap) # `current_open` here is prev_close for gap calc

        # Denormalize based on this predicted Open
        pred_c = pred_o * (1 + val_body)
        
        # Upper wick value is (H - max(O,C))/O_pred => H = UW_val * O_pred + max(O_pred, C_pred)
        pred_h = val_uw * pred_o + max(pred_o, pred_c)
        # Lower wick value is (min(O,C) - L)/O_pred => L = min(O_pred, C_pred) - LW_val * O_pred
        pred_l = min(pred_o, pred_c) - val_lw * pred_o
        
        # Ensure OHLC integrity
        # H must be max, L must be min
        h_temp = max(pred_o, pred_c, pred_h)
        l_temp = min(pred_o, pred_c, pred_l)
        pred_h = h_temp
        pred_l = l_temp
        if pred_l > pred_h : pred_l = pred_h # Should not happen if wicks are positive

        predicted_ohlc_bars.append({'open': pred_o, 'high': pred_h, 'low': pred_l, 'close': pred_c})
        
        current_open = pred_c # The close of this predicted day is the basis for next day's gap

    # --- Plotting Candlesticks ---
    num_plotted_days = len(predicted_ohlc_bars)
    if num_plotted_days == 0: print("No full predicted days to plot."); return

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot Actual OHLC Candlesticks
    for i in range(min(num_plotted_days, len(ohlc_bars_target_period))):
        bar = ohlc_bars_target_period[i]
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
        color = 'green' if c >= o else 'red'
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o), facecolor=color, edgecolor='black'))

    # Plot Predicted OHLC Candlesticks
    for i in range(num_plotted_days):
        p_bar = predicted_ohlc_bars[i]
        po, ph, pl, pc = p_bar['open'], p_bar['high'], p_bar['low'], p_bar['close']
        p_color = 'lime' if pc >= po else 'fuchsia' # Different colors for predicted
        # Offset predicted slightly for visibility
        ax.plot([i + 0.05, i + 0.05], [pl, ph], color=p_color, linewidth=1, linestyle=':')
        ax.add_patch(plt.Rectangle((i - 0.3 + 0.05, min(po, pc)), 0.6, abs(pc - po), facecolor=p_color, edgecolor='grey', alpha=0.6))
    
    # Create proxy artists for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Actual OHLC'),
        Line2D([0], [0], color='grey', lw=2, linestyle=':', alpha=0.6, label='Predicted OHLC (from Tokens)')]
    ax.legend(handles=legend_elements)

    ax.set_title(f"Actual vs. Predicted OHLC for {config['primary_symbol']} (Sequential Feature Tokens)")
    ax.set_xlabel(f"Trading Days into Prediction Period (Prompt: {L_prompt_days} days)")
    ax.set_ylabel("Price")
    ax.grid(True); plt.tight_layout()
    plot_save_path = os.path.join(config.get("PROJECT_ROOT", "."), "data", "test_sequential_ohlc_plot.png")
    plt.savefig(plot_save_path); print(f"Plot saved to {plot_save_path}")
    plt.show()