# data_handling.py
import os
import numpy as np
import torch
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import timedelta, datetime

from config import config

api_key = config["api_keys"]["alpaca_api_key_id"]
secret_key = config["api_keys"]["alpaca_api_secret_key"]
if api_key and secret_key:
    client = StockHistoricalDataClient(api_key, secret_key, raw_data=True)
else:
    client = None; print("Warning: Alpaca API keys not configured.")
now_dt = datetime.now()

def savedata(X_enc_ids, X_dec_in_ids, Y_tgt_ids, Xval_enc_ids, Xval_dec_in_ids, Yval_tgt_ids, data_file_path_arg):
    try:
        data_dir = os.path.dirname(data_file_path_arg); os.makedirs(data_dir, exist_ok=True)
        print(f"Saving tokenized training data to: {data_file_path_arg}")
        np.savez_compressed(data_file_path_arg,
                 Xtrain_enc=X_enc_ids, Xtrain_dec_in=X_dec_in_ids, Ytrain_tgt=Y_tgt_ids,
                 Xval_enc=Xval_enc_ids, Xval_dec_in=Xval_dec_in_ids, Yval_tgt=Yval_tgt_ids)
    except Exception as e: print(f"Error saving data to {data_file_path_arg}: {e}")

def save_model(model, optimizer, epoch, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'config': config
    }
    torch.save(checkpoint, model_path); print(f"Model checkpoint saved to {model_path} at epoch {epoch}")

def _get_token_for_feature(value: float, feature_type: str) -> int:
    """Helper to get token ID for a given feature value and type."""
    bins_list = config["ALL_FEATURE_BINS"][feature_type]
    no_data_token_name_suffix = f"NO_{feature_type.upper()}_DATA"
    
    for lower, upper, token_name_suffix in bins_list:
        full_token_name = f"{feature_type.upper()}_{token_name_suffix}"
        if token_name_suffix == no_data_token_name_suffix: # Should be handled by caller if value is None
            continue
        if value is None: # If value is None, use the NO_DATA token for this feature
            return config["STOCK_TOKEN_TO_ID"][f"{feature_type.upper()}_{no_data_token_name_suffix}"]

        if lower is None and value < upper: return config["STOCK_TOKEN_TO_ID"][full_token_name]
        if upper is None and value >= lower: return config["STOCK_TOKEN_TO_ID"][full_token_name]
        if lower is not None and upper is not None and lower <= value < upper:
            return config["STOCK_TOKEN_TO_ID"][full_token_name]
    
    print(f"Warning: No bin for {feature_type} matched value {value:.4f}. Using NO_DATA token.")
    return config["STOCK_TOKEN_TO_ID"][f"{feature_type.upper()}_{no_data_token_name_suffix}"]


def tokenize_ohlc_bar_sequentially(prev_bar_data, current_bar_data) -> list[int]:
    """Tokenizes a single day's OHLC into a sequence of 4 feature tokens."""
    tokens_for_day = []

    o, h, l, c = float(current_bar_data["o"]), float(current_bar_data["h"]), \
                 float(current_bar_data["l"]), float(current_bar_data["c"])

    # 1. Gap Token
    if prev_bar_data and hasattr(prev_bar_data, 'close') and prev_bar_data["c"] is not None and float(prev_bar_data["c"]) != 0:
        prev_c = float(prev_bar_data["c"])
        gap_value = (o - prev_c) / prev_c
        tokens_for_day.append(_get_token_for_feature(gap_value, "gap"))
    else:
        tokens_for_day.append(_get_token_for_feature(None, "gap")) # NO_GAP_DATA

    # Denominator for wick/body normalization (use Open, or Close if Open is zero)
    norm_denom = o if o != 0 else c
    if norm_denom == 0: # If both O and C are 0, cannot normalize meaningfully
        tokens_for_day.extend([_get_token_for_feature(None, feature) for feature in ["upper_wick", "lower_wick", "body"]])
        return tokens_for_day

    # 2. Upper Wick Token
    uw_value = (h - max(o, c)) / norm_denom
    tokens_for_day.append(_get_token_for_feature(uw_value, "upper_wick"))

    # 3. Lower Wick Token
    lw_value = (min(o, c) - l) / norm_denom
    tokens_for_day.append(_get_token_for_feature(lw_value, "lower_wick"))

    # 4. Body Token
    body_value = (c - o) / norm_denom
    tokens_for_day.append(_get_token_for_feature(body_value, "body"))
    
    assert len(tokens_for_day) == config["tokens_per_day"]
    return tokens_for_day


def gen_sequential_ohlc_token_sequence_with_bars(primary_symbol: str, num_days: int, end_date: datetime):
    """
    Generates a sequence of feature tokens (4 per day) and the corresponding Bar objects.
    Returns (list of all feature_tokens, list of Bar objects for these days, first_prev_bar_for_context).
    The Bar objects list will have 'num_days' elements.
    The token list will have 'num_days * tokens_per_day' elements.
    """
    if client is None:
        print("Using dummy data for sequential OHLC tokens."); tokens_per_day = config["tokens_per_day"]
        dummy_tokens_flat = np.random.randint(config["sos_token_id"]+1, config["vocab_size"], size=num_days * tokens_per_day).tolist()
        dummy_bars_list = []
        price = 100.0; dummy_prev_bar = None
        for i in range(num_days + 1): # +1 to get a prev_bar for the first day
            o = price + np.random.randn()*0.1; c = o + np.random.randn()*0.5
            h = max(o,c) + np.random.rand()*0.2; l = min(o,c) - np.random.rand()*0.2
            bar = type('DummyBar', (), {'open':o,'high':h,'low':l,'close':c, 'timestamp':end_date - timedelta(days=num_days-i)})()
            if i==0: dummy_prev_bar = bar
            else: dummy_bars_list.append(bar)
            price = c
        return dummy_tokens_flat, dummy_bars_list, dummy_prev_bar

    # Need num_days bars + 1 previous bar for the first day's gap calculation.
    bars_to_fetch = num_days + 1
    days_to_fetch_buffer = int(bars_to_fetch * 2.5) + 30
    start_date = end_date - timedelta(days=days_to_fetch_buffer)

    try:
        bars_data_req = StockBarsRequest(
            symbol_or_symbols=[primary_symbol],
            timeframe=TimeFrame(config["timeframe_value"], TimeFrameUnit.Day),
            start=start_date, end=end_date, adjustment='split'
        )
        bars_response = client.get_stock_bars(bars_data_req)
    except Exception as e: print(f"Error fetching stock data for {primary_symbol} ending {end_date}: {e}"); return None,None,None

    if not bars_response or primary_symbol not in bars_response:
        print(f"No data for {primary_symbol} ending {end_date}."); return None,None,None

    symbol_bars = bars_response[primary_symbol]
    if len(symbol_bars) < bars_to_fetch:
        print(f"Need {bars_to_fetch} bars, got {len(symbol_bars)} for {primary_symbol} ending {end_date}."); return None,None,None
    
    relevant_bars_with_prev = symbol_bars[-bars_to_fetch:] # Last N+1 bars

    all_feature_tokens = []
    ohlc_bars_for_tokens = [] # These are the 'current_bar_data'

    first_prev_bar_for_context = relevant_bars_with_prev[0] # The bar *before* the first day we tokenize

    for i in range(len(relevant_bars_with_prev) - 1):
        prev_bar = relevant_bars_with_prev[i]
        current_bar = relevant_bars_with_prev[i+1]
        daily_feature_tokens = tokenize_ohlc_bar_sequentially(prev_bar, current_bar)
        all_feature_tokens.extend(daily_feature_tokens)
        ohlc_bars_for_tokens.append(current_bar)
    
    assert len(ohlc_bars_for_tokens) == num_days
    assert len(all_feature_tokens) == num_days * config["tokens_per_day"]
    return all_feature_tokens, ohlc_bars_for_tokens, first_prev_bar_for_context


def gen_data_for_model(): # For training
    L_days = config["max_seq_len"] // config["tokens_per_day"] # Number of days for enc/target
    total_days_needed = 2 * L_days

    min_offset_days = total_days_needed + 90
    max_offset_days = min_offset_days + 1500
    random_offset_days = np.random.randint(min_offset_days, max_offset_days)
    end_dt_for_sequence = now_dt - timedelta(days=random_offset_days)

    # full_token_sequence is now flat list of all feature tokens
    full_token_sequence, _, _ = gen_sequential_ohlc_token_sequence_with_bars(
        config["primary_symbol"], total_days_needed, end_dt_for_sequence
    )

    if full_token_sequence is None or len(full_token_sequence) < total_days_needed * config["tokens_per_day"]:
        return None, None, None

    # L_tokens is the number of tokens for the encoder part (and for target part)
    L_tokens = L_days * config["tokens_per_day"] # Should equal config["max_seq_len"]

    encoder_context_ids = np.array(full_token_sequence[:L_tokens], dtype=np.int64)
    target_ids_to_predict_flat = full_token_sequence[L_tokens:]

    decoder_input_ids = np.concatenate((
        [config["sos_token_id"]], target_ids_to_predict_flat[:-1]
    )).astype(np.int64)
    target_output_ids = np.array(target_ids_to_predict_flat, dtype=np.int64)

    assert len(encoder_context_ids) == L_tokens
    assert len(decoder_input_ids) == L_tokens
    assert len(target_output_ids) == L_tokens
    
    return encoder_context_ids, decoder_input_ids, target_output_ids