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
    print("Warning: Alpaca API keys not configured.")
    exit(0)
now_dt = datetime.now()


def savedata(X_train, Y_train, X_val, Y_val, data_file_path_arg):
    try:
        data_dir = os.path.dirname(data_file_path_arg)
        os.makedirs(data_dir, exist_ok=True)
        print(f"Saving training data to: {data_file_path_arg}")
        np.savez_compressed(data_file_path_arg,
                            X_train=X_train, Y_train=Y_train,
                            X_val=X_val, Y_val=Y_val)
    except Exception as e:
        print(f"Error saving data to {data_file_path_arg}: {e}")

def save_model(model, optimizer, epoch, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'config': config
    }
    torch.save(checkpoint, model_path); print(f"Model checkpoint saved to {model_path} at epoch {epoch}")

def inter_time(timepoint1,timepoint2):
    start_time = datetime.fromisoformat(timepoint1.replace('Z', '+00:00'))
    end_time = datetime.fromisoformat(timepoint2.replace('Z', '+00:00'))

    duration = end_time - start_time

    mid_time = start_time + duration / 2

    mid_time_str = mid_time.isoformat().replace('+00:00', 'Z')
    return mid_time_str

def get_stock_data(symbols, start_date, end_date):
    """
    Fetches stock bar data for a given list of symbols and a date range.
    This version does NOT take an 'N' parameter. It fetches all data in the range.
    """
    try:
        # Ensure symbols is a list, as the API expects
        if not isinstance(symbols, list):
            symbols = [symbols] 

        bars_data_req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start_date,
            end=end_date
        )
        # Use the raw_data=True client you already have configured
        bars_response = client.get_stock_bars(bars_data_req)
        
        # Convert the raw response to a more usable format
        stockdata = {}
        for s in symbols:
            if s in bars_response:
                # Store as a list of tuples: (time, open, close, high, low)
                # IMPORTANT: Unpacking o, c, h, l from the datapoint
                stockdata[s] = [(dp["t"], dp["o"], dp["c"], dp["h"], dp["l"]) for dp in bars_response[s]]
            else:
                print(f"Warning: No data returned for symbol {s} in the given timeframe.")
                stockdata[s] = [] # Return empty list if no data for a symbol
        return stockdata

    except Exception as e:
        print(f"Error fetching stock data for {symbols}: {e}")
        # Return None to indicate failure, which the calling function should handle
        return None
    
def prepare_dataset_from_api(symbols, primary_symbol, years_of_data=10):
    """
    Performs a single, large API call to fetch historical data and processes it
    into a complete (X, Y) dataset for training.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (all_X, all_Y) numpy arrays.
    """
    print(f"Preparing dataset from API: Fetching {years_of_data} years of data...")
    
    seq_len = config["sequence_length"]
    
    # 1. Fetch a large chunk of historical data
    end_date = datetime.now()- timedelta(days=1)
    start_date = end_date - timedelta(days=int(years_of_data * 365.25))
    
    all_symbols_data = get_stock_data(symbols, start_date, end_date)
    if all_symbols_data is None or not all_symbols_data[primary_symbol]:
        raise RuntimeError("Failed to fetch sufficient data from the API.")

    # Convert to NumPy arrays for easier processing and alignment
    # We also need to handle cases where some stocks have shorter histories.
    # We find the common set of dates across all symbols.
    dates_per_symbol = {s: {bar[0] for bar in data} for s, data in all_symbols_data.items()}
    common_dates = sorted(list(set.intersection(*dates_per_symbol.values())))
    
    print(f"Found {len(common_dates)} common trading days across all symbols.")

    # Create a unified data dictionary with aligned dates
    aligned_data = {s: [] for s in symbols}
    data_by_date = {s: {bar[0]: bar[1:] for bar in data} for s, data in all_symbols_data.items()}
    for date in common_dates:
        for s in symbols:
            aligned_data[s].append(data_by_date[s][date])

    # Convert to numpy arrays
    for s in symbols:
        aligned_data[s] = np.array(aligned_data[s], dtype=np.float32)

    # 2. Create the complete X and Y arrays using sliding windows
    all_X, all_Y = [], []
    
    primary_ohlc = aligned_data[primary_symbol]
    
    # Concatenate all features for the input X
    # Shape: (num_days, num_symbols * 4)
    all_features_X = np.concatenate([aligned_data[s] for s in symbols], axis=1)

    # The total number of samples we can create is len - (2 * seq_len) + 1
    num_possible_samples = len(common_dates) - (2 * seq_len) + 1
    
    print(f"Creating {num_possible_samples} sliding window samples...")
    for i in range(num_possible_samples):
        # Input window for X
        x_window = all_features_X[i : i + seq_len]
        
        # Target window for Y. It starts right after the input window ends.
        y_start_idx = i + seq_len
        y_window = primary_ohlc[y_start_idx : y_start_idx + seq_len]
        
        all_X.append(x_window)
        all_Y.append(y_window)
        
    if not all_X:
        raise ValueError("Could not create any training samples. The date range might be too short or lack common trading days.")

    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)
