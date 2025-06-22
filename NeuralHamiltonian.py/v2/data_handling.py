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

# def get_stock_data(symbols,start_date,end_date,N):
#     try:
#         bars_data_req = StockBarsRequest(
#             symbol_or_symbols=symbols,
#             timeframe=TimeFrame(1, TimeFrameUnit.Day),
#             start=start_date, end=end_date
#         )
#         bars_response = client.get_stock_bars(bars_data_req)
#     except Exception as e: print(f"Error fetching stock data for {symbols}: {e}"); return None,None,None
#     stockdata={}
#     for s in symbols:
#         stockdata_s=[(datapoint["t"],datapoint["o"],datapoint["c"],datapoint["h"],datapoint["l"]) for datapoint in bars_response[s]]
#         if len(stockdata_s)>N:
#             for i in range(len(stockdata_s)-N):
#                 rand_ind=np.random.randint(0, len(stockdata_s))
#                 stockdata_s.pop()
#         elif len(stockdata_s)<N:
#             for i in range(N-len(stockdata_s)):
#                 rand_ind=np.random.randint(1, len(stockdata_s)-1)
#                 prev_item = stockdata_s[rand_ind - 1]
#                 next_item = stockdata_s[rand_ind]

#                 new_time = inter_time(prev_item[0], next_item[0])

#                 new_o = 0.5 * (prev_item[1] + next_item[1])
#                 new_c = 0.5 * (prev_item[2] + next_item[2])
#                 new_h = 0.5 * (prev_item[3] + next_item[3])
#                 new_l = 0.5 * (prev_item[4] + next_item[4])
                
#                 new_val = (new_time, new_o, new_c, new_h, new_l)
#                 stockdata_s.insert(rand_ind,new_val)
#         stockdata[s]=np.array(stockdata_s,dtype = [
#         ('t', 'U24'), 
#         ('o', 'f8'), 
#         ('c', 'f8'), 
#         ('h', 'f8'), 
#         ('l', 'f8')
#     ])
#     return stockdata

#def gen_data_for_model():
#
#    predL=np.random.randint(int(0.25*config["sequence_length"]), int(2*config["sequence_length"]))
#
#    end_prediction = now_dt - timedelta(days=np.random.randint(2, 10**3))
#    end_real = end_prediction - timedelta(days=predL)
#    start_real = end_real - timedelta(days=config["sequence_length"])
#
#    all_persymbol_data=get_stock_data(config["symbols"],start_real,end_real,config["sequence_length"])
#    target_data=get_stock_data(config["primary_symbol"],end_real,end_prediction,config["sequence_length"])[config["primary_symbol"]]
#    
#    Alldata=[]
#    for i in range(config["sequence_length"]):
#        Alldata.append([])
#        for s in config["symbols"]:
#            Alldata[-1].extend(all_persymbol_data[s][i][1:])
#    Alldata=np.array([Alldata])
#
#
#    inp_data=Alldata[:config["sequence_length"]]
#    tgt_data=[tg[1:] for tg in target_data]
#    return inp_data, tgt_data

def prepare_dataset_from_api(symbols, primary_symbol, years_of_data=7):
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

def gen_data_for_model():
    """
    Generates one sample of (X, Y) data.
    X: Input features from all symbols for `sequence_length` days.
       Shape: (sequence_length, num_symbols * 4)
    Y: Target features (OHLC) for the primary symbol for the *next* `sequence_length` days.
       Shape: (sequence_length, 4)
    """
    seq_len = config["sequence_length"]
    
    # We need a total of 2 * seq_len days of data to create one (X, Y) pair
    try:
        end_date = now_dt - timedelta(days=np.random.randint(2, 5*365)) # Go back up to 5 years
        start_date = end_date - timedelta(days=seq_len * 2 + 30) # Fetch a bit more to ensure we have enough data

        # Fetch data for all symbols
        all_symbols_data = get_stock_data(config["symbols"], start_date, end_date, seq_len * 2)
        #print(all_symbols_data)
        if all_symbols_data is None: return None, None

        # Fetch data for the primary symbol (target)
        target_symbol_data = all_symbols_data[config["primary_symbol"]]

        # Combine all symbol data into a single feature matrix
        # This is our input data `X`
        input_data_list = []
        for i in range(seq_len):
            timestep_features = []
            for s in config["symbols"]:
                # Append o, c, h, l for each symbol
                timestep_features.extend([all_symbols_data[s][i][j] for j in range(1,len(all_symbols_data[s][i]))])

            input_data_list.append(timestep_features)

        X = np.array(input_data_list, dtype=np.float32)
        # Create the target data `Y` from the primary symbol's *future*
        # It starts right after the input data ends
        target_data_list = [[row[i] for i in range(1,len(row))] for row in target_symbol_data[seq_len : seq_len * 2]]
        Y = np.array(target_data_list, dtype=np.float32)
        # Final check for correct shapes
        if X.shape == (seq_len, config["input_dim"]) and Y.shape == (seq_len, config["output_dim"]):
            return X, Y
        else:
            print(f"Warning: Mismatched data shape. X: {X.shape}, Y: {Y.shape}. Skipping.")
            return None, None

    except Exception as e:
        print(f"Error in gen_data_for_model: {e}") # Can be noisy, enable for debugging
        return None, None