# stockbot/data.py - Prepares real stock data.
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timedelta

try:
    from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("Alpaca API keys not found in environment variables.")
    client = StockHistoricalDataClient(api_key, secret_key, raw_data=True)
    ALPACA_AVAILABLE = True
    print("Alpaca client initialized successfully.")
except (ImportError, ValueError) as e:
    print(f"Warning: Alpaca setup failed. Real data fetching is disabled. Error: {e}")
    ALPACA_AVAILABLE = False

def _get_stock_data(symbols, start_date, end_date):
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(1, TimeFrameUnit.Day), start=start_date, end=end_date)
    bars = client.get_stock_bars(req)
    return {s: [(dp["t"], dp["o"], dp["h"], dp["l"], dp["c"], dp["v"]) for dp in bars.get(s, [])] for s in symbols}

def prepare_stock_loaders(config):
    if not ALPACA_AVAILABLE:
        raise RuntimeError("Alpaca API not available.")
        
    print(f"Fetching {config.years_of_data} years of data for {config.symbols}...")
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(config.years_of_data * 365.25))
    
    all_data = _get_stock_data(config.symbols, start_date, end_date)
    if not all_data or not all_data[config.primary_symbol]:
        raise RuntimeError("API fetch failed for the primary symbol.")
        
    dates_per_symbol = {s: {bar[0] for bar in data} for s, data in all_data.items()}
    common_dates = sorted(list(set.intersection(*dates_per_symbol.values())))
    print(f"Found {len(common_dates)} common trading days.")
    
    data_by_date = {s: {bar[0]: bar[1:] for bar in data} for s, data in all_data.items()}
    aligned_data = {s: np.array([data_by_date[s][date] for date in common_dates], dtype=np.float32) for s in config.symbols}
    
    all_features = np.concatenate([aligned_data[s] for s in config.symbols], axis=1)
    
    all_X = []
    for i in range(len(common_dates) - config.sequence_length):
        all_X.append(all_features[i: i + config.sequence_length])
        
    all_X = np.array(all_X, dtype=np.float32)
    all_Y = all_X.copy() # Target is the same as input
    
    indices = np.arange(len(all_X))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - config.val_split_ratio))
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(all_X[indices[:split_idx]]), torch.from_numpy(all_Y[indices[:split_idx]])), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(all_X[indices[split_idx:]]), torch.from_numpy(all_Y[indices[split_idx:]])), batch_size=config.batch_size)
    
    print(f"Data prepared (RAW values): {len(train_loader)} train batches, {len(val_loader)} val batches.")
    return train_loader, val_loader