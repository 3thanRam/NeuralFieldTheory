# utils/data_loaders.py
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader, TensorDataset

from transformers import GPT2TokenizerFast
from datasets import load_dataset
from datetime import datetime, timedelta


class TextDataset(Dataset):
    """ Generic PyTorch Dataset for handling text tokenization. """
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(
            f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}",
            max_length=self.max_len,
            truncation=True,
            padding="max_length"
        )
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


# --- Stockbot Data Dependencies ---
try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    
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

# --- Chatbot Data Loader ---
def prepare_chatbot_loaders(config):
    """
    Prepares the tokenizer and data loaders for the NLP task.
    """
    print("--- Preparing Chatbot Data ---")
    
    tokenizer_dir = config.tokenizer_directory
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        print("Loaded existing tokenizer.")
    except (OSError, ValueError,TypeError):
        print("Initializing new tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"})
        tokenizer.save_pretrained(tokenizer_dir)

    dataset = load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)
    corpus = "\n".join([ex["sentence"] for split in dataset for ex in dataset[split] if ex["sentence"].strip()])
    
    chunk_len = config.embed_dim - 2
    chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus) - chunk_len + 1, chunk_len // 2)]
    random.shuffle(chunks)
    
    split_idx = int(len(chunks) * 0.9)
    train_texts, val_texts = chunks[:split_idx], chunks[split_idx:]
    print(f"Data prepared: {len(train_texts)} train chunks, {len(val_texts)} val chunks.")
    
    train_loader = DataLoader(TextDataset(train_texts, tokenizer, config.embed_dim), batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TextDataset(val_texts, tokenizer, config.embed_dim), batch_size=config.batch_size, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, tokenizer

# --- Stockbot Data Loader ---
def _get_stock_data(symbols, start_date, end_date):
    """ Internal helper function to fetch raw data from Alpaca API. """
    if not isinstance(symbols, list):
        symbols = [symbols]
    
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(request_params)
    
    stock_data = {}
    for s in symbols:
        if s in bars:
            stock_data[s] = [(dp["t"], dp["o"], dp["h"], dp["l"], dp["c"], dp["v"]) for dp in bars[s]]
        else:
            stock_data[s] = []
    return stock_data

def prepare_stock_loaders(config):
    """
    Prepares data loaders for the stock prediction task.
    """
    if not ALPACA_AVAILABLE:
        raise RuntimeError("Cannot prepare stock data. `alpaca-py` is not installed or API keys are not set.")
        
    print(f"Fetching {config.years_of_data} years of data for {config.symbols}...")
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(config.years_of_data * 365.25))
    
    all_data = _get_stock_data(config.symbols, start_date, end_date)
    if not all_data or not all_data[config.primary_symbol]:
        raise RuntimeError("API fetch failed for the primary symbol.")
        
    dates_per_symbol = {symbol: {bar[0] for bar in data} for symbol, data in all_data.items()}
    common_dates = sorted(list(set.intersection(*dates_per_symbol.values())))
    print(f"Found {len(common_dates)} common trading days.")
    
    data_by_date = {symbol: {bar[0]: bar[1:] for bar in data} for symbol, data in all_data.items()}
    aligned_data = {symbol: np.array([data_by_date[symbol][date] for date in common_dates], dtype=np.float32) for symbol in config.symbols}
    
    all_features = np.concatenate([aligned_data[symbol] for symbol in config.symbols], axis=1)
    
    all_X = []
    for i in range(len(common_dates) - config.sequence_length):
        all_X.append(all_features[i: i + config.sequence_length])
        
    all_X = np.array(all_X, dtype=np.float32)
    all_Y = all_X.copy()
    
    indices = np.arange(len(all_X))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - config.val_split_ratio))
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(all_X[indices[:split_idx]]), torch.from_numpy(all_Y[indices[:split_idx]])), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(all_X[indices[split_idx:]]), torch.from_numpy(all_Y[indices[split_idx:]])), batch_size=config.batch_size)
    
    print(f"Data prepared (RAW values): {len(train_loader)} train batches, {len(val_loader)} val batches.")
    return train_loader, val_loader