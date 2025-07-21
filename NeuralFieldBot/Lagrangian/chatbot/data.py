# chatbot/data.py - Prepares data for the chatbot task.
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

class TextDataset(Dataset):
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

def prepare_chatbot_loaders(config):
    print("--- Preparing Chatbot Data ---")
    tokenizer_dir = os.path.join(os.path.dirname(config.ckpt_path), "chatbot_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    except:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"})
        tokenizer.save_pretrained(tokenizer_dir)

    config.vocab_size = len(tokenizer)
    config.pad_idx = tokenizer.pad_token_id
    if config.pad_idx is None:
        raise ValueError("Tokenizer pad token not found.")
        
    dataset = load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)
    corpus = "\n".join([ex["sentence"] for split in dataset for ex in dataset[split] if ex["sentence"].strip()])
    
    chunk_len = config.embed_dim - 2
    chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus) - chunk_len + 1, chunk_len // 2)]
    random.shuffle(chunks)
    
    split_idx = int(len(chunks) * 0.9)
    train_texts, val_texts = chunks[:split_idx], chunks[split_idx:]
    print(f"Data: {len(train_texts)} train, {len(val_texts)} val chunks.")
    
    train_loader = DataLoader(TextDataset(train_texts, tokenizer, config.embed_dim), batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TextDataset(val_texts, tokenizer, config.embed_dim), batch_size=config.batch_size, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, tokenizer