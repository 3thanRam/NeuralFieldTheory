# In your main training script

from transformers import GPT2TokenizerFast # <<< CORRECT IMPORT
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def setup_data(config):
    tokenizer = GPT2TokenizerFast.from_pretrained(config['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    config['vocab_size'] = len(tokenizer)
    config['pad_idx'] = tokenizer.pad_token_id
    #print(f"Tokenizer -> Vocab: {config['vocab_size']}, Pad ID: {config['pad_idx']}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize(e): return tokenizer(e["text"], truncation=False)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    def group(e):
        block_size = config['seq_len']
        concat = {k: sum(e[k], []) for k in e.keys()}
        total = len(concat[list(e.keys())[0]])
        total = (total // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total, block_size)] for k, t in concat.items()}
        return result

    lm_dataset = tokenized.map(group, batched=True)
    lm_dataset.set_format(type="torch")
    
    train_dataset = lm_dataset.filter(lambda e: len(e['input_ids']) == config['seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    return train_loader, tokenizer