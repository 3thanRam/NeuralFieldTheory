# common/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.base_modules import ReturnNorm # We only need ReturnNorm

class BaseModel(nn.Module):
    def __init__(self, core_network, config):
        super().__init__()
        self.config = config
        self.core = core_network
        
        self.use_tokenization = config.get('use_tokenization', False)
        self.use_normalization = config.get('use_normalization', False) # This is for the 'returns' transform
        
        if self.use_normalization:
            self.normalizer = ReturnNorm()
        else:
            self.normalizer = None
        
        if self.use_tokenization:
            self.input_head = nn.Embedding(config['vocab_size'], config['embed_dim'], padding_idx=config['pad_idx'])
            self.output_head = nn.Linear(config['embed_dim'], config['vocab_size'])
        else:
            self.input_head = nn.Linear(config['num_input_features'], config['embed_dim'])
            self.output_head = nn.Linear(config['embed_dim'], config['num_output_predictions'])

    def forward(self, x, return_internals=False):
        # The input `x` is always raw data (prices or token IDs)
        
        if self.use_normalization:
            # For stocks, convert raw prices to returns. The model core sees returns.
            x_norm, _ = self.normalizer.forward(x) # We don't need the stats for training
            initial_vectors = self.input_head(x_norm)
        else: # For NLP
            initial_vectors = self.input_head(x)

        if return_internals:
            final_state, internals = self.core(initial_vectors, return_internals=True)
        else:
            final_state = self.core(initial_vectors, return_internals=False)

        # The model's final output is ALWAYS in the normalized space
        # For stocks, this means it predicts returns.
        predictions_norm = self.output_head(final_state)
        
        if return_internals:
            return predictions_norm, internals
        return predictions_norm

    @torch.no_grad()
    def generate(self, start_input, max_new_tokens, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        
        if self.use_tokenization:
            tokenizer = kwargs.get('tokenizer')
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for text generation.")
            ids = tokenizer.encode(start_input, return_tensors='pt').to(device) if isinstance(start_input, str) else start_input.to(device)
            for _ in range(max_new_tokens):
                logits = self.forward(ids)
                next_logits = logits[:, -1, :] / kwargs.get('temperature', 0.8)
                top_k = kwargs.get('top_k', 20)
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, -1]] = -float('Inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                ids = torch.cat((ids, next_id), dim=1)
                
                if hasattr(tokenizer, 'eos_token_id') and next_id.item() == tokenizer.eos_token_id:
                    break
            
            return tokenizer.decode(ids[0], skip_special_tokens=True)
        else: # Timeseries
            # This loop now correctly handles the raw price -> return -> raw price conversion
            current_sequence_raw = start_input.clone().to(device)
            predicted_prices_list = []
            
            for _ in range(max_new_tokens):
                # The forward pass takes raw prices and outputs predicted returns
                predicted_returns = self.forward(current_sequence_raw)
                
                # We only care about the return for the very last time step
                last_step_return = predicted_returns[:, -1:, :]
                
                # Get the last known price to un-normalize
                last_known_price = current_sequence_raw[:, -1:, :]
                
                # Calculate the next predicted price
                next_price_pred = last_known_price * last_step_return
                
                predicted_prices_list.append(next_price_pred.cpu())
                
                # Update the input sequence with the new raw price prediction
                current_sequence_raw = torch.cat([current_sequence_raw[:, 1:, :], next_price_pred], dim=1)
            
            return torch.cat(predicted_prices_list, dim=1).squeeze(0)