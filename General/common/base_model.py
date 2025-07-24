#base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.modules.base_modules import DynamicNorm





class BaseModel(nn.Module):
    """ A single, universal model that handles different tasks and normalization. """
    def __init__(self,**kwargs):
        super().__init__()
        config=kwargs['config']
        
        self.use_normalization=config['use_normalization']
        self.use_tokenization=config['use_tokenization']
        
        if self.use_normalization:
            self.normalizer = DynamicNorm()
        else:
            self.normalizer =None
        
        if self.use_tokenization:
            self.input_head = nn.Embedding(config['vocab_size'], config['embed_dim'], padding_idx=config['pad_idx'])
            self.output_head = nn.Linear(config['embed_dim'], config['vocab_size'])
        else:
            self.input_head = nn.Linear(config['num_input_features'], config['embed_dim'])
            self.output_head = nn.Linear(config['embed_dim'], config['num_output_predictions'])
            
        
        self.core = kwargs['core_network']

    def forward(self, x, return_internals=False):
        if self.normalizer:
            x, norm_mean, norm_std = self.normalizer.forward(x)
        initial_vectors = self.input_head(x)

        if return_internals:
            final_state, internals = self.core(initial_vectors, return_internals=True)
        else:
            final_state = self.core(initial_vectors, return_internals=False)

        predictions = self.output_head(final_state)

        if self.normalizer:
            predictions = self.normalizer.inverse(predictions, norm_mean, norm_std)
        
        if return_internals:
            return predictions, internals
        return predictions

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
        
        else:
            current_sequence = start_input.clone().to(device)
            predicted_steps = []
            for _ in range(max_new_tokens):
                next_step_prediction = self.forward(current_sequence)[:, -1:, :]
                predicted_steps.append(next_step_prediction.cpu())
                current_sequence = torch.cat([current_sequence[:, 1:, :], next_step_prediction], dim=1)
            return torch.cat(predicted_steps, dim=1).squeeze(0)