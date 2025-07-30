#base_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Correct calculation using the d_model parameter
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize pe tensor with correct integer sizes
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # The pe buffer has shape [max_len, 1, d_model]. We need to select the
        # right number of positions and match the batch dimension.
        # Original code had a transpose which might be needed if input is (N, B, D)
        # For (B, N, D) input, this is correct:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class LearnableFrFT(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.alpha_predictor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, q, p):
        state_summary = q.mean(dim=1).detach()
        alpha = self.alpha_predictor(state_summary)
        c, s = torch.cos(alpha), torch.sin(alpha)
        self.last_alpha = alpha
        return q * c.unsqueeze(-1) + p * s.unsqueeze(-1), -q * s.unsqueeze(-1) + p * c.unsqueeze(-1)

    def inverse(self, q, p):
        if not hasattr(self, 'last_alpha'):
            raise RuntimeError("Forward must be called before inverse for DynamicFrFT.")
        # The inverse is a rotation by -alpha
        c, s = torch.cos(self.last_alpha), -torch.sin(self.last_alpha) 
        return q * c.unsqueeze(-1) + p * s.unsqueeze(-1), -q * s.unsqueeze(-1) + p * c.unsqueeze(-1)
    
class Normalize(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True); std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps), (mean, std)
    def inverse(self, x_norm, stats):
        mean, std = stats
        return x_norm * (std + self.eps) + mean