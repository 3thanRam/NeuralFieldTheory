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

class PhaseSpaceTransform(nn.Module):
    def __init__(self, embed_dim, dt=0.1):
        super().__init__()
        self.dt = dt
        # This would be your actual LearnableFrFT module
        self.frft = LearnableFrFT(embed_dim) 
        # For this example, let's make a simple learnable combination
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_q_dot = nn.Linear(embed_dim, embed_dim)
        self.inv_proj_q= nn.Linear(embed_dim, embed_dim)

    def forward(self, q):
        """
        Takes a single tensor 'q' and returns a single transformed tensor.
        """
        # 1. Calculate q_dot internally
        q_dot = torch.zeros_like(q)
        q_dot[:, 1:] = (q[:, 1:] - q[:, :-1]) / self.dt
        
        # 2. Apply the transform (e.g., your FrFT) internally
        q, q_dot = self.frft(q, q_dot) 
        
        # 3. Combine the transformed parts into a single output tensor.
        # A simple linear combination is a good, learnable way to do this.
        q_transformed = self.proj_q(q) + self.proj_q_dot(q_dot)
        
        return q_transformed

    def inverse(self, q_transformed):
        """
        A placeholder for the inverse transform. In a real FrFT, this would be
        calling the inverse FrFT. For our simple linear combination,
        an exact inverse isn't possible, but another linear layer is a
        common way to approximate the return to the original space.
        """
        # This is a conceptual inverse, not a mathematical one.
        q_final = self.inv_proj_q(q_transformed) 
        return q_final