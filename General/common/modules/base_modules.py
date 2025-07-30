# common/modules/base_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class DynamicNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True); std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps), (mean, std)
    def inverse(self, x_norm, stats):
        mean, std = stats
        return x_norm * (std + self.eps) + mean

class RelativeNorm(nn.Module):
    def __init__(self, eps=1e-8, mode='divide'):
        super().__init__()
        self.eps = eps; self.mode = mode
    def forward(self, x):
        ref = x[:, -1:, :]
        if self.mode == 'divide': normalized_x = x / (ref + self.eps)
        else: normalized_x = x - ref
        return normalized_x, (ref,)
    def inverse(self, x_norm, stats):
        (ref,) = stats
        if self.mode == 'divide': return x_norm * (ref + self.eps)
        else: return x_norm + ref

class LastPointNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        last = x[:, -1:, :]; std = x.std(dim=1, keepdim=True)
        return (x - last) / (std + self.eps), (last, std)
    def inverse(self, x_norm, stats):
        last, std = stats
        return (x_norm * (std + self.eps)) + last

class ReturnNorm(nn.Module):
    """
    Transforms a raw price series into a return series (P_t / P_{t-1}) and back.
    The inverse operation is a vectorized cumulative product, which correctly
    reconstructs the price path from a series of returns.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """ Normalizes the input into a return series. """
        # x shape: [batch, seq_len, features]
        
        # Prepend the first value to create the "previous step" tensor
        # This ensures the first return is P_1 / P_0
        prev_steps = torch.cat([x[:, 0:1, :], x[:, :-1, :]], dim=1)
        
        # The statistic needed for inversion is the price just BEFORE the sequence starts.
        # So we take the first value of the `prev_steps` tensor.
        anchor = prev_steps[:, 0:1, :]
        
        # Calculate returns: R_t = P_t / P_{t-1}
        normalized_x = x / (prev_steps + self.eps)
        
        return normalized_x, (anchor,)

    def inverse(self, x_norm, stats):
        """
        Un-normalizes a sequence of returns back into a price series
        using the initial anchor value and a vectorized cumulative product.
        """
        (anchor,) = stats
        # x_norm is a sequence of predicted returns, e.g., shape [batch, seq_len, features]
        # where values are like [1.01, 0.99, 1.02, ...]

        # Use torch.cumprod to get the cumulative return factors.
        # This is a vectorized operation, safe for autograd.
        # [R_1, R_2, R_3] -> [R_1, R_1*R_2, R_1*R_2*R_3]
        cumulative_returns = torch.cumprod(x_norm, dim=1)
        
        # Multiply the anchor price by the cumulative returns to get the price path.
        # This correctly reconstructs the path:
        # [P_0*R_1, P_0*R_1*R_2, P_0*R_1*R_2*R_3, ...]
        return anchor * cumulative_returns
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

class ParallelForceBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.potential_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.noise_level = nn.Parameter(torch.zeros(1))
        self.force_conv = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
            groups=embed_dim
        )

    def forward(self, q):
        potential_field = self.potential_mlp(q)
        force_permuted = self.force_conv(potential_field.permute(0, 2, 1))
        
        if self.training:
            noise_scale = F.softplus(self.noise_level)
            noise = torch.randn_like(force_permuted) * noise_scale
            force_permuted = force_permuted + noise
            
        return force_permuted.permute(0, 2, 1)

class MultiHeadForceBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, hidden_dim, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.original_embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            self.head_dim = (embed_dim + num_heads - 1) // num_heads
            self.padded_embed_dim = self.head_dim * num_heads
        else:
            self.head_dim = embed_dim // num_heads
            self.padded_embed_dim = embed_dim
        self.heads = nn.ModuleList([
            ParallelForceBlock(self.head_dim, hidden_dim, kernel_size, dropout_rate)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(self.padded_embed_dim, self.original_embed_dim)

    def forward(self, q):
        if self.original_embed_dim != self.padded_embed_dim:
            pad_size = self.padded_embed_dim - self.original_embed_dim
            padding = torch.zeros(*q.shape[:-1], pad_size, device=q.device, dtype=q.dtype)
            q_padded = torch.cat([q, padding], dim=-1)
        else:
            q_padded = q
        q_chunks = torch.chunk(q_padded, self.num_heads, dim=-1)
        head_outputs = [head(chunk) for head, chunk in zip(self.heads, q_chunks)]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.out_proj(concatenated)