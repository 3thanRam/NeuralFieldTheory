# common/modules/base_modules.py
import torch
import torch.nn as nn
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
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + self.eps), mean, std

    def inverse(self, x_norm, mean, std):
        return x_norm * (std + self.eps) + mean

class LearnableFrFT(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Use separate parameters for each dimension
        self.alpha = nn.Parameter(torch.randn(embed_dim // 2) * 0.01)

    def forward(self, q, p):
        # Split into pairs and apply rotation per pair
        q_pairs = q.view(*q.shape[:-1], -1, 2)
        p_pairs = p.view(*p.shape[:-1], -1, 2)
        
        c, s = torch.cos(self.alpha), torch.sin(self.alpha)
        c, s = c.unsqueeze(0).unsqueeze(0), s.unsqueeze(0).unsqueeze(0)
        
        q_rot = q_pairs * c.unsqueeze(-1) + p_pairs * s.unsqueeze(-1)
        p_rot = -q_pairs * s.unsqueeze(-1) + p_pairs * c.unsqueeze(-1)
        
        return q_rot.flatten(-2), p_rot.flatten(-2)
    def inverse(self, q, p):
        # Split into pairs and apply rotation per pair
        q_pairs = q.view(*q.shape[:-1], -1, 2)
        p_pairs = p.view(*p.shape[:-1], -1, 2)
        
        c, s = torch.cos(self.alpha), -torch.sin(self.alpha)
        c, s = c.unsqueeze(0).unsqueeze(0), s.unsqueeze(0).unsqueeze(0)
        
        q_rot = q_pairs * c.unsqueeze(-1) + p_pairs * s.unsqueeze(-1)
        p_rot = -q_pairs * s.unsqueeze(-1) + p_pairs * c.unsqueeze(-1)
        
        return q_rot.flatten(-2), p_rot.flatten(-2)

class ParallelForceBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.potential_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        #self.force_conv = nn.Conv1d(
        #    in_channels=embed_dim,
        #    out_channels=embed_dim,
        #    kernel_size=kernel_size,
        #    padding=(kernel_size - 1) // 2,
        #    groups=embed_dim # Now this is always correct
        #)
        self.force_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=min(embed_dim // 4, 32)
        )


    def forward(self, q):
        potential_field = self.potential_mlp(q)
        force = self.force_conv(potential_field.permute(0, 2, 1))
        return force.permute(0, 2, 1)