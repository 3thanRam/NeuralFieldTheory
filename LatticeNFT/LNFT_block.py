# LNFT_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class LowRankGaugeFieldGenerator(nn.Module):
    def __init__(self, embed_dim, mlp_dim, rank):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, mlp_dim),
            nn.GELU(),
            spectral_norm(nn.Linear(mlp_dim, 2 * embed_dim * rank))
        )
        with torch.no_grad():
            nn.init.xavier_uniform_(self.mlp[0].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.xavier_uniform_(self.mlp[-1].weight_orig, gain=0.01)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, combined_pairs):
        factors = self.mlp(combined_pairs)
        u_factors = factors[..., :self.embed_dim * self.rank]
        v_factors = factors[..., self.embed_dim * self.rank:]
        u = u_factors.reshape(-1, self.embed_dim, self.rank)
        v = v_factors.reshape(-1, self.embed_dim, self.rank)
        U_ij = torch.matmul(u, v.transpose(-1, -2))
        identity = torch.eye(self.embed_dim, device=U_ij.device).expand_as(U_ij)
        return 0.98 * identity + 0.02 * U_ij


class GaugeConvolutionBlock(nn.Module):
    """
    Final, stabilized, and compiler-friendly Gauge Convolution block.
    Uses an explicit for-loop for local windowing to ensure torch.compile compatibility.
    """
    def __init__(self, embed_dim, d_ff, dropout=0.1, rank=16, window_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.window_size = window_size
        
        self.gauge_generator = LowRankGaugeFieldGenerator(embed_dim, d_ff, rank)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.gauge_scale = nn.Parameter(torch.tensor(0.01))
        self.mlp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, q):
        phi_res = q
        phi = self.norm1(q)
        batch_size, seq_len, embed_dim = phi.shape
        
        # --- ROBUST LOCAL GAUGE FIELD COMPUTATION (with for-loop) ---
        
        aggregated_outputs = []
        
        # Pad the sequence for easy windowing at the edges.
        pad_size = self.window_size - 1
        half_pad = pad_size // 2
        phi_padded = F.pad(phi.transpose(1, 2), (half_pad, pad_size - half_pad)).transpose(1, 2)
        
        for i in range(seq_len):
            # The "source" token for this window
            phi_i = phi[:, i, :] # Shape: (B, D)
            
            # The "context" window from the padded sequence
            phi_j_window = phi_padded[:, i : i + self.window_size, :] # Shape: (B, k, D)
            
            # Expand phi_i to match the window dimension for the generator
            phi_i_expanded = phi_i.unsqueeze(1).expand(-1, self.window_size, -1) # Shape (B, k, D)
            
            # Concatenate all pairs for this window and flatten for the MLP call
            combined_pairs_flat = torch.cat([phi_i_expanded, phi_j_window], dim=-1).reshape(-1, 2 * embed_dim)
            
            # Generate all U_local matrices for this window in one go
            U_local_flat = self.gauge_generator(combined_pairs_flat)
            U_local = U_local_flat.view(batch_size, self.window_size, embed_dim, embed_dim)
            
            # Apply local gauge transformations and aggregate
            # einsum: b(batch), w(window), d,e(embed_dim)
            local_contrib = torch.einsum('bwde,bwe->bd', U_local, phi_j_window)
            
            aggregated_outputs.append(local_contrib)
        
        # Stack the results from each step to form the final tensor
        aggregated_info = torch.stack(aggregated_outputs, dim=1) # Shape (B, N, D)
        
        # --- Rest of the block ---
        q = phi_res + self.gauge_scale * self.dropout(aggregated_info)
        
        phi_res = q
        phi = self.norm2(q)
        mlp_output = self.mlp(phi)
        
        q_final = phi_res + self.mlp_scale * self.dropout(mlp_output)
        
        return q_final