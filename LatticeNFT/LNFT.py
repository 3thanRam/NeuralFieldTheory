# LNFT.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# You will need your other modules imported here
from base_modules import PositionalEncoding
from LNFT_block import GaugeConvolutionBlock

class ImprovedEnergyHead(nn.Module):
    """
    A more sophisticated energy function that captures local, pairwise,
    and global sequence structure.
    """
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Calculates energy contribution from each token individually
        self.local_energy_mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1)
        )
        
        # Calculates energy from pairwise token interactions (simplified attention)
        self.interaction_proj = nn.Linear(embed_dim, embed_dim // 4) # Project to a smaller dim
        
        # Calculates energy from the overall sequence representation
        self.global_energy_mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1)
        )
        
    def forward(self, x):
        B, N, D = x.shape
        
        # 1. Local energy contribution (sum of per-token energies)
        local_energies = self.local_energy_mlp(x).squeeze(-1)  # Shape: (B, N)
        total_local_energy = local_energies.sum(dim=1)      # Shape: (B,)
        
        # 2. Pairwise interaction energy (dot-product of projected tokens)
        x_proj = self.interaction_proj(x)  # Shape: (B, N, D//4)
        pairwise_scores = torch.bmm(x_proj, x_proj.transpose(1, 2))  # Shape: (B, N, N)
        # Use only upper triangular part to avoid double counting and self-interaction
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        interaction_energy = pairwise_scores.masked_select(mask).view(B, -1).sum(dim=-1) # Shape: (B,)
        
        # 3. Global coherence energy (from the mean representation)
        global_repr = x.mean(dim=1)
        global_energy = self.global_energy_mlp(global_repr).squeeze(-1) # Shape: (B,)
        
        # Combine the energies. Here we just sum them. Learnable weights could be added.
        total_energy = total_local_energy + interaction_energy + global_energy
        
        return total_energy


class LNFT_EBM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_head = nn.Embedding(
            config['vocab_size'], 
            config['embed_dim'], 
            padding_idx=config['pad_idx']
        )
        nn.init.normal_(self.input_head.weight, mean=0, std=0.02)
        
        self.pos_encoder = PositionalEncoding(
            d_model=config['embed_dim'], 
            dropout=config['dropout']
        )
        
        self.core = nn.ModuleList([
            GaugeConvolutionBlock(
                config['embed_dim'], 
                config['d_ff'], 
                config['dropout']
            ) for _ in range(config['num_blocks'])
        ])

        self.energy_head = ImprovedEnergyHead(config['embed_dim'], config['d_ff'])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids):
        q = self.input_head(input_ids)
        return self.forward_from_embeddings(q)

    def forward_from_embeddings(self, q):
        q = self.pos_encoder(q)
        for block in self.core:
            q = block(q) 
        
        energy = self.energy_head(q)
        return energy

    @torch.enable_grad()
    def generate(self, batch_size, seq_len, steps, step_size, noise_scale, tokenizer=None):
        self.eval()
        device = next(self.parameters()).device
        
        # Start from random embeddings with small variance
        q = torch.randn(
            batch_size, seq_len, self.config['embed_dim'], 
            device=device, requires_grad=True
        ) * 0.1
        
        # Use Langevin dynamics for high-quality generation
        q = self._langevin_dynamics(q, steps=steps, step_size=step_size, noise_scale=noise_scale)
        
        # Convert final continuous embeddings back to discrete token IDs
        embedding_matrix = self.input_head.weight
        q_flat = q.view(-1, self.config['embed_dim'])
        distances = torch.cdist(q_flat, embedding_matrix)
        final_token_ids = torch.argmin(distances, dim=1).view(batch_size, seq_len)
        
        if tokenizer:
            return [tokenizer.decode(ids, skip_special_tokens=True) for ids in final_token_ids]
        return final_token_ids

    @torch.enable_grad()
    def _langevin_dynamics(self, q, steps, step_size, noise_scale):
        """Helper for running Langevin dynamics with proper gradient clipping."""
        for _ in range(steps):
            # Set requires_grad=True on the tensor for this iteration
            q.requires_grad_(True)

            energy = self.forward_from_embeddings(q).sum()

            # Calculate the gradient of the energy w.r.t. the input q
            grad_q, = torch.autograd.grad(energy, q, retain_graph=False)
            torch.nn.utils.clip_grad_norm_([grad_q], max_norm=10.0)
        
            # Detach q from the graph before the in-place update
            q = q.detach()
            # Langevin update
            noise = torch.randn_like(q) * noise_scale
            q.data = q.data - step_size * grad_q + noise

        return q