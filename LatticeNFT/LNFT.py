#LNFT.py

import torch
import torch.nn as nn
from LNFT_block import GaugeConvolutionBlock
from base_modules import PositionalEncoding


class LNFT_EBM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_head = nn.Embedding(config['vocab_size'], config['embed_dim'], padding_idx=config['pad_idx'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'], config['dropout'])
        
        self.core = nn.ModuleList([
            GaugeConvolutionBlock(config['embed_dim'], config['d_ff'], config['dropout'])
            for _ in range(config['num_blocks'])
        ])

        self.energy_head = nn.Sequential(
            nn.Linear(config['embed_dim'], config['d_ff']),
            nn.GELU(),
            nn.Linear(config['d_ff'], 1)
        )

    def forward(self, input_ids):
        q = self.input_head(input_ids)
        return self.forward_from_embeddings(q)

    def forward_from_embeddings(self, q):
        q = self.pos_encoder(q)
        for block in self.core:
            q = block(q)
        
        mean_representation = torch.mean(q, dim=1)
        energy = self.energy_head(mean_representation)
        return energy.squeeze(-1)

    @torch.enable_grad()
    def generate(self, batch_size=1, seq_len=32, hmc_steps=10, hmc_step_size=1e-2, tokenizer=None):
        self.eval()
        device = next(self.parameters()).device
        q = torch.randn(batch_size, seq_len, self.config['embed_dim'], device=device, requires_grad=True)
        p = torch.randn_like(q)

        energy = self.forward_from_embeddings(q).sum()
        grad_S_q, = torch.autograd.grad(energy, q)
        
        p = p - (hmc_step_size / 2) * grad_S_q
        for _ in range(hmc_steps):
            q.data += hmc_step_size * p
            energy = self.forward_from_embeddings(q).sum()
            grad_S_q, = torch.autograd.grad(energy, q)
            p = p - hmc_step_size * grad_S_q
        p = p - (hmc_step_size / 2) * grad_S_q
        p = -p

        embedding_matrix = self.input_head.weight
        q_flat = q.view(-1, self.config['embed_dim'])
        distances = torch.cdist(q_flat, embedding_matrix)
        final_token_ids = torch.argmin(distances, dim=1).view(batch_size, seq_len)
        
        if tokenizer:
            return [tokenizer.decode(ids, skip_special_tokens=True) for ids in final_token_ids]
        return final_token_ids