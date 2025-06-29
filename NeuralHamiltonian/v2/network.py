import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import config

class LocalProxyStatistics(nn.Module):
    """
    Computes a Hamiltonian contribution from low-dimensional summary statistics
    for each state (q, p) in a parallel batch. This is a computationally cheap
    way to introduce expressive, non-linear interactions.
    """
    def __init__(self):
        super().__init__()
        self.w_mean_q = nn.Parameter(torch.randn(1) * 0.01)
        self.w_std_q = nn.Parameter(torch.randn(1) * 0.01)
        self.w_mean_p = nn.Parameter(torch.randn(1) * 0.01)
        self.w_std_p = nn.Parameter(torch.randn(1) * 0.01)
        self.w_mean_q_std_p = nn.Parameter(torch.randn(1) * 0.01)
        self.w_std_q_mean_p = nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Calculates the contribution for each state in the flattened batch.
        q, p shape: (N, d_embedding) where N = batch_size * seq_len
        Returns a tensor of shape (N,)
        """
        mean_q = torch.mean(q, dim=-1, keepdim=True)
        std_q = torch.std(q, dim=-1, keepdim=True)
        mean_p = torch.mean(p, dim=-1, keepdim=True)
        std_p = torch.std(p, dim=-1, keepdim=True)
        
        h_proxy_contrib = (
            self.w_mean_q * mean_q + self.w_std_q * std_q +
            self.w_mean_p * mean_p + self.w_std_p * std_p +
            self.w_mean_q_std_p * mean_q * std_p +
            self.w_std_q_mean_p * std_q * mean_p
        )
        return h_proxy_contrib.squeeze(-1)

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

class MomentumNet(nn.Module):
    """
    Learns momentum p from position q. This version is simplified.
    To re-introduce derivatives, change the input layer to d_embedding*3 and
    add the internal derivative calculations back to the forward pass.
    """
    def __init__(self, d_embedding, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embedding, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_embedding)
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)

class ParallelHamiltonianBlock(nn.Module):
    """
    This block computes the Hamiltonian for all sequence elements in parallel.
    The depth of the model is now controlled by the MLP's depth, not a for-loop.
    """
    def __init__(self, d_embedding, d_hidden_dim, num_mlp_layers, dropout):
        super().__init__()
        self.proxy_stats = LocalProxyStatistics()
        
        # Build a deeper MLP to give the model expressive power
        layers = [nn.Linear(d_embedding * 2, d_hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_mlp_layers - 1):
            layers.extend([nn.Linear(d_hidden_dim, d_hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(d_hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self.h_offset = nn.Parameter(torch.randn(1))
        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding))
        self.coef_linear_p = nn.Parameter(torch.randn(d_embedding))
        # Note: Non-local quadratic terms like einsum('bid,dk,bjd... are omitted
        # for this parallel strategy as they break the independence between time steps.
        # The deep MLP and proxy stats are now responsible for learning non-linear interactions.

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian H for each state in the flattened batch.
        q, p shape: (N, d_embedding)
        Returns H_scalar: a single scalar value (the sum over all states in the batch)
        """
        # Linear terms (local to each time step)
        linear_q = torch.einsum('nd,d->n', q, self.coef_linear_q)
        linear_p = torch.einsum('nd,d->n', p, self.coef_linear_p)
        
        # Proxy statistics (local to each time step)
        proxy_contrib = 0#self.proxy_stats(q, p)
        
        # Deep MLP (local to each time step)
        mlp_contrib = self.mlp(torch.cat([q, p], dim=-1)).squeeze(-1)

        # Total Hamiltonian for each state
        H_per_state = self.h_offset + linear_q + linear_p + proxy_contrib + mlp_contrib
        
        # Return the sum, which is the scalar needed for autograd to compute parallel gradients
        return H_per_state.sum()

class HamiltonianModel(nn.Module):
    """
    A parallel-in-time Hamiltonian model. It evolves all time steps of the
    input sequence simultaneously through a single, deep Hamiltonian block.
    """
    def __init__(self, num_layers, input_dim, d_embedding, d_hidden_dim, output_dim, sequence_length, timestep, dropout):
        super().__init__()
        self.d_embedding = d_embedding
        self.timestep = timestep
        self.clip_value = 1.0

        self.input_projection = nn.Linear(input_dim, d_embedding)
        self.pos_encoder = PositionalEncoding(d_embedding)
        self.dropout = nn.Dropout(dropout)
        self.momentum_net = MomentumNet(d_embedding, d_hidden_dim // 2)
        
        # We now have ONE parallel block, not a list.
        # The `num_layers` config parameter now controls the depth of the MLP inside this block.
        self.block = ParallelHamiltonianBlock(d_embedding, d_hidden_dim, num_layers, dropout)
        
        self.norm = nn.LayerNorm(d_embedding)
        self.lm_head = nn.Linear(d_embedding, output_dim)

    def update_vars(self, q: torch.Tensor, p: torch.Tensor, H_scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a parallel update for the entire flattened batch of states.
        q, p shape: (N, d_embedding), H_scalar: a single scalar value
        """
        grad_H_q = torch.autograd.grad(H_scalar, q, create_graph=True)[0]
        grad_H_p = torch.autograd.grad(H_scalar, p, create_graph=True)[0]
        
        # Using the stable semi-implicit Euler integrator
        p_new = p - self.timestep * grad_H_q
        q_new = q + self.timestep * grad_H_p

        # Clamp for stability
        q_new = torch.clamp(q_new, -10, 10)
        p_new = torch.clamp(p_new, -10, 10)
        
        return q_new, p_new

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q_initial = self.dropout(self.pos_encoder(self.input_projection(x)))
        q = q_initial.clone().requires_grad_(True)
        p = self.momentum_net(q).requires_grad_(True)

        q_flat = q.view(-1, self.d_embedding)
        p_flat = p.view(-1, self.d_embedding)

        H_scalar = self.block(q_flat, p_flat)
        q_new_flat, p_new_flat = self.update_vars(q_flat, p_flat, H_scalar)

        q_final = q_new_flat.view(batch_size, seq_len, -1)
        
        output = self.norm(q_final)
        logits = self.lm_head(output)
        return logits

    @torch.no_grad()
    def generate(self, input_sequence: torch.Tensor, n_to_pred: int):
        self.eval()
        from config import config # Local import
        symbols = config["symbols"]
        primary_symbol = config["primary_symbol"]
        
        try:
            primary_symbol_idx = symbols.index(primary_symbol)
        except ValueError:
            raise ValueError(f"primary_symbol '{primary_symbol}' not found in symbols list in config.")
        
        start_feature_idx = primary_symbol_idx * 4
        end_feature_idx = start_feature_idx + 4
        
        predictions = []
        current_input_seq = input_sequence.to(next(self.parameters()).device)

        for _ in range(n_to_pred):
            input_for_pred = current_input_seq[:, -config["sequence_length"]:, :]

            with torch.enable_grad():
                # This automatically uses the new parallel forward pass
                full_prediction_sequence = self(input_for_pred)

            next_primary_ohlc = full_prediction_sequence[:, -1:, :]
            predictions.append(next_primary_ohlc.cpu().numpy())
            last_full_step = current_input_seq[:, -1:, :].clone()
            last_full_step[:, :, start_feature_idx:end_feature_idx] = next_primary_ohlc
            current_input_seq = torch.cat([current_input_seq, last_full_step], dim=1)

        return np.concatenate(predictions, axis=1).squeeze(0)