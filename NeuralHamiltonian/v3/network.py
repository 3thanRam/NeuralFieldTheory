import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import config

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

class SubspaceInteraction(nn.Module):
    """
    This is the f_theta network. It learns the non-linear interaction energy
    by projecting q and p into S subspaces, creating product features in a
    vectorized way, and then processing the combined representation.
    """
    def __init__(self, d_embedding, d_hidden_dim, num_subspaces=4, subspace_dim=64):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        
        self.q_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)
        ])
        self.p_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)
        ])
        
        # --- The input dimension calculation remains the same ---
        # The number of features we create hasn't changed, only how we create them.
        num_q_q_products = num_subspaces * num_subspaces # Now all pairs for simplicity
        num_p_p_products = num_subspaces * num_subspaces
        num_q_p_products = num_subspaces * num_subspaces
        
        num_original_features = num_subspaces * 2
        num_product_features = num_q_q_products + num_p_p_products + num_q_p_products
        
        interaction_input_dim = (num_original_features + num_product_features) * subspace_dim
        
        self.interaction_net = nn.Sequential(
            nn.Linear(interaction_input_dim, d_hidden_dim),
            nn.GELU(),
            nn.Linear(d_hidden_dim, d_hidden_dim),
            nn.GELU(),
            nn.Linear(d_hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q, p shape: (batch_size, seq_len, d_embedding)
        
        # Project q and p into their respective subspaces
        q_subs = [proj(q) for proj in self.q_projections]
        p_subs = [proj(p) for proj in self.p_projections]

        # Stack the subspaces into tensors of shape (batch, seq, num_subspaces, subspace_dim)
        Q_stacked = torch.stack(q_subs, dim=2)
        P_stacked = torch.stack(p_subs, dim=2)

        # --- NEW: Vectorized creation of product features using einsum ---
        # 'bsid' means (batch, seq, subspace_i, dim)
        # 'bsjd' means (batch, seq, subspace_j, dim)
        # The result 'bsijd' will have shape (batch, seq, subspace_i, subspace_j, dim)
        
        # q_i * q_j products
        qq_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, Q_stacked)
        
        # p_i * p_j products
        pp_products = torch.einsum('bsid, bsjd -> bsijd', P_stacked, P_stacked)

        # q_i * p_j products
        qp_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, P_stacked)

        # --- Flatten all features for the MLP ---
        # Flatten the original subspaces
        flat_q_subs = Q_stacked.flatten(start_dim=2) # Shape: (batch, seq, num_subspaces * subspace_dim)
        flat_p_subs = P_stacked.flatten(start_dim=2)
        
        # Flatten the product features
        # The shape is (b, s, i, j, d). We want to flatten i, j, and d together.
        flat_qq = qq_products.flatten(start_dim=2)
        flat_pp = pp_products.flatten(start_dim=2)
        flat_qp = qp_products.flatten(start_dim=2)
        
        # Combine all flattened features
        all_features = [flat_q_subs, flat_p_subs, flat_qq, flat_pp, flat_qp]
        concatenated_features = torch.cat(all_features, dim=-1)
        
        # Pass the rich representation through the interaction network
        energy_contribution = self.interaction_net(concatenated_features)
        
        return energy_contribution.sum()

class HamiltonianBlock(nn.Module):
    """
    Computes the total Hamiltonian H by combining explicit physical terms
    (linear, quadratic) with a learned non-linear interaction term.
    """
    def __init__(self, d_embedding, d_hidden_dim, sequence_length, timestep, dropout):
        super().__init__()
        
        # --- Explicit Inductive Bias Terms ---
        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding))
        self.coef_linear_p = nn.Parameter(torch.randn(d_embedding))
        self.coef_quadratic_qq = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_pp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_qp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        
        # --- Learned Non-linear Interaction Term ---
        self.interaction_module = SubspaceInteraction(
            d_embedding=d_embedding,
            d_hidden_dim=d_hidden_dim,
            num_subspaces=6, # Hyperparameter: how many "lenses" to view the data through
            subspace_dim=d_hidden_dim//4  # Hyperparameter: dimension of each "lens"
        )
        self.h_offset = nn.Parameter(torch.randn(1))

    def forward(self, q: torch.Tensor, p: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Linear Energy Contributions
        H_linear = torch.einsum('bsd,d->b', q, self.coef_linear_q).sum() + torch.einsum('bsd,d->b', p, self.coef_linear_p).sum()

        # 2. Quadratic Energy Contributions
        H_quad_pp = torch.einsum('bid,dk,bjd->b', p, self.coef_quadratic_pp, p).sum()
        H_quad_qp = torch.einsum('bid,dk,bjd->b', q, self.coef_quadratic_qp, p).sum()
        H_quad_qq = torch.einsum('bid,dk,bjd->b', q, self.coef_quadratic_qq, q).sum()

        # 3. Learned Non-linear Energy Contribution from Subspace Interactions
        H_neural = self.interaction_module(q, p)

        # 4. Total Hamiltonian
        H_total =( self.h_offset + H_linear +
        H_quad_qq + H_quad_pp + H_quad_qp +
        H_neural)
        
        return H_total

class HamiltonianModel(nn.Module):
    """
    The main model that orchestrates the system's evolution based on the
    learned Hamiltonian.
    """
    def __init__(self, num_blocks, input_dim, d_embedding, d_hidden_dim, output_dim, sequence_length, timestep, dropout):
        super().__init__()
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.clip_value = 1.0
        self.eps = 1e-4
        self.input_projection = nn.Linear(input_dim, d_embedding)
        self.pos_encoder = PositionalEncoding(d_embedding)
        self.dropout = nn.Dropout(dropout)
        
        self.q_shift = nn.Parameter(torch.zeros(d_embedding))
        self.momentum_net = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(d_hidden_dim // 2, d_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(d_hidden_dim // 2, d_embedding)
        )
        
        self.blocks = nn.ModuleList([
            HamiltonianBlock(d_embedding, d_hidden_dim, sequence_length, timestep, dropout) 
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(d_embedding)
        self.lm_head =nn.Sequential( nn.Linear(d_embedding, output_dim),nn.Softplus())

    def update_vars(self, q: torch.Tensor, p: torch.Tensor, H_func) -> tuple[torch.Tensor, torch.Tensor]:
        H_initial = H_func(q, p)
        grad_H_q = torch.autograd.grad(H_initial.sum(), q, create_graph=True)[0]
        p_half = p - (self.timestep / 2.0) * grad_H_q
    
        # Full step for position q using the half-step momentum
        H_mid = H_func(q, p_half) # Note: Technically grad should be evaluated at q, p_half
        grad_H_p_mid = torch.autograd.grad(H_mid.sum(), p_half, create_graph=True)[0]
        q_new = q + self.timestep * grad_H_p_mid
    
        # Final half-step for momentum p using the new position q
        H_final = H_func(q_new, p_half)
        grad_H_q_final = torch.autograd.grad(H_final.sum(), q_new, create_graph=True)[0]
        p_new = p_half - (self.timestep / 2.0) * grad_H_q_final
        
        # Clamp for stability
        q_new = torch.clamp(q_new, -10, 10)
        p_new = torch.clamp(p_new, -10, 10)
        
        return q_new, p_new

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q_proj = self.pos_encoder(self.input_projection(x))
        q = self.dropout(q_proj + self.q_shift)

        #q = q_initial.clone().requires_grad_(True)
        p = self.momentum_net(q) #.requires_grad_(True)
        
        for blk in self.blocks:
            q, p = self.update_vars(q, p,blk)


        output = self.norm(q)
        logits = self.lm_head(output)
        return logits

    @torch.no_grad()
    def generate(self, input_sequence: torch.Tensor, n_to_pred: int):
        self.eval()
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
                full_prediction_sequence = self(input_for_pred)
            
            next_primary_ohlc = full_prediction_sequence[:, -1:, :]
            predictions.append(next_primary_ohlc.cpu().numpy())
            last_full_step = current_input_seq[:, -1:, :].clone()
            last_full_step[:, :, start_feature_idx:end_feature_idx] = next_primary_ohlc
            current_input_seq = torch.cat([current_input_seq, last_full_step], dim=1)

        return np.concatenate(predictions, axis=1).squeeze(0)