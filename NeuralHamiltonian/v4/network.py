import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

class SubspaceInteraction(nn.Module):
    def __init__(self, d_embedding, d_hidden_dim, num_subspaces=4, subspace_dim=64):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.q_projections = nn.ModuleList([nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)])
        self.p_projections = nn.ModuleList([nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)])
        num_original_features = num_subspaces * 2
        num_product_features = 3 * (num_subspaces ** 2)
        interaction_input_dim = (num_original_features + num_product_features) * subspace_dim
        self.interaction_net = nn.Sequential(nn.Linear(interaction_input_dim, d_hidden_dim), nn.GELU(), nn.Linear(d_hidden_dim, d_hidden_dim), nn.GELU(), nn.Linear(d_hidden_dim, 1))

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q_subs = [proj(q) for proj in self.q_projections]
        p_subs = [proj(p) for proj in self.p_projections]
        Q_stacked = torch.stack(q_subs, dim=2); P_stacked = torch.stack(p_subs, dim=2)
        qq_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, Q_stacked)
        pp_products = torch.einsum('bsid, bsjd -> bsijd', P_stacked, P_stacked)
        qp_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, P_stacked)
        flat_q_subs = Q_stacked.flatten(start_dim=2); flat_p_subs = P_stacked.flatten(start_dim=2)
        flat_qq = qq_products.flatten(start_dim=2); flat_pp = pp_products.flatten(start_dim=2); flat_qp = qp_products.flatten(start_dim=2)
        concatenated_features = torch.cat([flat_q_subs, flat_p_subs, flat_qq, flat_pp, flat_qp], dim=-1)
        energy_contribution = self.interaction_net(concatenated_features)
        return energy_contribution.sum()

class InvertibleTransform(nn.Module):
    # ... (no changes) ...
    def __init__(self, d_embedding, d_hidden_transform):
        super().__init__()
        self.s_net1 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))
        self.t_net1 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))
        self.s_net2 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))
        self.t_net2 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))

    def forward(self, q, p):
        log_s1 = self.s_net1(p); t1 = self.t_net1(p)
        q_intermediate = torch.exp(log_s1) * q + t1; p_intermediate = p
        log_s2 = self.s_net2(q_intermediate); t2 = self.t_net2(q_intermediate)
        a_real = q_intermediate; a_imag = torch.exp(log_s2) * p_intermediate + t2
        return a_real, a_imag

    def inverse(self, a_real, a_imag):
        log_s2 = self.s_net2(a_real); t2 = self.t_net2(a_real)
        p_intermediate = (a_imag - t2) * torch.exp(-log_s2); q_intermediate = a_real
        log_s1 = self.s_net1(p_intermediate); t1 = self.t_net1(p_intermediate)
        q = (q_intermediate - t1) * torch.exp(-log_s1); p = p_intermediate
        return q, p

class HamiltonianBlock(nn.Module):
    def __init__(self, d_embedding, d_hidden_dim, sequence_length, timestep, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_embedding)
        self.norm_p = nn.LayerNorm(d_embedding)

        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding))
        self.coef_linear_p = nn.Parameter(torch.randn(d_embedding))
        self.coef_quadratic_qq = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_pp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_qp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.interaction_module = SubspaceInteraction(d_embedding, d_hidden_dim) 
        self.h_offset = nn.Parameter(torch.randn(1))

    def forward(self, qi: torch.Tensor, pi: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q=self.norm_q(qi)
        p=self.norm_p(pi)
        H_linear = torch.einsum('bsd,d->b', q, self.coef_linear_q).sum() + torch.einsum('bsd,d->b', p, self.coef_linear_p).sum()
        H_quad_pp = torch.einsum('bid,dk,bjd->b', p, self.coef_quadratic_pp, p).sum()
        H_quad_qp = torch.einsum('bid,dk,bjd->b', q, self.coef_quadratic_qp, p).sum()
        H_quad_qq = torch.einsum('bid,dk,bjd->b', q, self.coef_quadratic_qq, q).sum()
        H_neural = self.interaction_module(q, p)
        H_total = self.h_offset + H_linear + H_quad_qq + H_quad_pp + H_quad_qp + H_neural
        return H_total

class HamiltonianModel(nn.Module):
    def __init__(self, num_blocks, input_dim, d_embedding, d_hidden_dim, output_dim, sequence_length, timestep, dropout):
        super().__init__()
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.input_projection = nn.Linear(input_dim, d_embedding)
        self.pos_encoder = PositionalEncoding(d_embedding)
        self.q_shift = nn.Parameter(torch.zeros(d_embedding))
        self.dropout = nn.Dropout(dropout)
        self.momentum_net = nn.Sequential(nn.Linear(d_embedding, d_hidden_dim // 2), nn.Tanh(), nn.Linear(d_hidden_dim // 2, d_hidden_dim // 2), nn.Tanh(), nn.Linear(d_hidden_dim // 2, d_embedding))
        
        self.coord_transform = InvertibleTransform(d_embedding, d_hidden_dim // 2)
        self.blocks = nn.ModuleList([
            HamiltonianBlock(d_embedding, d_hidden_dim, sequence_length, timestep, dropout) 
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(d_embedding)
        self.lm_head = nn.Sequential(nn.Linear(d_embedding, output_dim), nn.Softplus())

    def leapfrog_update(self, q, p, hamiltonian_block):
        q = q.requires_grad_(True); p = p.requires_grad_(True)
        H_1 = hamiltonian_block(q, p)
        grad_H1_q, grad_H1_p = torch.autograd.grad(H_1.sum(), [q, p], create_graph=True)
        p_half = p - (self.timestep / 2.0) * grad_H1_q
        q_new = q + self.timestep * grad_H1_p
        q_new_detached = q_new.detach().requires_grad_(True)
        H_2 = hamiltonian_block(q_new_detached, p_half)
        grad_H2_q = torch.autograd.grad(H_2.sum(), q_new_detached, create_graph=True)[0]
        p_new = p_half - (self.timestep / 2.0) * grad_H2_q
        q_new = torch.clamp(q_new.detach(), -10, 10); p_new = torch.clamp(p_new.detach(), -10, 10)
        return q_new, p_new

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_internals: bool = False):
        q_initial_proj = self.pos_encoder(self.input_projection(x))
        q_initial = self.dropout(q_initial_proj + self.q_shift)
        p_initial = self.momentum_net(q_initial)

        Q_initial, P_initial = self.coord_transform(q_initial, p_initial)

        Q, P = Q_initial, P_initial # Start the evolution
        for blk in self.blocks:
            Q, P = self.leapfrog_update(Q, P, blk)

        Q_final, P_final = Q, P # End of evolution

        q_final, _ = self.coord_transform.inverse(Q_final, P_final)

        output = self.norm(q_final)
        logits = self.lm_head(output)

        if return_internals:
            return logits, (Q_initial, P_initial, Q_final, P_final)
        else:
            return logits

    @torch.no_grad()
    def generate(self, input_sequence: torch.Tensor, n_to_pred: int):
        # ... (no changes to this method) ...
        self.eval()
        from config import config
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