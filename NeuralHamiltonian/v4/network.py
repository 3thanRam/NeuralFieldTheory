import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding)); pe = torch.zeros(max_len, d_embedding)
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1)]

# In network.py, replace the old SubspaceInteraction class with this one.

class SubspaceInteraction(nn.Module):
    """
    Upgraded version that uses a multi-stage pipeline to efficiently
    capture higher-order (third-order and beyond) interactions.
    """
    def __init__(self, d_embedding, d_hidden_dim, num_subspaces=4, subspace_dim=64):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim

        # --- Stage 1: Linear Projections ---
        # The q and p projections remain the same.
        self.q_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)
        ])
        self.p_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=False) for _ in range(num_subspaces)
        ])

        # --- Stage 3: The new "Second-Order Processor" MLP ---
        # This network learns to mix the 2nd-order features to create rich representations.
        # Its input is the flattened set of all 2nd-order products.
        num_product_feature_groups = 3 * (num_subspaces ** 2)
        second_order_dim = num_product_feature_groups * subspace_dim
        
        # The output dimension of this processor is a new hyperparameter.
        # d_hidden_dim is a good choice as it matches the model's general capacity.
        self.second_order_processor = nn.Sequential(
            nn.Linear(second_order_dim, d_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(d_hidden_dim) # Add LayerNorm for stability
        )

        # --- Stage 4: The Final Interaction Network ---
        # Its input is now the original subspace features plus the processed 2nd-order features.
        num_original_features = num_subspaces * 2
        original_feature_dim = num_original_features * subspace_dim
        
        # The input dimension is much more controlled now.
        interaction_input_dim = original_feature_dim + d_hidden_dim # Output size of the processor
        
        self.interaction_net = nn.Sequential(
            nn.Linear(interaction_input_dim, d_hidden_dim),
            nn.GELU(),
            nn.Linear(d_hidden_dim, d_hidden_dim),
            nn.GELU(),
            nn.Linear(d_hidden_dim, 1)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # --- Stage 1 & 2: Create Linear and Second-Order Features ---
        q_subs = [proj(q) for proj in self.q_projections]
        p_subs = [proj(p) for proj in self.p_projections]
        Q_stacked = torch.stack(q_subs, dim=2)
        P_stacked = torch.stack(p_subs, dim=2)
        
        # Second-order products
        qq_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, Q_stacked).flatten(start_dim=2)
        pp_products = torch.einsum('bsid, bsjd -> bsijd', P_stacked, P_stacked).flatten(start_dim=2)
        qp_products = torch.einsum('bsid, bsjd -> bsijd', Q_stacked, P_stacked).flatten(start_dim=2)
        
        # Concatenate all second-order features for processing
        all_second_order_features = torch.cat([qq_products, pp_products, qp_products], dim=-1)

        # --- Stage 3: Process Second-Order Features to get Rich Representations ---
        # This is where 3rd-order (and higher) information is implicitly created.
        rich_interaction_features = self.second_order_processor(all_second_order_features)
        
        # --- Stage 4: Final Combination ---
        # Flatten the original linear subspace features
        flat_q_subs = Q_stacked.flatten(start_dim=2)
        flat_p_subs = P_stacked.flatten(start_dim=2)
        
        # Combine original features with the new rich features
        all_features_for_final_net = torch.cat([flat_q_subs, flat_p_subs, rich_interaction_features], dim=-1)
        
        # Pass through the final MLP to get the energy contribution
        energy_contribution = self.interaction_net(all_features_for_final_net)
        
        return energy_contribution.sum()

class InvertibleTransform(nn.Module):
    def __init__(self, d_embedding, d_hidden_transform):
        super().__init__(); self.s_net1 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding)); self.t_net1 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))
        self.s_net2 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding)); self.t_net2 = nn.Sequential(nn.Linear(d_embedding, d_hidden_transform), nn.Tanh(), nn.Linear(d_hidden_transform, d_embedding))
    def forward(self, q, p):
        log_s1 = self.s_net1(p); t1 = self.t_net1(p); q_intermediate = torch.exp(log_s1) * q + t1; p_intermediate = p
        log_s2 = self.s_net2(q_intermediate); t2 = self.t_net2(q_intermediate); a_real = q_intermediate; a_imag = torch.exp(log_s2) * p_intermediate + t2
        return a_real, a_imag
    def inverse(self, a_real, a_imag):
        log_s2 = self.s_net2(a_real); t2 = self.t_net2(a_real); p_intermediate = (a_imag - t2) * torch.exp(-log_s2); q_intermediate = a_real
        log_s1 = self.s_net1(p_intermediate); t1 = self.t_net1(p_intermediate); q = (q_intermediate - t1) * torch.exp(-log_s1); p = p_intermediate
        return q, p

class HamiltonianBlock(nn.Module):
    def __init__(self, d_embedding, d_hidden_dim, **kwargs):
        super().__init__(); self.norm_q = nn.LayerNorm(d_embedding); self.norm_p = nn.LayerNorm(d_embedding)
        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding)); self.coef_linear_p = nn.Parameter(torch.randn(d_embedding))
        self.coef_quadratic_qq = nn.Parameter(torch.randn(d_embedding, d_embedding)); self.coef_quadratic_pp = nn.Parameter(torch.randn(d_embedding, d_embedding)); self.coef_quadratic_qp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.interaction_module = SubspaceInteraction(d_embedding, d_hidden_dim); self.h_offset = nn.Parameter(torch.randn(1))
    def forward(self, q, p, mask=None):
        q_norm = self.norm_q(q); p_norm = self.norm_p(p)
        
        H_linear = torch.einsum('bsd,d->b', q_norm, self.coef_linear_q).sum() + torch.einsum('bsd,d->b', p_norm, self.coef_linear_p).sum()
        
        H_quad_pp = torch.einsum('bid,dk,bjd->b', p_norm, self.coef_quadratic_pp, p_norm).sum(); H_quad_qp = torch.einsum('bid,dk,bjd->b', q_norm, self.coef_quadratic_qp, p_norm).sum(); H_quad_qq = torch.einsum('bid,dk,bjd->b', q_norm, self.coef_quadratic_qq, q_norm).sum()
        
        H_neural = self.interaction_module(q_norm, p_norm)
        
        H_total = self.h_offset + H_linear + H_quad_qq + H_quad_pp + H_quad_qp + H_neural
        return H_total

class HamiltonianModel(nn.Module):
    def __init__(self, num_blocks, input_dim, d_embedding, d_hidden_dim, output_dim, **kwargs):
        super().__init__(); self.num_blocks = num_blocks; self.timestep = kwargs.get('timestep', 0.1)
        self.input_projection = nn.Linear(input_dim, d_embedding); self.pos_encoder = PositionalEncoding(d_embedding)
        self.q_shift = nn.Parameter(torch.zeros(d_embedding)); self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.momentum_net = nn.Sequential(
            nn.Linear(2 * d_embedding, d_hidden_dim),
            nn.Tanh(),
            nn.Linear(d_hidden_dim, d_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(d_hidden_dim // 2, d_embedding)
        )
        self.coord_transform = InvertibleTransform(d_embedding, d_hidden_dim // 2)
        self.q_norm_for_transform = nn.LayerNorm(d_embedding)
        self.p_norm_for_transform = nn.LayerNorm(d_embedding)
        self.blocks = nn.ModuleList([HamiltonianBlock(d_embedding, d_hidden_dim, **kwargs) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(d_embedding); self.lm_head = nn.Sequential(nn.Linear(d_embedding, output_dim), nn.Softplus())
        self.apply(self._init_weights)
    def _init_weights(self, m):
        """
        Applies Kaiming He initialization to linear layers.
        This method is designed to be used with `self.apply()`.
        """
        if isinstance(m, nn.Linear):
            # Kaiming He initialization for layers followed by a ReLU/GELU/Tanh
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                                  
    def leapfrog_update(self, q, p, hamiltonian_block):
        q = q.requires_grad_(True); p = p.requires_grad_(True)

        H_1 = hamiltonian_block(q, p)
        grad_H1_q, grad_H1_p = torch.autograd.grad(H_1.sum(), [q, p], create_graph=True)

        grad_H1_q = torch.clamp(grad_H1_q, -1.0, 1.0)
        grad_H1_p = torch.clamp(grad_H1_p, -1.0, 1.0)
        

        p_half = p - (self.timestep / 2.0) * grad_H1_q
        q_new = q + self.timestep * grad_H1_p

        q_new_detached = q_new.requires_grad_(True)
        H_2 = hamiltonian_block(q_new_detached, p_half)
        grad_H2_q = torch.autograd.grad(H_2.sum(), q_new_detached, create_graph=True)[0]

        grad_H2_q = torch.clamp(grad_H2_q, -1.0, 1.0)

        p_new = p_half - (self.timestep / 2.0) * grad_H2_q

        q_new = torch.clamp(q_new.detach(), -10, 10)
        p_new = torch.clamp(p_new.detach(), -10, 10)

        return q_new, p_new

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_internals: bool = False):
        q_initial_proj = self.pos_encoder(self.input_projection(x)); q_initial = self.dropout(q_initial_proj + self.q_shift)
        dq = torch.zeros_like(q_initial)
        dq[:, 1:, :] = q_initial[:, 1:, :] - q_initial[:, :-1, :]
        momentum_net_input = torch.cat([q_initial, dq], dim=-1)
        p_initial = self.momentum_net(momentum_net_input)

        q_norm = self.q_norm_for_transform(q_initial)
        p_norm = self.p_norm_for_transform(p_initial)
        Q_initial, P_initial = self.coord_transform(q_norm, p_norm)
        Q, P = Q_initial, P_initial
        
        for block in self.blocks:
            Q, P = self.leapfrog_update(Q, P, block)
            
        Q_final, P_final = Q, P
        q_final, _ = self.coord_transform.inverse(Q_final, P_final)
        output = self.norm(q_final + q_initial)
        logits = self.lm_head(output)
        if return_internals: return logits, (Q_initial, P_initial, Q_final, P_final)
        else: return logits

    @torch.no_grad()
    def generate(self, input_sequence, n_to_pred):
        self.eval(); from config import config; symbols = config["symbols"]; primary_symbol = config["primary_symbol"]
        try: primary_symbol_idx = symbols.index(primary_symbol)
        except ValueError: raise ValueError(f"primary_symbol '{primary_symbol}' not found in symbols list in config.")
        num_features_per_symbol = config["input_dim"] // len(symbols)
        start_feature_idx = primary_symbol_idx * num_features_per_symbol; end_feature_idx = start_feature_idx + 4
        predictions = []; current_input_seq = input_sequence.to(next(self.parameters()).device)
        for _ in range(n_to_pred):
            input_for_pred = current_input_seq[:, -config["sequence_length"]:, :]
            with torch.enable_grad(): full_prediction_sequence = self(input_for_pred)
            next_primary_ohlc = full_prediction_sequence[:, -1:, :]
            predictions.append(next_primary_ohlc.cpu().numpy())
            last_full_step = current_input_seq[:, -1:, :].clone()
            last_full_step[:, :, start_feature_idx:end_feature_idx] = next_primary_ohlc
            current_input_seq = torch.cat([current_input_seq, last_full_step], dim=1)
        return np.concatenate(predictions, axis=1).squeeze(0)