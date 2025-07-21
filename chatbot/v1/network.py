# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
import warnings

class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]

class SubspaceInteraction(nn.Module):
    def __init__(self, d_embedding: int, d_hidden_dim: int, num_subspaces: int = 4,
                 subspace_dim: int = 64, n_head: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.n_head = n_head
        
        if subspace_dim % n_head != 0:
            raise ValueError(f"subspace_dim ({subspace_dim}) must be divisible by n_head ({n_head})")
        self.head_dim = subspace_dim // n_head

        # Use bias=True for better expressiveness
        self.q_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=True) for _ in range(num_subspaces)
        ])
        self.p_projections = nn.ModuleList([
            nn.Linear(d_embedding, subspace_dim, bias=True) for _ in range(num_subspaces)
        ])
        
        # --- Manual Attention Layers ---
        # A single linear layer to project all inputs to Q, K, V
        self.qkv_layer = nn.Linear(subspace_dim, 3 * subspace_dim)
        # Final output projection
        self.output_layer = nn.Linear(subspace_dim, subspace_dim)
        self.attn_dropout = nn.Dropout(dropout)
        # --- End of Manual Attention Layers ---
        
        # Add pre-norm for better stability
        self.pre_norm = nn.LayerNorm(subspace_dim)
        self.post_norm = nn.LayerNorm(subspace_dim)
        
        # Improved MLP with residual connection
        interaction_output_dim = (num_subspaces * 2) * subspace_dim
        self.energy_mlp = nn.Sequential(
            nn.Linear(interaction_output_dim, d_hidden_dim),
            nn.LayerNorm(d_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_dim, d_hidden_dim // 2),
            nn.LayerNorm(d_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_dim // 2, 1)
        )
        
        # Add learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = q.shape
        
        # Apply projections
        q_subs = torch.stack([proj(q) for proj in self.q_projections], dim=2)
        p_subs = torch.stack([proj(p) for proj in self.p_projections], dim=2)
        
        # Combine subspaces
        all_subspaces = torch.cat([q_subs, p_subs], dim=2)
        subspace_seq = all_subspaces.view(batch_size * seq_len, self.num_subspaces * 2, self.subspace_dim)

        # Pre-norm before attention
        normed_input = self.pre_norm(subspace_seq)
        
        # --- MANUAL MULTI-HEAD ATTENTION LOGIC ---
        B, T, C = normed_input.shape # Here B=batch*seq_len, T=num_subspaces*2, C=subspace_dim

        # 1. Project to Q, K, V
        qkv = self.qkv_layer(normed_input) # (B, T, 3*C)
        
        # 2. Split into Q, K, V and reshape for multi-head
        qkv = qkv.reshape(B, T, self.n_head, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # (B, n_head, T, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1) # Each is (B, n_head, T, head_dim)
        
        # 3. Calculate attention scores (Q @ K.T)
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1) # (B, n_head, T, T)
        weights = self.attn_dropout(weights)
        
        # 4. Apply attention weights to V
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        attention = torch.matmul(weights, v)
        
        # 5. Reshape and project output
        attention = attention.permute(0, 2, 1, 3).reshape(B, T, C)
        attn_output = self.output_layer(attention)
        # --- END OF MANUAL ATTENTION LOGIC ---
        
        # Post-norm after residual
        processed_subspaces = self.post_norm(normed_input + attn_output)

        # Apply temperature scaling
        flat_features = processed_subspaces.flatten(start_dim=1) * self.temperature
        energy_contribution_flat = self.energy_mlp(flat_features)
        energy_contribution = energy_contribution_flat.view(batch_size, seq_len)
        
        return energy_contribution.sum()

class LearnableFrFT(nn.Module):
    """Improved Learnable Fractional Fourier Transform with constraints"""
    
    def __init__(self, init_alpha: float = 0.01):
        super().__init__()
        # Initialize with small random value
        self.alpha_raw = nn.Parameter(torch.randn(1) * init_alpha)
        
    @property
    def alpha(self) -> torch.Tensor:
        # Constrain alpha to [-π, π] for stability
        return torch.tanh(self.alpha_raw) * math.pi
    
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_a = torch.cos(self.alpha)
        sin_a = torch.sin(self.alpha)

        q_new = q * cos_a + p * sin_a
        p_new = -q * sin_a + p * cos_a
        return q_new, p_new

    def inverse(self, q_prime: torch.Tensor, p_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_a = torch.cos(self.alpha)
        sin_a = torch.sin(self.alpha)

        q_orig = q_prime * cos_a - p_prime * sin_a
        p_orig = q_prime * sin_a + p_prime * cos_a
        return q_orig, p_orig

class InvertibleTransform(nn.Module):
    """Improved coupling layer with better initialization"""
    
    def __init__(self, d_embedding: int, d_hidden_transform: int, activation: str = 'tanh'):
        super().__init__()
        
        # Choose activation function
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # First coupling layer
        self.s_net1 = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_transform),
            nn.LayerNorm(d_hidden_transform),
            act_fn,
            nn.Linear(d_hidden_transform, d_embedding)
        )
        self.t_net1 = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_transform),
            nn.LayerNorm(d_hidden_transform),
            act_fn,
            nn.Linear(d_hidden_transform, d_embedding)
        )
        
        # Second coupling layer
        self.s_net2 = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_transform),
            nn.LayerNorm(d_hidden_transform),
            act_fn,
            nn.Linear(d_hidden_transform, d_embedding)
        )
        self.t_net2 = nn.Sequential(
            nn.Linear(d_embedding, d_hidden_transform),
            nn.LayerNorm(d_hidden_transform),
            act_fn,
            nn.Linear(d_hidden_transform, d_embedding)
        )
        
        # Initialize final layers with small weights for stability
        self._init_coupling_weights()
    
    def _init_coupling_weights(self):
        """Initialize coupling layers with small weights"""
        for net in [self.s_net1, self.t_net1, self.s_net2, self.t_net2]:
            if isinstance(net[-1], nn.Linear):
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # First coupling layer
        log_s1 = self.s_net1(p)
        # Clamp log_s1 for numerical stability
        log_s1 = torch.clamp(log_s1, -10, 10)
        t1 = self.t_net1(p)
        q_intermediate = torch.exp(log_s1) * q + t1

        # Second coupling layer
        log_s2 = self.s_net2(q_intermediate)
        log_s2 = torch.clamp(log_s2, -10, 10)
        t2 = self.t_net2(q_intermediate)
        a_real = q_intermediate
        a_imag = torch.exp(log_s2) * p + t2
        
        return a_real, a_imag, log_s1, log_s2

    def inverse(self, a_real: torch.Tensor, a_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inverse of second coupling layer
        log_s2 = self.s_net2(a_real)
        log_s2 = torch.clamp(log_s2, -10, 10)
        t2 = self.t_net2(a_real)
        p_intermediate = (a_imag - t2) * torch.exp(-log_s2)
        q_intermediate = a_real

        # Inverse of first coupling layer
        log_s1 = self.s_net1(p_intermediate)
        log_s1 = torch.clamp(log_s1, -10, 10)
        t1 = self.t_net1(p_intermediate)
        q = (q_intermediate - t1) * torch.exp(-log_s1)
        p = p_intermediate
        
        return q, p

class HamiltonianBlock(nn.Module):
    """Improved Hamiltonian block with better numerical stability"""
    
    def __init__(self, d_embedding: int, d_hidden_dim: int, **kwargs):
        super().__init__()
        self.d_embedding = d_embedding
        
        # Layer normalization
        self.norm_q = nn.LayerNorm(d_embedding)
        self.norm_p = nn.LayerNorm(d_embedding)
        
        # Learnable parameters with better initialization
        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding) * 0.01)
        self.coef_linear_p = nn.Parameter(torch.randn(d_embedding) * 0.01)
        
        # Symmetric matrices for quadratic terms
        self.coef_quadratic_qq = nn.Parameter(torch.eye(d_embedding) * 0.01)
        self.coef_quadratic_pp = nn.Parameter(torch.eye(d_embedding) * 0.01)
        self.coef_quadratic_qp = nn.Parameter(torch.randn(d_embedding, d_embedding) * 0.01)
        
        # Interaction module
        self.interaction_module = SubspaceInteraction(d_embedding, d_hidden_dim, **kwargs)
        
        # Learnable offset
        self.h_offset = nn.Parameter(torch.zeros(1))
        
        # Add gradient clipping parameter
        self.grad_clip_value = kwargs.get('grad_clip_value', 1.0)

    def forward(self, q: torch.Tensor, p: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_norm = self.norm_q(q)
        p_norm = self.norm_p(p)
        
        # Apply mask if provided
        if mask is not None:
            q_norm = q_norm * mask.unsqueeze(-1)
            p_norm = p_norm * mask.unsqueeze(-1)
        
        # Linear terms
        H_linear_q = torch.einsum('bsd,d->bs', q_norm, self.coef_linear_q)
        H_linear_p = torch.einsum('bsd,d->bs', p_norm, self.coef_linear_p)
        H_linear = (H_linear_q + H_linear_p).sum()
        
        # Quadratic terms (more efficient computation)
        H_quad_qq = torch.einsum('bsd,de,bse->bs', q_norm, self.coef_quadratic_qq, q_norm).sum()
        H_quad_pp = torch.einsum('bsd,de,bse->bs', p_norm, self.coef_quadratic_pp, p_norm).sum()
        H_quad_qp = torch.einsum('bsd,de,bse->bs', q_norm, self.coef_quadratic_qp, p_norm).sum()
        
        # Neural interaction term
        H_neural = self.interaction_module(q_norm, p_norm)
        
        # Total Hamiltonian
        H_total = self.h_offset + H_linear + H_quad_qq + H_quad_pp + H_quad_qp + H_neural
        
        return H_total

    def compute_gradients_explicitly(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients with better numerical stability"""
        q_norm = self.norm_q(q)
        p_norm = self.norm_p(p)
        
        # Make tensors require gradients
        q_norm = q_norm.requires_grad_(True)
        p_norm = p_norm.requires_grad_(True)

        # Analytical gradients for quadratic terms
        grad_q_quad_qq = torch.einsum('de,bse->bsd', self.coef_quadratic_qq + self.coef_quadratic_qq.T, q_norm)
        grad_q_quad_qp = torch.einsum('de,bse->bsd', self.coef_quadratic_qp, p_norm)
        grad_p_quad_pp = torch.einsum('de,bse->bsd', self.coef_quadratic_pp + self.coef_quadratic_pp.T, p_norm)
        grad_p_quad_qp = torch.einsum('ed,bse->bsd', self.coef_quadratic_qp, q_norm)
        
        # Analytical gradients
        grad_q_analytic = self.coef_linear_q.unsqueeze(0).unsqueeze(0) + grad_q_quad_qq + grad_q_quad_qp
        grad_p_analytic = self.coef_linear_p.unsqueeze(0).unsqueeze(0) + grad_p_quad_pp + grad_p_quad_qp
        
        # Neural gradients
        H_neural = self.interaction_module(q_norm, p_norm)
        grad_outputs = torch.ones_like(H_neural)
        
        try:
            grad_q_neural, grad_p_neural = torch.autograd.grad(
                outputs=H_neural,
                inputs=[q_norm, p_norm],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
        except RuntimeError as e:
            warnings.warn(f"Gradient computation failed: {e}")
            grad_q_neural = grad_p_neural = None
        
        # Combine gradients
        total_grad_q_norm = grad_q_analytic
        if grad_q_neural is not None:
            total_grad_q_norm = total_grad_q_norm + grad_q_neural
        
        total_grad_p_norm = grad_p_analytic
        if grad_p_neural is not None:
            total_grad_p_norm = total_grad_p_norm + grad_p_neural
        
        # Apply gradient clipping
        total_grad_q_norm = torch.clamp(total_grad_q_norm, -self.grad_clip_value, self.grad_clip_value)
        total_grad_p_norm = torch.clamp(total_grad_p_norm, -self.grad_clip_value, self.grad_clip_value)
        
        # Compute gradients w.r.t. original inputs
        try:
            dH_dq, dH_dp = torch.autograd.grad(
                outputs=[q_norm, p_norm],
                inputs=[q, p],
                grad_outputs=[total_grad_q_norm, total_grad_p_norm],
                create_graph=True,
                retain_graph=True
            )
        except RuntimeError as e:
            warnings.warn(f"Jacobian computation failed: {e}")
            # Fallback to direct gradients
            dH_dq = total_grad_q_norm
            dH_dp = total_grad_p_norm
        
        return dH_dq, dH_dp

class HamiltonianModel(nn.Module):
    """Improved Hamiltonian Model with better architecture and training stability"""
    
    def __init__(self, num_blocks: int, input_dim: int, d_embedding: int, d_hidden_dim: int, 
                 output_dim: int, vocab_size: int, embed_dim: int, pad_idx: int, 
                 timestep: float = 0.1, dropout: float = 0.1,momentum_noise_sigma:float=0.1 ,**kwargs):
        super().__init__()
        
        # Model parameters
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.d_embedding = d_embedding
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.input_projection = nn.Linear(input_dim, d_embedding)
        self.pos_encoder = PositionalEncoding(d_embedding)
        
        # Learnable parameters
        self.q_shift = nn.Parameter(torch.zeros(d_embedding))
        self.dropout = nn.Dropout(dropout)
        self.momentum_noise_sigma=momentum_noise_sigma
        # Improved momentum network
        self.momentum_net = nn.Sequential(
            nn.Linear(2 * d_embedding, d_hidden_dim),
            nn.LayerNorm(d_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_dim, d_hidden_dim // 2),
            nn.LayerNorm(d_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_dim // 2, d_embedding)
        )
        
        # Transforms
        self.frft_transform = LearnableFrFT()
        self.coord_transform = InvertibleTransform(d_embedding, d_hidden_dim // 2)
        
        # Normalization for transforms
        self.q_norm_for_transform = nn.LayerNorm(d_embedding)
        self.p_norm_for_transform = nn.LayerNorm(d_embedding)
        
        # Hamiltonian blocks
        self.blocks = nn.ModuleList([
            HamiltonianBlock(d_embedding, d_hidden_dim, **kwargs) 
            for _ in range(num_blocks)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_embedding)
        self.lm_head = nn.Linear(d_embedding, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Training parameters
        self.energy_regularization = kwargs.get('energy_regularization', 0.01)
        self.reversibility_weight = kwargs.get('reversibility_weight', 0.1)
        
    def _init_weights(self, m):
        """Improved weight initialization"""
        if isinstance(m, nn.Linear):
            # Use Xavier initialization for better gradient flow
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def leapfrog_update(self, q: torch.Tensor, p: torch.Tensor, 
                       hamiltonian_block: HamiltonianBlock, 
                       timestep: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Improved leapfrog integration with better stability"""
        
        # Store initial values
        q_start, p_start = q.clone(), p.clone()
        
        # Forward integration
        q_fwd = q_start.requires_grad_(True)
        p_fwd = p_start.requires_grad_(True)
        
        # Compute gradients
        grad_H1_q, grad_H1_p = hamiltonian_block.compute_gradients_explicitly(q_fwd, p_fwd)
        
        # Leapfrog steps with adaptive clipping
        p_half = p_fwd - (timestep / 2.0) * grad_H1_q
        q_new = q_fwd + timestep * grad_H1_p
        
        # Second gradient computation
        grad_H2_q, _ = hamiltonian_block.compute_gradients_explicitly(q_new, p_half)
        p_new = p_half - (timestep / 2.0) * grad_H2_q
        
        # Apply stability constraints
        q_final = torch.clamp(q_new, -10, 10)
        p_final = torch.clamp(p_new, -10, 10)
        
        # Compute reversibility loss if in training mode
        reversibility_loss = torch.tensor(0.0, device=q.device)
        
        if self.training:
            # Reverse integration for reversibility check
            q_rev = q_final.requires_grad_(True)
            p_rev = p_final.requires_grad_(True)
            
            rev_timestep = -timestep
            
            # Reverse leapfrog
            rev_grad_H1_q, rev_grad_H1_p = hamiltonian_block.compute_gradients_explicitly(q_rev, p_rev)
            
            rev_p_half = p_rev - (rev_timestep / 2.0) * rev_grad_H1_q
            rev_q_new = q_rev + rev_timestep * rev_grad_H1_p
            
            rev_grad_H2_q, _ = hamiltonian_block.compute_gradients_explicitly(rev_q_new, rev_p_half)
            rev_p_new = rev_p_half - (rev_timestep / 2.0) * rev_grad_H2_q
            
            # Compute reversibility loss
            q_error = F.mse_loss(rev_q_new, q_start)
            p_error = F.mse_loss(rev_p_new, p_start)
            reversibility_loss = q_error + p_error
        
        return q_final.detach(), p_final.detach(), reversibility_loss

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_internals: bool = False):
        # --- Initial setup is the same ---
        embedded_x = self.token_embedding(x)
        q_initial_proj = self.pos_encoder(self.input_projection(embedded_x))
        q_initial = self.dropout(q_initial_proj + self.q_shift)
        dq = torch.zeros_like(q_initial)
        dq[:, 1:, :] = q_initial[:, 1:, :] - q_initial[:, :-1, :]
        momentum_net_input = torch.cat([q_initial, dq], dim=-1)
        p_initial = self.momentum_net(momentum_net_input)
        
        q_rot, p_rot = self.frft_transform(q_initial, p_initial)
        
        q_norm = self.q_norm_for_transform(q_rot)
        p_norm = self.p_norm_for_transform(p_rot)
        Q_initial, P_initial, log_s1, log_s2 = self.coord_transform(q_norm, p_norm)
        
        Q, P = Q_initial, P_initial
        total_reversibility_loss = 0.0
        
        # --- START OF REFACTORING ---
        # Create lists to store energies for the conservation loss
        energies_initial = []
        energies_final = []
        
        for block in self.blocks:
            # 1. Calculate energy BEFORE the leapfrog step
            # This requires calling the block's forward pass.
            if return_internals: # Only compute if needed for the loss
                H_i = block(Q, P)
                energies_initial.append(H_i)

            # 2. Perform the leapfrog update as before
            Q_new, P_new, local_rev_loss = self.leapfrog_update(Q, P, block, timestep=self.timestep)
            total_reversibility_loss += local_rev_loss

            # 3. Calculate energy AFTER the leapfrog step
            if return_internals:
                H_f = block(Q_new, P_new)
                energies_final.append(H_f)
            
            # 4. Update state for the next block
            Q, P = Q_new, P_new
        
        total_reversibility_loss = total_reversibility_loss / self.num_blocks if self.num_blocks > 0 else 0.0

        Q_final, P_final = Q, P
        
        # --- Final part is the same ---
        q_transformed_final, p_transformed_final = self.coord_transform.inverse(Q_final, P_final)
        q_final, p_final_from_inverse = self.frft_transform.inverse(q_transformed_final, p_transformed_final)
        hidden_state = self.norm(q_final + q_initial)
        logits = self.lm_head(hidden_state)
        
        if return_internals:
            hamiltonian_internals = (Q_initial, P_initial, Q_final, P_final)
            jacobian_internals = (log_s1, log_s2)
            consistency_internals = (q_final, p_final_from_inverse)
            
            # --- Pass out the collected energies ---
            energy_internals = (energies_initial, energies_final)
            
            return (
                logits, 
                hidden_state, 
                hamiltonian_internals, 
                jacobian_internals, 
                consistency_internals, 
                total_reversibility_loss,
                energy_internals  # The new return value
            )
        else:
            return logits

    def generate(self, tokenizer, start_text: str, max_new_tokens: int, 
                 device: torch.device, temperature: float = 0.8, top_k: int = 40) -> str:
        """Improved generation with correctly scoped gradient context."""
        self.eval()
        
        try:
            input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
        except Exception as e:
            raise ValueError(f"Failed to encode start text: {e}")
        
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            
            with torch.no_grad():
                # Get next token logits
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-7)
                
                # Apply top-k filtering
                if top_k > 0:
                    values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < values[:, [-1]]] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Check for end token (outside the no_grad block is fine)
            if hasattr(tokenizer, 'eos_token_id') and next_token_id.item() == tokenizer.eos_token_id:
                break
        
        # Decoding does not require grads
        with torch.no_grad():
            try:
                generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            except Exception as e:
                raise ValueError(f"Failed to decode generated tokens: {e}")
        
        return generated_text