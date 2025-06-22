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
        """x: (batch_size, seq_len, d_embedding)"""
        return x + self.pe[:x.size(1)]

class MomentumNet(nn.Module):
    """
    A neural network to learn momentum 'p' as a function of 
    position 'q' and velocity 'q_dot'.
    p = f(q, q_dot)
    """
    def __init__(self, d_embedding, d_hidden):
        super().__init__()
        # Input will be q and q_dot concatenated, so dimension is d_embedding * 2
        self.net = nn.Sequential(
            nn.Linear(d_embedding * 2, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_embedding) # Output has same dimension as q and p
        )

    def forward(self, q, q_dot):
        # Concatenate along the feature dimension
        x = torch.cat([q, q_dot], dim=-1)
        return self.net(x)

class HamiltonianBlock(nn.Module):
    def __init__(self, d_embedding, d_hidden_dim,sequence_length,timestep):
        super().__init__()
        self.d_embedding=d_embedding
        self.d_hidden_potential=d_hidden_dim
        self.output_dim_internal=1
        self.sequence_length=sequence_length
        self.timestep=timestep

        self.masses = nn.Parameter(torch.ones(sequence_length))

        self.mlp = nn.Sequential(
            nn.Linear(d_embedding * 2, d_hidden_dim), # Input is concat(q_i, p_i)
            nn.Tanh(),
            nn.Linear(d_hidden_dim, self.output_dim_internal) 
        )

        self.h_offset = nn.Parameter(torch.randn(1))

        self.coef_linear_q = nn.Parameter(torch.randn(d_embedding))
        self.coef_linear_p = nn.Parameter(torch.randn(d_embedding))

        self.coef_quadratic_qp = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_qq = nn.Parameter(torch.randn(d_embedding, d_embedding))
        self.coef_quadratic_pp = nn.Parameter(torch.randn(d_embedding, d_embedding))



    def forward(self,q: torch.Tensor,p: torch.Tensor ,mask: torch.Tensor | None = None):

        batch_size, seq_len, d_embed = q.shape
        
        linear_q_contrib = torch.einsum('bsd,d->bs', q, self.coef_linear_q).sum(dim=1) 
        linear_p_contrib = torch.einsum('bsd,d->bs', p, self.coef_linear_p).sum(dim=1)

        quadratic_qp = torch.einsum('bid,dk,bjd->bij', q, self.coef_quadratic_qp, p)
        quadratic_qp_contrib = quadratic_qp.sum(dim=(1,2))

        quadratic_qq = torch.einsum('bid,dk,bjd->bij', q, self.coef_quadratic_qq, q)
        quadratic_qq_contrib = quadratic_qq.sum(dim=(1,2))

        quadratic_pp = torch.einsum('bid,dk,bjd->bij', p, self.coef_quadratic_pp, p)
        quadratic_pp_contrib = quadratic_pp.sum(dim=(1,2))

        qp_concat = torch.cat([q, p], dim=-1)
        mlp_output_resh = self.mlp(qp_concat.view(-1, self.d_embedding * 2)) 
        mlp_term_per_token = mlp_output_resh.view(batch_size, seq_len) 
        mlp_contrib = mlp_term_per_token.sum(dim=1)

        H_batch : torch.Tensor= ( self.h_offset +
                    linear_q_contrib + linear_p_contrib +
                    quadratic_qp_contrib + quadratic_qq_contrib + quadratic_pp_contrib +
                    mlp_contrib )
        
        return q,p,H_batch.sum()



class HamiltonianModel(nn.Module):
    def __init__(self, num_blocks,input_dim,d_embedding, d_hidden_dim,output_dim,sequence_length,timestep):
        super().__init__()
        self.num_blocks=num_blocks
        self.input_projection = nn.Linear(input_dim, d_embedding)
        self.pos_encoder = PositionalEncoding(d_embedding)
        self.blocks = nn.ModuleList([
            HamiltonianBlock(d_embedding, d_hidden_dim,sequence_length,timestep) for _ in range(num_blocks)
        ])
        self.momentum_net = MomentumNet(d_embedding, d_hidden_dim // 2 )
        self.norm = nn.LayerNorm(d_embedding)
        self.lm_head = nn.Linear(d_embedding, output_dim)
    def update_vars(self,q: torch.Tensor,p: torch.Tensor,H_scalar,padding_mask: torch.Tensor | None = None):

        grad_H_wrt_q_dec = torch.autograd.grad(H_scalar, q, grad_outputs=torch.ones_like(H_scalar), create_graph=True, retain_graph=True, allow_unused=True)[0]
        grad_H_wrt_p_dec = torch.autograd.grad(H_scalar, p, grad_outputs=torch.ones_like(H_scalar), create_graph=True, retain_graph=True, allow_unused=True)[0]

        dq_dt = grad_H_wrt_p_dec if grad_H_wrt_p_dec is not None else torch.zeros_like(p)
        dp_dt = -grad_H_wrt_q_dec if grad_H_wrt_q_dec is not None else torch.zeros_like(q)
        if padding_mask is not None:
            dq_dt = dq_dt * padding_mask.unsqueeze(-1).float()
            dp_dt = dp_dt * padding_mask.unsqueeze(-1).float()
        return q+dq_dt, p+dp_dt
    def forward(self,x: torch.Tensor, mask: torch.Tensor | None = None):
        q_initial: torch.Tensor = self.pos_encoder(self.input_projection(x))
        
        # We'll use this q for the evolution
        q = q_initial.clone().requires_grad_(True)
        q_dot = q[:, 1:, :] - q[:, :-1, :]

        zero_velocity = torch.zeros_like(q[:, :1, :])
        q_dot = torch.cat([zero_velocity, q_dot], dim=1)

        # 3. Calculate initial momentum `p` using the new network
        p: torch.Tensor = self.momentum_net(q, q_dot).requires_grad_(True)
        if mask is not None:
            p = p * mask.unsqueeze(-1).float()

        for blk in self.blocks:
            _,_,H_scalar=blk(q,p,mask)
            q_new, p_new = self.update_vars(q, p, H_scalar)
            
            q = q_new.detach().requires_grad_(True)
            p = p_new.detach().requires_grad_(True)

        output = self.norm(q)
        logits = self.lm_head(output)
        return logits
    
    @torch.no_grad()
    def generate(self, input_sequence: torch.Tensor, n_to_pred: int):
        self.eval()
        
        # --- NEW: Get symbol information from config ---
        from config import config # Local import
        symbols = config["symbols"]
        primary_symbol = config["primary_symbol"]
        
        # Find the index of the primary symbol. This is crucial.
        # Each symbol has 4 features (o, c, h, l).
        try:
            primary_symbol_idx = symbols.index(primary_symbol)
        except ValueError:
            raise ValueError(f"primary_symbol '{primary_symbol}' not found in symbols list in config.")
        
        # Calculate the feature slice for the primary symbol
        start_feature_idx = primary_symbol_idx * 4
        end_feature_idx = start_feature_idx + 4
        
        # --- End of new setup ---
        
        predictions = []
        # Ensure input is on the correct device
        current_input_seq = input_sequence.to(next(self.parameters()).device)

        for _ in range(n_to_pred):
            # The model expects a sequence of length `config["sequence_length"]`
            # So we only feed the last `sequence_length` steps
            input_for_pred = current_input_seq[:, -config["sequence_length"]:, :]

            with torch.enable_grad():
                # Get model prediction for the entire sequence
                full_prediction_sequence = self(input_for_pred)

            # We only care about the prediction for the very last time step
            # This is the predicted OHLC for the primary symbol
            next_primary_ohlc = full_prediction_sequence[:, -1:, :] # Shape: (1, 1, output_dim=4)

            # Append the prediction to our list of results
            predictions.append(next_primary_ohlc.cpu().numpy())

            # --- NEW: Construct the next full input step ---
            # Take the last known full feature vector from the input
            last_full_step = current_input_seq[:, -1:, :].clone() # Shape: (1, 1, input_dim=20)
            
            # Replace the old primary symbol's data with the new prediction
            last_full_step[:, :, start_feature_idx:end_feature_idx] = next_primary_ohlc

            # Now, `last_full_step` is the complete feature vector for the next time step.
            # It has the new prediction for GLD and carries over the last values for the others.
            
            # Append this complete new step to the input sequence for the next iteration
            current_input_seq = torch.cat([current_input_seq, last_full_step], dim=1)

        # Concatenate list of predictions into a single numpy array
        return np.concatenate(predictions, axis=1).squeeze(0)