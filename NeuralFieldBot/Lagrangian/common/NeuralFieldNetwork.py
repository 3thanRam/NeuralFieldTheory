# common/NeuralFieldNetwork.py - A highly parallelizable LNN architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Classes ---

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
    """
    Performs instance-wise normalization ACROSS THE TIME DIMENSION for each feature.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """ Normalizes each feature channel across the sequence length. """
        # x shape: [batch, seq_len, features]
        
        # --- THE FIX: Normalize along the sequence dimension (dim=1) ---
        # We calculate the mean and std for each feature channel independently.
        mean = x.mean(dim=1, keepdim=True) # Shape: [batch, 1, features]
        std = x.std(dim=1, keepdim=True)   # Shape: [batch, 1, features]
        
        normalized_x = (x - mean) / (std + self.eps)
        
        # Return the stats so the inverse can use them
        return normalized_x, mean, std

    def inverse(self, x_norm, mean, std):
        """ Un-normalizes using the explicitly passed-in stats. """
        # The mean and std have shape [batch, 1, features] and will broadcast correctly.
        return x_norm * (std + self.eps) + mean
class LearnableFrFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, q, p):
        c, s = torch.cos(self.alpha), torch.sin(self.alpha)
        return q * c + p * s, -q * s + p * c

    def inverse(self, q, p):
        c, s = torch.cos(self.alpha), torch.sin(self.alpha)
        return q * c - p * s, q * s + p * c

# --- NEW: Parallel "Force Field" Block ---
# This replaces the old PotentialBlock and get_acceleration logic.
class ParallelForceBlock(nn.Module):
    """
    Learns a "force field" directly using convolutions. This is highly parallelizable
    across the sequence dimension.
    """
    def __init__(self, embed_dim, hidden_dim, kernel_size=3):
        super().__init__()
        # An MLP to create a "potential field" vector at each position
        self.potential_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim) # Output a vector of the same dimension
        )
        # A convolution to calculate the "force" from the potential of neighbors
        self.force_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=embed_dim # Depthwise convolution for efficiency
        )

    def forward(self, q):
        # q shape: [batch, seq_len, embed_dim]
        potential_field = self.potential_mlp(q)
        # Conv1d expects [batch, channels, length]
        force = self.force_conv(potential_field.permute(0, 2, 1))
        # Return force in the original shape
        return force.permute(0, 2, 1)

# --- The Main LNN Class, now with Parallel Blocks ---
class LNN(nn.Module):
    def __init__(self, mode: str, embed_dim: int, d_hidden_dim: int, num_blocks: int, **kwargs):
        super().__init__()
        self.mode = mode; self.dt = kwargs.get('dt', 0.1)
        
        # This LNN now assumes the input is ALREADY the correct embedding dimension.
        # It no longer has an input_head or output_head. It is a pure processor.
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.q_shift = nn.Parameter(torch.zeros(embed_dim))
        self.coord_transform = LearnableFrFT()
        self.F_blocks = nn.ModuleList([ParallelForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])
        self.G_blocks = nn.ModuleList([ParallelForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, initial_vectors, return_internals=False):
        # The input `initial_vectors` is now the starting point.
        q_0 = self.dropout(self.pos_encoder(initial_vectors) + self.q_shift)
        q_dot_0 = torch.zeros_like(q_0); q_dot_0[:, 1:] = (q_0[:, 1:] - q_0[:, :-1]) / self.dt
        Q, Q_dot = self.coord_transform(q_0, q_dot_0)
        
        force_f_history, force_g_history = [], []
        for f_block, g_block in zip(self.F_blocks, self.G_blocks):
            accel_f, accel_g = f_block(Q), g_block(Q)
            if return_internals: force_f_history.append(accel_f); force_g_history.append(accel_g)
            Q_dot = Q_dot + accel_f * self.dt
            Q = Q + accel_g * self.dt
            
        q_final, _ = self.coord_transform.inverse(Q, Q_dot)
        # The output of the core is the final hidden state.
        final_state = self.final_norm(q_final + q_0)
        
        if return_internals:
            internals = { 'final_q': Q, 'final_q_dot': Q_dot, 'forces_f': force_f_history, 'forces_g': force_g_history }
            return final_state, internals
        return final_state

# --- NEW: A Generic Model Wrapper ---
class UniversalModel(nn.Module):
    """ A single, universal model that handles different tasks and normalization. """
    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode
        
        if self.mode == 'timeseries':
            self.normalizer = DynamicNorm()
            self.input_head = nn.Linear(kwargs['num_input_features'], kwargs['embed_dim'])
            self.output_head = nn.Linear(kwargs['embed_dim'], kwargs['num_output_predictions'])
        elif self.mode == 'nlp':
            self.normalizer = None
            self.input_head = nn.Embedding(kwargs['vocab_size'], kwargs['embed_dim'], padding_idx=kwargs['pad_idx'])
            self.output_head = nn.Linear(kwargs['embed_dim'], kwargs['vocab_size'])
        
        self.core = LNN(mode=mode, **kwargs)

    def forward(self, x, return_internals=False):
        if self.mode == 'timeseries':
            # 1. Normalize the raw input
            x_norm, norm_mean, norm_std = self.normalizer.forward(x)
            # 2. Project to embedding space
            initial_vectors = self.input_head(x_norm)
        else: # NLP
            initial_vectors = self.input_head(x)

        # 3. Run core dynamics
        if return_internals:
            final_state, internals = self.core(initial_vectors, return_internals=True)
        else:
            final_state = self.core(initial_vectors, return_internals=False)

        # 4. Project to output space
        predictions = self.output_head(final_state)

        # 5. Un-normalize if necessary
        if self.mode == 'timeseries':
            predictions = self.normalizer.inverse(predictions, norm_mean, norm_std)
        
        if return_internals:
            return predictions, internals
        return predictions

    @torch.no_grad()
    def generate(self, start_input, max_new_tokens, **kwargs):
        # This method is now compatible with the new parallel forward pass
        # The logic within remains the same.
        self.eval()
        device = next(self.parameters()).device

        if self.mode == 'nlp':
            tokenizer = kwargs.get('tokenizer')
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for text generation.")
            ids = tokenizer.encode(start_input, return_tensors='pt').to(device) if isinstance(start_input, str) else start_input.to(device)

            for _ in range(max_new_tokens):
                logits = self.forward(ids)
                next_logits = logits[:, -1, :] / kwargs.get('temperature', 0.8)
                top_k = kwargs.get('top_k', 20)
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, -1]] = -float('Inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                ids = torch.cat((ids, next_id), dim=1)
                
                if hasattr(tokenizer, 'eos_token_id') and next_id.item() == tokenizer.eos_token_id:
                    break
            
            return tokenizer.decode(ids[0], skip_special_tokens=True)
        
        elif self.mode == 'timeseries':
            current_sequence = start_input.clone().to(device)
            predicted_steps = []
            for _ in range(max_new_tokens):
                next_step_prediction = self.forward(current_sequence)[:, -1:, :]
                predicted_steps.append(next_step_prediction.cpu())
                current_sequence = torch.cat([current_sequence[:, 1:, :], next_step_prediction], dim=1)
            return torch.cat(predicted_steps, dim=1).squeeze(0)