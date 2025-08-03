# LNFT.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


from LNFT_block import GaugeConvolutionBlock
from base_modules import PositionalEncoding

class LNFT_Noise_Predictor(nn.Module):
    """
    The LNFT architecture re-purposed as a noise prediction model,
    which is the core of a score-based/diffusion model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Timestep Embedding ---
        # The model needs to know the noise level (t) of its input.
        # This is encoded using a sinusoidal time embedding, just like position.
        time_embed_dim = config['embed_dim'] * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(config['embed_dim'], time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, config['embed_dim'])
        )
        # ------------------------

        self.input_head = nn.Embedding(
            config['vocab_size'], config['embed_dim'], padding_idx=config['pad_idx']
        )
        self.pos_encoder = PositionalEncoding(
            d_model=config['embed_dim'], dropout=config['dropout']
        )
        
        self.core = nn.ModuleList([
            GaugeConvolutionBlock(
                config['embed_dim'], config['d_ff'], config['dropout'],
                rank=config.get('gauge_rank', 16),
                window_size=config.get('gauge_window_size', 8)
            ) for _ in range(config['num_blocks'])
        ])
        
        # --- The head now predicts the noise, not the energy ---
        # It needs to output a tensor of the same shape as the input.
        self.noise_prediction_head = nn.Linear(config['embed_dim'], config['embed_dim'])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, noisy_embeddings, timesteps):
        """
        The forward pass now takes noised embeddings and the noise level (timesteps).
        It returns the predicted noise.
        """
        # 1. Encode the noise level 't' into a time embedding
        time_embedding = self._get_time_embedding(timesteps, self.config['embed_dim']).to(noisy_embeddings.device)
        time_embedding = self.time_mlp(time_embedding)
        
        # 2. Add time embedding to the input
        # We add it to each token in the sequence.
        x = noisy_embeddings + time_embedding.unsqueeze(1)
        x = self.pos_encoder(x)
        
        # 3. Process through the core architecture
        for block in self.core:
            x = block(x)
        
        # 4. Predict the noise from the final state
        predicted_noise = self.noise_prediction_head(x)
        return predicted_noise

    def _get_time_embedding(self, timesteps, embedding_dim):
        """Creates sinusoidal time embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1: # Zero pad if odd
            emb = F.pad(emb, (0, 1))
        return emb