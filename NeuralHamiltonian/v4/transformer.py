import torch
import torch.nn as nn
import numpy as np
from network import PositionalEncoding # Re-use the existing PositionalEncoding
from config import config # For accessing hyperparameters and symbol info

class TransformerModel(nn.Module):
    """
    A standard Transformer Encoder model for sequence prediction.
    It takes a sequence of feature vectors and predicts a sequence of the same length.
    """
    def __init__(self, input_dim, output_dim, d_embedding, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_embedding = d_embedding

        # 1. Input Projection: Project input features to the model's embedding dimension
        self.input_projection = nn.Linear(input_dim, d_embedding)

        # 2. Positional Encoding: Add time-step information
        self.pos_encoder = PositionalEncoding(d_embedding)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_embedding, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Important: our data is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 4. Output Head: Project from embedding dimension to the final output dimension
        self.lm_head = nn.Linear(d_embedding, output_dim)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
            src_mask: Tensor, shape [seq_len, seq_len] (optional)
        
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
        """
        # Project, add positional encoding, and apply dropout
        src = self.input_projection(src) * torch.tensor(self.d_embedding).sqrt()
        src = self.pos_encoder(src)
        
        # Pass through the encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # Project to output dimension
        logits = self.lm_head(output)
        return logits

    @torch.no_grad()
    def generate(self, input_sequence: torch.Tensor, n_to_pred: int):
        """
        Autoregressively generate future predictions. This logic is identical
        to the HamiltonianModel's generate method to ensure compatibility with the
        testing script.
        """
        self.eval()
        
        # Get symbol information from config
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
            # Feed the last `sequence_length` steps to the model
            input_for_pred = current_input_seq[:, -config["sequence_length"]:, :]

            # Get model prediction for the entire sequence
            full_prediction_sequence = self(input_for_pred)

            # We only need the prediction for the very last time step
            next_primary_ohlc = full_prediction_sequence[:, -1:, :]

            predictions.append(next_primary_ohlc.cpu().numpy())

            # Construct the next full input step for autoregression
            last_full_step = current_input_seq[:, -1:, :].clone()
            
            # Replace the old primary symbol's data with the new prediction
            last_full_step[:, :, start_feature_idx:end_feature_idx] = next_primary_ohlc

            # Append this complete new step to the input sequence
            current_input_seq = torch.cat([current_input_seq, last_full_step], dim=1)

        return np.concatenate(predictions, axis=1).squeeze(0)