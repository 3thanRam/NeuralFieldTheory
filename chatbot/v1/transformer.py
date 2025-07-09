import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from network import PositionalEncoding # Re-use the existing PositionalEncoding

class TransformerModel(nn.Module):
    """
    A standard Transformer Encoder model for language modeling.
    It takes a sequence of token IDs and predicts the logits for the next token at each position.
    """
    def __init__(self, vocab_size, embed_dim, nhead, num_encoder_layers, dim_feedforward, dropout, pad_idx):
        super().__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim

        # 1. Token Embedding: Converts token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 2. Positional Encoding: Adds time-step information
        self.pos_encoder = PositionalEncoding(embed_dim)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: our data is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 4. Output Head (Language Model Head): Projects from embedding dimension to the vocabulary size
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None, return_internals: bool = False) -> torch.Tensor | tuple:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] of token IDs
            mask: Tensor, shape [batch_size, seq_len]. Boolean tensor where True indicates a non-padding token.
            return_internals: bool, for API compatibility with the training loop.

        Returns:
            - logits: Tensor, shape [batch_size, seq_len, vocab_size]
            - (if return_internals=True) A tuple (logits, (None, None, None, None)) for compatibility
        """
        # Embed tokens, scale, add positional encoding, and apply dropout
        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)

        # The nn.TransformerEncoder expects a src_key_padding_mask where True means "ignore".
        # Our input mask has True for "keep". So we must invert it.
        if mask is not None:
            # `~` is the bitwise NOT operator, which works as logical NOT for booleans.
            padding_mask = ~mask
        else:
            padding_mask = None

        # Pass through the encoder
        output = self.transformer_encoder(src_emb, src_key_padding_mask=padding_mask)

        # Project to output vocabulary
        logits = self.lm_head(output)

        if return_internals:
            # Return a tuple that matches the HamiltonianModel's output signature
            # This ensures the training loop `output, (Q_i, ...)` unpacking works.
            return logits, (None, None, None, None)
        else:
            return logits

    @torch.no_grad()
    def generate(self, tokenizer, start_text, max_new_tokens, device, temperature=0.8, top_k=40):
        """
        Autoregressively generate text. This is compatible with `cli.py`.
        """
        self.eval()
        input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)

        for _ in range(max_new_tokens):
            # The model is not stateful, so we pass the full sequence each time
            logits = self(input_ids) # Shape: (1, seq_len, vocab_size)

            # Get logits for the very last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the new token and check for end-of-sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text