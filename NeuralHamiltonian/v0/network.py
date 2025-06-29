# network.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HamiltonianBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_order_configured: int = 2,
        dropout_p: float = 0.1
    ):
        super().__init__()
        if max_order_configured < 1:
            raise ValueError("max_order_configured must be at least 1.")
        self.embed_dim = embed_dim
        self.max_order_configured = max_order_configured

        self.input_projection = nn.Linear(embed_dim, embed_dim)

        self.order_q_projs = nn.ModuleList()
        self.order_k_projs = nn.ModuleList()
        self.order_v_projs = nn.ModuleList()
        self.order_final_processing_layers = nn.ModuleList()

        for k_idx in range(self.max_order_configured): # 0-indexed: 0 to max_order_configured-1
            actual_order = k_idx + 1
            self.order_final_processing_layers.append(nn.Linear(embed_dim, embed_dim))

            if actual_order == 1:
                self.order_q_projs.append(nn.Identity())
                self.order_k_projs.append(nn.Identity())
                self.order_v_projs.append(nn.Identity())
            else: # Orders >= 2
                self.order_q_projs.append(nn.Linear(embed_dim, embed_dim))
                self.order_k_projs.append(nn.Linear(embed_dim, embed_dim))
                self.order_v_projs.append(nn.Linear(embed_dim, embed_dim))

        self.order_combination_weights = nn.Parameter(torch.ones(self.max_order_configured))
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def _crosstokens_interaction(self, h_base: torch.Tensor, order_idx: int, prev_order_result: torch.Tensor | None):
        actual_order = order_idx + 1

        if actual_order == 1:
            interim_result = h_base # For order 1, use h_base before final processing
        elif actual_order >= 2:
            if actual_order == 2:
                q_input, k_input, v_input = h_base, h_base, h_base
            else: # Orders >= 3
                q_input = prev_order_result if prev_order_result is not None else h_base
                k_input = prev_order_result if prev_order_result is not None else h_base
                v_input = h_base

            q = self.order_q_projs[order_idx](q_input)
            k = self.order_k_projs[order_idx](k_input)
            v = self.order_v_projs[order_idx](v_input)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
            attn_weights = F.softmax(scores, dim=-1)
            interim_result = torch.matmul(self.dropout(attn_weights), v) # Dropout on attention weights
        else:
            return torch.zeros_like(h_base)

        return self.order_final_processing_layers[order_idx](interim_result)

    def forward(
        self,
        x: torch.Tensor,
        current_fixed_order_limit: int,
        mask: torch.Tensor | None = None, # True for non-pad/keep
    ) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device

        h_projected = self.input_projection(x)

        all_order_features = []
        previous_order_output = None

        for order_idx in range(self.max_order_configured):
            actual_order_num = order_idx + 1
            if actual_order_num <= current_fixed_order_limit:
                current_order_interaction = self._crosstokens_interaction(
                    h_projected, order_idx, previous_order_output
                )
                all_order_features.append(current_order_interaction)
                # Update previous_order_output with the result of the current interaction
                # This allows order_idx+1 to use the output of order_idx
                previous_order_output = current_order_interaction
            else:
                all_order_features.append(torch.zeros_like(h_projected))
        
        if not all_order_features: # Should not happen if current_fixed_order_limit >=1
             print("Warning: all_order_features is empty in HamiltonianBlock forward.")
             # Fallback to avoid error on empty stack, though this indicates an issue upstream
             combined_interaction_output = torch.zeros_like(h_projected)
        else:
            stacked_order_features = torch.stack(all_order_features, dim=0)
            order_indices_for_mask = torch.arange(self.max_order_configured, device=device)
            active_order_mask = (order_indices_for_mask < current_fixed_order_limit).float()
            effective_weights = self.order_combination_weights * active_order_mask
            effective_weights = effective_weights.view(self.max_order_configured, 1, 1, 1)
            weighted_features = stacked_order_features * effective_weights
            combined_interaction_output = torch.sum(weighted_features, dim=0)

        processed_output = self.dropout(combined_interaction_output)
        final_output = self.output_norm(x + processed_output)

        if mask is not None:
            if mask.dtype != torch.bool: mask = mask.bool()
            final_output = final_output * mask.unsqueeze(-1).float()
        
        return final_output

class HamiltonianModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        max_seq_len: int,
        model_max_order: int = 2,
        pad_idx: int = 0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.model_max_order = model_max_order # Max order blocks are configured for
        self.pad_idx = pad_idx
        self.current_order_limit_to_use: int = model_max_order # Default

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        # Ensure pos_ids is long for embedding lookup
        self.register_buffer("pos_ids", torch.arange(max_seq_len, dtype=torch.long), persistent=False)
        self.embed_dropout = nn.Dropout(dropout_p)

        self.blocks = nn.ModuleList(
            [HamiltonianBlock(embed_dim, model_max_order, dropout_p=dropout_p)
             for _ in range(num_blocks)]
        )

        self.output_final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def add_positional_embedding(self, x: torch.Tensor, T: int) -> torch.Tensor:
        pos_ids_for_T = self.pos_ids[:T]
        pos_embed = self.position_embedding(pos_ids_for_T)
        return x + pos_embed

    def forward(
        self,
        input_ids: torch.Tensor, # (B, T) of Long token IDs
        padding_mask: torch.Tensor | None = None,  # (B, T) bool (True for non-pad/keep)
        return_hidden: bool = False,
    ):
        B, T = input_ids.shape
        if input_ids.dtype != torch.long:
            raise ValueError(f"HamiltonianModel expects input_ids of dtype torch.long, got {input_ids.dtype}")

        tok_embed = self.token_embedding(input_ids)
        x = self.add_positional_embedding(tok_embed, T)
        x = self.embed_dropout(x)
        
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool: padding_mask = padding_mask.bool()
            x = x * padding_mask.unsqueeze(-1).float()

        # Use the model's current_order_limit_to_use for all blocks in this pass
        k_limit_for_pass = self.current_order_limit_to_use
        # Ensure it's valid, otherwise default to model_max_order
        if not (1 <= k_limit_for_pass <= self.model_max_order):
            # print(f"Warning: Invalid current_order_limit_to_use ({k_limit_for_pass}). Defaulting to {self.model_max_order}.")
            k_limit_for_pass = self.model_max_order

        for blk in self.blocks:
            x = blk(x, current_fixed_order_limit=k_limit_for_pass, mask=padding_mask)

        x_final_norm = self.output_final_norm(x)
        logits = self.lm_head(x_final_norm) # (B, T, vocab_size)

        if return_hidden:
            return logits, x_final_norm
        return logits