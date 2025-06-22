#network.py
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from highorderstats import skewness, kurtosis, _expand_mask # Assuming highorderstats.py is correct

from config import config

# ... (high_order_statistics function and NFTBlock class remain unchanged) ...
def high_order_statistics(X: torch.Tensor, order: int, mask: Optional[torch.Tensor]):
    Out = []
    B, T_dim = X.shape[0], X.shape[1]
    C_dims_tuple = X.shape[2:] 
    calc_dim = 1 
    ref_tensor_for_shape_content = X.mean(dim=calc_dim, keepdim=True) 

    if mask is not None:
        mask_expanded_for_X = _expand_mask(mask, X.ndim)
        num_valid_elements_reduce = mask.sum(dim=calc_dim, keepdim=True).float()
        num_valid_view_shape = (X.shape[0], 1) + (1,) * (X.ndim - 2) 
        num_valid_elements_for_C_dim = num_valid_elements_reduce.view(num_valid_view_shape).clamp_min(1.0)
        masked_X_for_sum = X * mask_expanded_for_X
        mean_val = masked_X_for_sum.sum(dim=calc_dim, keepdim=True) / num_valid_elements_for_C_dim
        if order >= 1: Out.append(mean_val)
        if T_dim < 2: 
            if order >= 2: Out.extend([torch.zeros_like(mean_val)] * 2) 
            if order >= 3: Out.append(torch.zeros_like(mean_val))    
            if order >= 4: Out.append(torch.zeros_like(mean_val))    
        else: 
            if order >= 2:
                var_val_biased = ((X - mean_val).pow(2) * mask_expanded_for_X).sum(dim=calc_dim, keepdim=True) / num_valid_elements_for_C_dim
                can_unbias = (num_valid_elements_for_C_dim > 1.0).float()
                correction_factor = num_valid_elements_for_C_dim / (num_valid_elements_for_C_dim - 1.0).clamp_min(1e-9)
                var_val_unbiased = var_val_biased * correction_factor * can_unbias + var_val_biased * (1 - can_unbias)
                std_val_unbiased = var_val_unbiased.sqrt().clamp_min(1e-9)
                Out.append(std_val_unbiased); Out.append(var_val_unbiased)
            if order >= 3: Out.append(skewness(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True))
            if order >= 4: Out.append(kurtosis(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True, excess=True))
    else: 
        mean_val_no_mask = X.mean(dim=calc_dim, keepdim=True)
        if order >= 1: Out.append(mean_val_no_mask)
        if T_dim < 2:
            if order >= 2: Out.extend([torch.zeros_like(mean_val_no_mask)] * 2)
            if order >= 3: Out.append(torch.zeros_like(mean_val_no_mask))
            if order >= 4: Out.append(torch.zeros_like(mean_val_no_mask))
        else: 
            if order >= 2:
                Out.append(X.std(dim=calc_dim, keepdim=True, unbiased=True).clamp_min(1e-9))
                Out.append(X.var(dim=calc_dim, keepdim=True, unbiased=True))
            if order >= 3: Out.append(skewness(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True)) 
            if order >= 4: Out.append(kurtosis(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True, excess=True)) 
    expected_terms = 0
    if order >=1: expected_terms +=1
    if order >=2: expected_terms +=2
    if order >=3: expected_terms +=1
    if order >=4: expected_terms +=1
    ref_tensor_for_shape = Out[0] if Out else ref_tensor_for_shape_content
    while len(Out) < expected_terms:
        Out.append(torch.zeros_like(ref_tensor_for_shape))
    return Out

class NFTBlock(nn.Module):
    def __init__(self, embed_dim: int, max_order: int = 2, num_configs: int = 8, max_seq_len: int = 60, num_lags: int = 0, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim; self.max_order = max_order; self.num_configs = num_configs
        self.num_lags = num_lags; self.max_seq_len=max_seq_len
        self.projection = nn.Linear(embed_dim, embed_dim)
        num_stat_terms = sum([1 if max_order >= 1 else 0, 2 if max_order >= 2 else 0, 1 if max_order >= 3 else 0, 1 if max_order >= 4 else 0])
        D_interaction = embed_dim + (num_stat_terms * embed_dim) + (num_lags * 1)
        self.config_networks = nn.ModuleList([nn.Sequential(nn.Linear(D_interaction, embed_dim), nn.GELU()) for _ in range(num_configs)])
        self.energy_head = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_configs)])
        self.config_prior = nn.Parameter(torch.zeros(num_configs))
        self.log_beta = nn.Parameter(torch.tensor(0.0)) 
        self.dropout = nn.Dropout(dropout_rate)
    
    def _compute_lagged_dot_products(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
        B, T, C = h.shape; lagged_dp_features = []
        if self.num_lags == 0: return lagged_dp_features
        for lag_k in range(1, self.num_lags + 1):
            h_padded = F.pad(h, (0, 0, lag_k, 0)); h_prev_for_dot = h_padded[:, :T, :] 
            dot_product_tk = (h * h_prev_for_dot).sum(dim=-1, keepdim=True) 
            if mask is not None:
                current_mask_for_lag = torch.ones_like(dot_product_tk, dtype=torch.bool, device=h.device); current_mask_for_lag[:, :lag_k, :] = False 
                final_dp_mask = mask.unsqueeze(-1) & current_mask_for_lag
                dot_product_tk = dot_product_tk.masked_fill(~final_dp_mask, 0.0)
            else: dot_product_tk[:, :lag_k, :] = 0.0 
            lagged_dp_features.append(dot_product_tk)
        return lagged_dp_features

    def _compute_configuration_energies(self, interaction: torch.Tensor):
        config_outputs_list, config_energies_list = [], []
        for net, energy_head_layer in zip(self.config_networks, self.energy_head):
            config_out_single = net(interaction)
            config_energy_single = energy_head_layer(config_out_single).squeeze(-1)
            config_outputs_list.append(config_out_single)
            config_energies_list.append(config_energy_single)
        
        stacked_outputs = torch.stack(config_outputs_list, dim=2)
        stacked_energies = torch.stack(config_energies_list, dim=2)
        return stacked_energies, stacked_outputs

    def _compute_partition_function(self, energies_arg: torch.Tensor, mask: Optional[torch.Tensor], beta: torch.Tensor):
        config_prior_viewed = self.config_prior.view(1, 1, -1)
        biased_energies = energies_arg + config_prior_viewed
        neg_beta_energies = -beta * biased_energies 
        masked_neg_beta_energies = neg_beta_energies.masked_fill(~mask.unsqueeze(-1).bool(), -torch.finfo(energies_arg.dtype).max) if mask is not None else neg_beta_energies
        log_Z = torch.logsumexp(masked_neg_beta_energies, dim=-1)
        config_probs = torch.exp(masked_neg_beta_energies - log_Z.unsqueeze(-1))
        if mask is not None: config_probs = config_probs * mask.unsqueeze(-1).float()
        return log_Z, config_probs

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, temperature_override: Optional[float]=None, return_diagnostics: bool=False, sampling_mode: str="expectation"):
        B, T, C = x.shape; device=x.device
        h = self.projection(x)
        hot_terms_calculated = high_order_statistics(h, self.max_order, mask)
        expanded_hot_terms = [ht.expand(-1, T, -1) for ht in hot_terms_calculated]
        lagged_dp_features = self._compute_lagged_dot_products(h, mask)
        interaction = torch.cat([x] + expanded_hot_terms + lagged_dp_features, dim=-1)
        
        energies, outputs = self._compute_configuration_energies(interaction)
        beta = torch.tensor(1.0 / temperature_override, device=device, dtype=x.dtype) if temperature_override is not None else torch.exp(self.log_beta.to(x.dtype))
        log_Z, config_probs = self._compute_partition_function(energies, mask, beta)

        if sampling_mode == "expectation": out = (config_probs.unsqueeze(-1) * outputs).sum(dim=2) 
        elif sampling_mode == "sample":
            flat_probs = config_probs.reshape(-1, self.num_configs)
            if mask is not None:
                flat_mask = mask.reshape(-1); active_indices = flat_mask.nonzero(as_tuple=True)[0]
                sampled_configs_flat = torch.zeros(B * T, dtype=torch.long, device=device)
                if active_indices.numel() > 0:
                    probs_for_active = (flat_probs[active_indices] + 1e-9) / (flat_probs[active_indices] + 1e-9).sum(dim=-1, keepdim=True)
                    sampled_configs_flat[active_indices] = torch.multinomial(probs_for_active, num_samples=1).squeeze(-1)
                sampled_configs = sampled_configs_flat.reshape(B, T)
            else:
                flat_probs_stable = (flat_probs + 1e-9) / (flat_probs + 1e-9).sum(dim=-1, keepdim=True)
                sampled_configs = torch.multinomial(flat_probs_stable, num_samples=1).squeeze(-1).reshape(B,T)
            idx = sampled_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, C)
            out = outputs.gather(dim=2, index=idx).squeeze(2)
        elif sampling_mode == "map":
            map_configs = config_probs.argmax(dim=-1)
            idx = map_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, C)
            out = outputs.gather(dim=2, index=idx).squeeze(2)
        else: raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        
        if mask is not None: out = out * mask.unsqueeze(-1).float()
        out = self.dropout(out)
        if return_diagnostics:
            safe_probs = config_probs.clamp_min(1e-20); entropy = -(config_probs * torch.log(safe_probs)).sum(dim=-1)
            if mask is not None: entropy = entropy * mask.float()
            diag_temp = 1.0 / beta.item() if beta.numel()==1 else (1.0/beta).tolist()
            return out, {'log_Z': log_Z, 'config_probs': config_probs, 'energies': energies, 'temperature': diag_temp, 'entropy': entropy}
        return out

class NFTModel(nn.Module):
    def __init__(self, embed_dim: int, num_blocks:int=1, max_order: int = 2,
                 num_configs: int = 8, max_seq_len: int = 60, num_lags: int = 0,
                 dropout_rate: float = 0.1,
                 # LSTM Positional Encoder parameters
                 lstm_pe_layers: int = 1,
                 lstm_pe_bidirectional: bool = False):
        super().__init__()
        self.embed_dim = embed_dim; self.max_order = max(1, min(max_order, 4)); self.num_blocks = num_blocks
        self.num_configs = num_configs; self.num_lags = num_lags; self.max_seq_len=max_seq_len

        if lstm_pe_layers < 1:
            raise ValueError("lstm_pe_layers must be at least 1.")

        # Determine LSTM hidden dimension based on bidirectionality to match embed_dim
        # If bidirectional, LSTM hidden dim is embed_dim / 2, output is 2 * (embed_dim / 2) = embed_dim
        # If unidirectional, LSTM hidden dim is embed_dim, output is embed_dim
        self.lstm_pe_hidden_dim = embed_dim // 2 if lstm_pe_bidirectional else embed_dim
        
        # Sanity check for bidirectional case if embed_dim is odd
        if lstm_pe_bidirectional and embed_dim % 2 != 0:
            print(f"Warning: embed_dim ({embed_dim}) is odd. For bidirectional LSTM PE, "
                  f"output dim will be {self.lstm_pe_hidden_dim * 2}, which might not perfectly match embed_dim. "
                  "Consider using an even embed_dim or a projection layer.")
            # If a projection is desired:
            # self.lstm_pe_output_proj = nn.Linear(self.lstm_pe_hidden_dim * 2, embed_dim)
        
        self.lstm_pe = nn.LSTM(
            input_size=embed_dim, # Input is the raw token embedding
            hidden_size=self.lstm_pe_hidden_dim,
            num_layers=lstm_pe_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_pe_layers > 1 else 0.0,
            bidirectional=lstm_pe_bidirectional
        )
        
        # Optional: Projection layer if bidirectional output needs to be precisely embed_dim
        # and embed_dim was odd or you want explicit control.
        # if lstm_pe_bidirectional and hasattr(self, 'lstm_pe_output_proj'):
        #    pass # already defined above
        # elif lstm_pe_bidirectional and self.lstm_pe_hidden_dim * 2 != embed_dim :
        #     self.lstm_pe_output_proj = nn.Linear(self.lstm_pe_hidden_dim * 2, embed_dim)
        # else:
        #     self.lstm_pe_output_proj = nn.Identity()


        self.blocks = nn.ModuleList([
            NFTBlock(embed_dim, self.max_order, num_configs, max_seq_len, num_lags, dropout_rate) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout_rate))
        self.lm_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None, temperature_override: Optional[float]=None, return_diagnostics: bool=False, sampling_mode: str="expectation"):
        B, T, C = x.shape
        if T > self.max_seq_len: raise ValueError(f"Input sequence length ({T}) exceeds model's max_seq_len ({self.max_seq_len})")

        # LSTM for positional encoding processes raw embeddings
        lstm_pe_out, _ = self.lstm_pe(x) # x is the raw input (B, T, embed_dim)
        h = lstm_pe_out
        
        # Apply projection if defined (e.g., for precise dimension matching with bidirectional LSTM)
        # if hasattr(self, 'lstm_pe_output_proj'):
        #    h = self.lstm_pe_output_proj(h)


        if mask is not None:
            h = h * mask.unsqueeze(-1).float() # Apply mask after LSTM PE

        all_diagnostics = [] if return_diagnostics else None

        # Proceed with NFTBlocks and MLP
        for blk_idx, blk in enumerate(self.blocks):
            h_residual_block = h
            current_block_input = self.norm(h)
            if return_diagnostics:
                block_output = blk(current_block_input, mask=mask, temperature_override=temperature_override, return_diagnostics=True, sampling_mode=sampling_mode)
                h_out_nft, block_diagn = block_output if isinstance(block_output, tuple) else (block_output, None)
                if all_diagnostics is not None and block_diagn is not None: all_diagnostics.append(block_diagn)
            else:
                h_out_nft = blk(current_block_input, mask=mask, temperature_override=temperature_override, return_diagnostics=False, sampling_mode=sampling_mode)
            h = h_residual_block + h_out_nft
            h_residual_mlp = h
            h_mlp_out = self.mlp(self.norm(h))
            h = h_residual_mlp + h_mlp_out

        h_normed_final = self.norm(h)
        logits = self.lm_head(h_normed_final)

        if return_diagnostics: return logits, h_normed_final, all_diagnostics
        return logits

# --- Update Encoder, Decoder, EncoderDecoderModel, load_encoder_decoder_model ---
# These classes need to be updated to accept and pass through the simplified LSTM-PE parameters.

class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_blocks: int, max_order: int,
                 num_configs: int, max_seq_len: int, num_lags: int, dropout_rate: float,
                 # Simplified PE params
                 lstm_pe_layers: int = 1, lstm_pe_bidirectional: bool = False):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.nft_encoder_core = NFTModel( # Pass only relevant params
            embed_dim=embed_dim, num_blocks=num_blocks, max_order=max_order,
            num_configs=num_configs, max_seq_len=max_seq_len, num_lags=num_lags,
            dropout_rate=dropout_rate,
            lstm_pe_layers=lstm_pe_layers,
            lstm_pe_bidirectional=lstm_pe_bidirectional
        )
    # ... forward method remains the same ...
    def forward(self, x_encoder_input: torch.Tensor, mask: Optional[torch.Tensor] = None, return_full_states=False):
        projected_input = self.input_projection(x_encoder_input)
        sampling_mode_to_use = config["training"].get("sampling_mode", "expectation")
        core_output = self.nft_encoder_core(projected_input, mask=mask, return_diagnostics=True, sampling_mode=sampling_mode_to_use)
        _, encoder_hidden_states, _ = core_output
        context_vector = encoder_hidden_states[:, -1, :]
        if return_full_states:
            return context_vector, encoder_hidden_states
        return context_vector


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, context_dim: int, num_blocks: int, max_order: int,
                 num_configs: int, max_seq_len: int, num_lags: int, dropout_rate: float,
                 # Simplified PE params
                 lstm_pe_layers: int = 1, lstm_pe_bidirectional: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_projection = nn.Linear(context_dim, embed_dim)
        self.nft_decoder_core = NFTModel( # Pass only relevant params
            embed_dim=embed_dim, num_blocks=num_blocks, max_order=max_order,
            num_configs=num_configs, max_seq_len=max_seq_len, num_lags=num_lags,
            dropout_rate=dropout_rate,
            lstm_pe_layers=lstm_pe_layers,
            lstm_pe_bidirectional=lstm_pe_bidirectional
        )
    # ... forward method remains the same ...
    def forward(self, x_decoder_input: torch.Tensor, encoder_context: torch.Tensor, mask: Optional[torch.Tensor] = None, return_full_diagnostics: bool = False):
        projected_context = self.context_projection(encoder_context)
        projected_context_expanded = projected_context.unsqueeze(1).expand(-1, x_decoder_input.size(1), -1)
        decoder_input_with_context = x_decoder_input + projected_context_expanded
        
        sampling_mode_to_use = config["training"].get("sampling_mode", "expectation")
        if return_full_diagnostics:
             predictions, hidden_state, diagnostics = self.nft_decoder_core(decoder_input_with_context, mask=mask, return_diagnostics=True, sampling_mode=sampling_mode_to_use)
             return predictions, hidden_state, diagnostics
        else:
            predictions = self.nft_decoder_core(decoder_input_with_context, mask=mask, return_diagnostics=False, sampling_mode=sampling_mode_to_use)
            return predictions


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder_input_dim: int, decoder_embed_dim: int, shared_embed_dim: int,
                 num_blocks_enc: int, num_blocks_dec: int, max_order_enc: int, max_order_dec: int,
                 num_configs_enc: int, num_configs_dec: int, max_seq_len_enc: int, max_seq_len_dec: int,
                 num_lags_enc: int, num_lags_dec: int, dropout_rate: float,
                 # Direct LSTM-PE parameters for encoder and decoder
                 lstm_pe_layers_enc: int = 1, lstm_pe_bidirectional_enc: bool = False,
                 lstm_pe_layers_dec: int = 1, lstm_pe_bidirectional_dec: bool = False
                 ):
        super().__init__()
        self.encoder = Encoder(
            input_dim=encoder_input_dim, embed_dim=shared_embed_dim, num_blocks=num_blocks_enc,
            max_order=max_order_enc, num_configs=num_configs_enc, max_seq_len=max_seq_len_enc,
            num_lags=num_lags_enc, dropout_rate=dropout_rate,
            lstm_pe_layers=lstm_pe_layers_enc, # Pass enc-specific
            lstm_pe_bidirectional=lstm_pe_bidirectional_enc
        )
        self.decoder = Decoder(
            embed_dim=decoder_embed_dim, context_dim=shared_embed_dim, num_blocks=num_blocks_dec,
            max_order=max_order_dec, num_configs=num_configs_dec, max_seq_len=max_seq_len_dec,
            num_lags=num_lags_dec, dropout_rate=dropout_rate,
            lstm_pe_layers=lstm_pe_layers_dec, # Pass dec-specific
            lstm_pe_bidirectional=lstm_pe_bidirectional_dec
        )
        self.decoder_embed_dim = decoder_embed_dim
        self.max_seq_len_dec = max_seq_len_dec
    def forward(self, x_encoder_input: torch.Tensor, x_decoder_input: torch.Tensor, 
                encoder_mask: Optional[torch.Tensor] = None, decoder_mask: Optional[torch.Tensor] = None,
                return_decoder_full_diagnostics: bool = False): 
        encoder_context = self.encoder(x_encoder_input, mask=encoder_mask) 
        if return_decoder_full_diagnostics:
            predictions, hidden_state, diagnostics = self.decoder(x_decoder_input, encoder_context, mask=decoder_mask, return_full_diagnostics=True)
            return predictions, hidden_state, diagnostics
        else:
            predictions = self.decoder(x_decoder_input, encoder_context, mask=decoder_mask, return_full_diagnostics=False)
            return predictions

    def predict_autoregressive(self, x_encoder_input: torch.Tensor, sos_token_val: float = 0.0, encoder_mask: Optional[torch.Tensor] = None, device='cpu'):
        batch_size = x_encoder_input.size(0)
        encoder_context = self.encoder(x_encoder_input, mask=encoder_mask) 
        built_decoder_input_sequence = torch.full((batch_size, 1, self.decoder_embed_dim), sos_token_val, device=device, dtype=x_encoder_input.dtype)
        predicted_sequence_outputs = []
        for _ in range(self.max_seq_len_dec):
            decoder_output_for_current_built_sequence = self.decoder(built_decoder_input_sequence, encoder_context, return_full_diagnostics=False)
            next_step_prediction = decoder_output_for_current_built_sequence[:, -1, :] 
            predicted_sequence_outputs.append(next_step_prediction)
            if len(predicted_sequence_outputs) < self.max_seq_len_dec:
                 built_decoder_input_sequence = torch.cat([built_decoder_input_sequence, next_step_prediction.unsqueeze(1)], dim=1)
        return torch.stack(predicted_sequence_outputs, dim=1)


def load_encoder_decoder_model(device):
    model_path = config["data"]["model_file_path"]
    if not os.path.exists(model_path):
        print(f"No checkpoint for EncoderDecoderModel at {model_path}. Starting fresh."); return None, None, 0
    checkpoint = torch.load(model_path, map_location=device); chkpt_model_params = checkpoint.get('config_model_params')
    if not chkpt_model_params: print("ERROR: Checkpoint 'config_model_params' missing."); return None, None, 0

    encoder_input_dim_to_use = chkpt_model_params.get("encoder_input_dim", config["model"]["encoder_input_dim"])

    # Resolve LSTM-PE params: Use specific from checkpoint, fallback to shared from checkpoint,
    # then specific from current config, then shared from current config, then default.
    default_lstm_pe_layers = 1
    default_lstm_pe_bidirectional = False

    # Encoder PE
    lstm_pe_layers_enc_to_use = chkpt_model_params.get(
        "lstm_pe_layers_enc",
        chkpt_model_params.get("lstm_pe_layers",
                               config["model"].get("lstm_pe_layers_enc",
                                                   config["model"].get("lstm_pe_layers", default_lstm_pe_layers)))
    )
    lstm_pe_bidirectional_enc_to_use = chkpt_model_params.get(
        "lstm_pe_bidirectional_enc",
        chkpt_model_params.get("lstm_pe_bidirectional",
                               config["model"].get("lstm_pe_bidirectional_enc",
                                                   config["model"].get("lstm_pe_bidirectional", default_lstm_pe_bidirectional)))
    )
    # Decoder PE
    lstm_pe_layers_dec_to_use = chkpt_model_params.get(
        "lstm_pe_layers_dec",
        chkpt_model_params.get("lstm_pe_layers",
                               config["model"].get("lstm_pe_layers_dec",
                                                   config["model"].get("lstm_pe_layers", default_lstm_pe_layers)))
    )
    lstm_pe_bidirectional_dec_to_use = chkpt_model_params.get(
        "lstm_pe_bidirectional_dec",
        chkpt_model_params.get("lstm_pe_bidirectional",
                               config["model"].get("lstm_pe_bidirectional_dec",
                                                   config["model"].get("lstm_pe_bidirectional", default_lstm_pe_bidirectional)))
    )

    try:
        model = EncoderDecoderModel(
            encoder_input_dim=encoder_input_dim_to_use,
            decoder_embed_dim=chkpt_model_params.get("decoder_embed_dim"),
            shared_embed_dim=chkpt_model_params.get("shared_embed_dim"),
            num_blocks_enc=chkpt_model_params.get("num_blocks_enc"), num_blocks_dec=chkpt_model_params.get("num_blocks_dec"),
            max_order_enc=chkpt_model_params.get("max_order_enc"), max_order_dec=chkpt_model_params.get("max_order_dec"),
            num_configs_enc=chkpt_model_params.get("num_configs_enc"), num_configs_dec=chkpt_model_params.get("num_configs_dec"),
            max_seq_len_enc=chkpt_model_params.get("max_seq_len_enc"), max_seq_len_dec=chkpt_model_params.get("max_seq_len_dec"),
            num_lags_enc=chkpt_model_params.get("num_lags_enc"), num_lags_dec=chkpt_model_params.get("num_lags_dec"),
            dropout_rate=chkpt_model_params.get("dropout_rate"),
            # Pass resolved enc/dec specific PE params
            lstm_pe_layers_enc=lstm_pe_layers_enc_to_use,
            lstm_pe_bidirectional_enc=lstm_pe_bidirectional_enc_to_use,
            lstm_pe_layers_dec=lstm_pe_layers_dec_to_use,
            lstm_pe_bidirectional_dec=lstm_pe_bidirectional_dec_to_use
        ).to(device)
    except KeyError as e: print(f"ERROR: Missing param in chkpt/config: {e}"); return None,None,0
    except Exception as e: print(f"ERROR instantiating EncoderDecoderModel: {e}"); return None,None,0
    # ... rest of loading ...
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Warning: RuntimeError loading state_dict, possibly due to architecture changes: {e}")
        print("Model will be used with initialized weights for new/changed layers.")

    optimizer_state_dict = checkpoint.get('optimizer_state_dict'); epoch = checkpoint.get('epoch', 0)
    print(f"EncoderDecoderModel loaded from {model_path}. Resuming from epoch {epoch}."); return model, optimizer_state_dict, epoch