import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict # Added Dict

class PartitionFunctionMFI(nn.Module):
    """Mean-Field Interaction using partition function formulation.
    
    Instead of directly computing weighted k-body terms, we:
    1. Compute energy for different interaction configurations
    2. Use partition function Z = Σ exp(-βE_i) to get probabilities
    3. Sample/weight configurations according to Boltzmann distribution
    """
    
    def __init__(self, embed_dim: int, max_order: int = 2, num_configs: int = 8): # max_order not used in current MFI
        super().__init__()
        self.embed_dim = embed_dim
        # self.max_order = max_order # Not directly used by this MFI implementation parts
        self.num_configs = num_configs
        
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.config_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim), # Interaction is h and z
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_configs)
        ])
        
        self.energy_head = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_configs)
        ])
        
        self._initial_log_beta = torch.log(torch.tensor(1.0))
        self._initial_config_prior = torch.zeros(num_configs)

        self.log_beta = nn.Parameter(self._initial_log_beta.clone())
        self.config_prior = nn.Parameter(self._initial_config_prior.clone())
        
        self.reset_parameters()

    def reset_parameters(self):
        # Reset PyTorch standard layers
        self.projection.reset_parameters()
        self.norm.reset_parameters()
        for seq_module in self.config_networks:
            for layer in seq_module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.energy_head:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Reset custom nn.Parameters to their initial values
        with torch.no_grad(): # Important to do this without tracking gradients
            self.log_beta.data.copy_(self._initial_log_beta)
            self.config_prior.data.copy_(self._initial_config_prior)

    def _compute_mean_field(self, h: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            masked_h = h * mask_f
            z = masked_h.sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_(min=1e-8)
        else:
            z = h.mean(dim=1, keepdim=True)
        z_normed = self.norm(z)
        return z_normed.expand(-1, h.size(1), -1) 
    
    def _compute_configuration_energies(self, interaction: torch.Tensor):
        B, T, _ = interaction.shape
        outputs = []
        energies = []
        
        for i, (net, energy_head) in enumerate(zip(self.config_networks, self.energy_head)):
            config_out = net(interaction)
            config_energy = energy_head(config_out).squeeze(-1)
            outputs.append(config_out)
            energies.append(config_energy)
        
        outputs = torch.stack(outputs, dim=2)
        energies = torch.stack(energies, dim=2)
        return energies, outputs
    
    def _compute_partition_function(self, energies: torch.Tensor, mask: Optional[torch.Tensor],beta):
        prior_bias = self.config_prior.view(1, 1, -1)
        biased_energies = energies + prior_bias
        boltzmann_weights = torch.exp(-beta * biased_energies)
        
        if mask is not None:
            boltzmann_weights = boltzmann_weights * mask.unsqueeze(-1).float()
        
        Z = boltzmann_weights.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        log_Z = torch.log(Z.squeeze(-1))
        config_probs = boltzmann_weights / Z
        return log_Z, config_probs
    
    def forward(
        self,
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[float] = None,
        return_diagnostics: bool = False, # Critical: this must be True if CompositeCriterion is used
        sampling_mode: str = "expectation"
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]: 
        B, T, C = x.shape
        
        z = self._compute_mean_field(x, mask)
        interaction = torch.cat([x, z], dim=-1)
        energies, outputs = self._compute_configuration_energies(interaction)
        
        if temperature_override is not None:
            beta = torch.tensor(1.0 / temperature_override, device=self.log_beta.device)
        else:
            beta = torch.exp(self.log_beta)
        
        log_Z, config_probs = self._compute_partition_function(energies, mask,beta)
        
        if sampling_mode == "expectation":
            out = (config_probs.unsqueeze(-1) * outputs).sum(dim=2)
        elif sampling_mode == "sample":
            flat_probs = config_probs.view(-1, self.num_configs)
            if mask is not None:
                flat_mask = mask.view(-1)
                active_indices = flat_mask.nonzero(as_tuple=True)[0]
                if active_indices.numel() > 0: 
                    sampled_configs_flat = torch.multinomial(flat_probs[active_indices], num_samples=1)
                    sampled_configs = torch.zeros(B * T, dtype=torch.long, device=x.device)
                    sampled_configs[active_indices] = sampled_configs_flat.squeeze(-1)
                    sampled_configs = sampled_configs.view(B, T)
                else: 
                    sampled_configs = torch.zeros(B, T, dtype=torch.long, device=x.device)
            else: 
                sampled_configs_flat = torch.multinomial(flat_probs, num_samples=1)
                sampled_configs = sampled_configs_flat.view(B, T)
            out = outputs.gather(2, sampled_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C))
            out = out.squeeze(2)
        elif sampling_mode == "map":
            map_configs = config_probs.argmax(dim=-1)
            out = outputs.gather(2, map_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C))
            out = out.squeeze(2)
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        
        if return_diagnostics: # This part is key
            diagnostics = {
                'log_Z': log_Z,
                'config_probs': config_probs, # This is what we need for gate_entropy_regularizer
                'energies': energies,
                'temperature': 1.0 / beta.item(), # Use .item() for scalar
                'entropy': -(config_probs * torch.log(config_probs.clamp_min(1e-10))).sum(dim=-1)
            }
            return out, diagnostics
        
        return out


class PartitionFunctionMFIBlock(nn.Module):
    def __init__(self, embed_dim: int, max_order: int = 2, num_configs: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.mfi = PartitionFunctionMFI(embed_dim, max_order, num_configs)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.reset_parameters()
    def reset_parameters(self):
        if hasattr(self.mfi, 'reset_parameters'):
            self.mfi.reset_parameters()
        self.norm1.reset_parameters()
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.norm2.reset_parameters()
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[float] = None,
        sampling_mode: str = "expectation",
        return_diagnostics: bool = False # This flag is passed down
    ):
        normed_x = self.norm1(x)

        mfi_output_data = self.mfi( # Always pass return_diagnostics flag
            normed_x,
            mask=mask,
            temperature_override=temperature_override,
            sampling_mode=sampling_mode,
            return_diagnostics=return_diagnostics # Pass it through
        )
        
        if return_diagnostics:
            mfi_out_val, diagnostics = mfi_output_data
            # diagnostics structure is {'mfi': {'log_Z': ..., 'config_probs': ...}}
            # We want to return diagnostics directly from self.mfi
        else:
            mfi_out_val = mfi_output_data
            diagnostics = None # No diagnostics to return
        
        res_x = x + mfi_out_val 
        out_x = res_x + self.mlp(self.norm2(res_x)) # Apply MLP to the output of first residual
        
        if return_diagnostics:
            return out_x, diagnostics # Return MFI's diagnostics directly
        return out_x
    


class OverallLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: Optional[int], 
                 max_order: int, num_configs: int, mlp_ratio: float, num_blocks: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.blocks = nn.ModuleList([
            PartitionFunctionMFIBlock(
                embed_dim=embed_dim,
                max_order=max_order, # This max_order is for MFIBlock init, not MFI itself
                num_configs=num_configs,
                mlp_ratio=mlp_ratio
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.reset_parameters() # Initialize correctly on creation

    def reset_parameters(self):
        self.token_embedding.reset_parameters()
        for block in self.blocks:
            if hasattr(block, 'reset_parameters'):
                block.reset_parameters()
        self.final_norm.reset_parameters()
        self.to_logits.reset_parameters()

    def forward(self, input_ids: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                # Renamed for clarity: `return_for_criterion` implies specific needs for CompositeCriterion
                return_for_criterion: bool = False, 
                temperature_override: Optional[float] = None,
                sampling_mode: str = "expectation"):

        x_emb = self.token_embedding(input_ids)
        
        collected_config_probs = [] # To collect 'config_probs' from each block

        current_x = x_emb
        for block in self.blocks:
            # We need diagnostics from MFI if return_for_criterion is True
            block_output = block(
                current_x,
                mask=mask,
                temperature_override=temperature_override,
                sampling_mode=sampling_mode,
                return_diagnostics=return_for_criterion # Pass flag down
            )
            if return_for_criterion:
                current_x, block_mfi_diagnostics = block_output
                collected_config_probs.append(block_mfi_diagnostics['config_probs'])
            else:
                current_x = block_output
        
        processed_emb = self.final_norm(current_x) # This is 'hidden' for CompositeCriterion
        logits = self.to_logits(processed_emb)

        if return_for_criterion:
            # Return logits, final processed_emb (as hidden), and list of config_probs
            return logits, processed_emb, collected_config_probs
        else:
            return logits # Default: just return logits

    # ... (generate method remains the same, it calls forward with return_for_criterion=False by default) ...
    @torch.no_grad()
    def generate(self, tokenizer, start_text: str, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 mfi_temperature_override: Optional[float] = None, 
                 mfi_sampling_mode: str = "expectation",      
                 device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
        generated_ids = input_ids
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id 

        for _ in range(max_new_tokens):
            gen_mask = (generated_ids != pad_token_id) if pad_token_id is not None else None
            
            # Generate method calls forward without requesting criterion-specific outputs
            outputs = self.forward(generated_ids, mask=gen_mask, 
                                   return_for_criterion=False, # Important for generation
                                   temperature_override=mfi_temperature_override,
                                   sampling_mode=mfi_sampling_mode)
            
            next_token_logits = outputs[:, -1, :]

            if temperature == 0: 
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)