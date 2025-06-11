# nft_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from highorderstats import skewness, kurtosis # Assuming highorderstats.py is in the same directory

def high_order_statistics(X: torch.Tensor, order: int, mask: Optional[torch.Tensor]):
    Out = []
    calc_dim = 1 

    if mask is not None:
        mask_expanded_for_X = mask.unsqueeze(-1).float()
        num_valid_elements_reduce = mask.sum(dim=calc_dim, keepdim=True).float() 
        num_valid_elements_for_C_dim = num_valid_elements_reduce.unsqueeze(-1).clamp_min(1.0)
        
        masked_X_for_sum = X * mask_expanded_for_X 

        if order >= 1: 
            mean_val = masked_X_for_sum.sum(dim=calc_dim, keepdim=True) / num_valid_elements_for_C_dim
            Out.append(mean_val)
        if order >= 2: 
            var_val_biased = ((X - Out[0]).pow(2) * mask_expanded_for_X).sum(dim=calc_dim, keepdim=True) / num_valid_elements_for_C_dim
            can_unbias = (num_valid_elements_for_C_dim > 1.0).float()
            correction_factor = num_valid_elements_for_C_dim / (num_valid_elements_for_C_dim - 1.0).clamp_min(1e-9)
            var_val_unbiased = var_val_biased * correction_factor * can_unbias + var_val_biased * (1-can_unbias)
            std_val_unbiased = var_val_unbiased.sqrt().clamp_min(1e-9)
            Out.append(std_val_unbiased)
            Out.append(var_val_unbiased)
    else: 
        if order >= 1:
            Out.append(X.mean(dim=calc_dim, keepdim=True))
        if order >= 2:
            Out.append(X.std(dim=calc_dim, keepdim=True, unbiased=True)) 
            Out.append(X.var(dim=calc_dim, keepdim=True, unbiased=True))
    
    if order >= 3:
        Out.append(skewness(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True))
    if order >= 4:
        Out.append(kurtosis(X, dim=calc_dim, keepdim=True, mask=mask, unbiased=True, excess=True))
    return Out

class NFTnetworkBlock(nn.Module):
    def __init__(self, embed_dim: int,max_order: int = 2, num_configs: int = 8):
        super().__init__()
        self.embed_dim=embed_dim
        if max_order > 4:
            print("Max order is too high, please choose <=4. Setting to 4.")
            self.max_order = 4
        elif max_order < 1:
            print("Max order is too low, please choose >=1. Setting to 1.")
            self.max_order = 1
        else:
            self.max_order=max_order
            
        self.num_configs=num_configs
        self.norm1 = nn.LayerNorm(embed_dim)

        num_stat_tensors = self._get_num_stats_terms(self.max_order)
        interaction_input_dim = embed_dim * (1 + num_stat_tensors) 

        self.config_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(interaction_input_dim, embed_dim), 
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_configs)
        ])
        
        self.energy_head = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_configs)
        ])
        
        self.log_beta = nn.Parameter(torch.zeros(1)) 
        self.config_prior = nn.Parameter(torch.zeros(num_configs))
        
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure parameters are on the correct device after initialization
        # This is more robust if the model itself is moved to a device later.
        # The .to(device) call on the nn.Module instance will handle parameters.
        # However, for parameters explicitly created like self.log_beta, it's good practice.
        # self.log_beta.data = self.log_beta.data.to(self.device)
        # self.config_prior.data = self.config_prior.data.to(self.device)


    def _get_num_stats_terms(self, max_order_val: int) -> int:
        count = 0
        if max_order_val >= 1: count += 1
        if max_order_val >= 2: count += 2
        if max_order_val >= 3: count += 1
        if max_order_val >= 4: count += 1
        return count

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

    def _compute_partition_function(self, energies: torch.Tensor, mask: Optional[torch.Tensor], beta: torch.Tensor):
        prior_bias = self.config_prior.view(1, 1, -1) 
        biased_energies = energies + prior_bias
        neg_beta_energies = -beta * biased_energies

        if mask is not None:
            masked_neg_beta_energies = neg_beta_energies.masked_fill(~mask.unsqueeze(-1).bool(), -torch.finfo(energies.dtype).max)
        else:
            masked_neg_beta_energies = neg_beta_energies

        log_Z_terms = torch.logsumexp(masked_neg_beta_energies, dim=-1)
        log_Z = log_Z_terms

        log_config_probs = masked_neg_beta_energies - log_Z.unsqueeze(-1)
        config_probs = torch.exp(log_config_probs)
        
        if mask is not None:
             config_probs = config_probs * mask.unsqueeze(-1).float()

        return log_Z, config_probs

    def forward(
        self,
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[float] = None,
        return_diagnostics: bool = False,
        sampling_mode: str = "expectation"
    ):
        B, T, C = x.shape
        current_device = x.device 
        
        h_norm = self.norm1(x)
        
        # hoterms will be a list of tensors, each potentially (B, 1, C_stat)
        hoterms_raw = high_order_statistics(h_norm, self.max_order, mask)
        
        # Expand/tile each statistical term to match the time dimension T of x
        hoterms_expanded = []
        for stat_term in hoterms_raw:
            if stat_term.size(1) == 1 and T > 1: # If time dim is 1 and x's time dim > 1
                # stat_term is (B, 1, C_stat), expand to (B, T, C_stat)
                hoterms_expanded.append(stat_term.expand(-1, T, -1))
            elif stat_term.size(1) == T: # Already matching (e.g. if T=1, or if a stat func didn't reduce time)
                hoterms_expanded.append(stat_term)
            else:
                # This case should ideally not happen if high_order_statistics is correct
                raise ValueError(f"Mismatch in time dimension for statistical term. Expected {T} or 1, got {stat_term.size(1)}")

        interaction_terms = [x] + hoterms_expanded # Use the expanded terms
        interaction = torch.cat(interaction_terms, dim=-1) # Now dimensions should match
        
        energies, outputs = self._compute_configuration_energies(interaction)

        if temperature_override is not None:
            beta = torch.tensor(1.0 / temperature_override, device=current_device, dtype=x.dtype)
        else:
            beta = torch.exp(self.log_beta.to(current_device).to(x.dtype)) 

        self.config_prior = self.config_prior.to(energies.device) # Ensure device consistency
        log_Z, config_probs = self._compute_partition_function(energies, mask, beta)

        # ... (rest of the forward pass remains the same) ...
        if sampling_mode == "expectation":
            out = (config_probs.unsqueeze(-1) * outputs).sum(dim=2)
        elif sampling_mode == "sample":
            flat_probs = config_probs.reshape(-1, self.num_configs)
            
            if mask is not None:
                flat_mask = mask.reshape(-1)
                active_indices = flat_mask.nonzero(as_tuple=True)[0]
                sampled_configs_flat = torch.zeros(B * T, dtype=torch.long, device=current_device)
                if active_indices.numel() > 0:
                    probs_for_active = flat_probs[active_indices]
                    probs_for_active = probs_for_active + 1e-9 
                    probs_for_active = probs_for_active / probs_for_active.sum(dim=-1, keepdim=True)
                    sampled_indices_for_active = torch.multinomial(probs_for_active, num_samples=1).squeeze(-1)
                    sampled_configs_flat[active_indices] = sampled_indices_for_active
                sampled_configs = sampled_configs_flat.reshape(B, T)
            else:
                flat_probs_stable = flat_probs + 1e-9
                flat_probs_stable = flat_probs_stable / flat_probs_stable.sum(dim=-1, keepdim=True)
                sampled_configs_flat = torch.multinomial(flat_probs_stable, num_samples=1).squeeze(-1)
                sampled_configs = sampled_configs_flat.reshape(B, T)

            idx = sampled_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C)
            out = outputs.gather(2, idx).squeeze(2)
        elif sampling_mode == "map":
            map_configs = config_probs.argmax(dim=-1)
            idx = map_configs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C)
            out = outputs.gather(2, idx).squeeze(2)
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        
        if return_diagnostics: 
            safe_config_probs = config_probs.clamp_min(1e-20)
            entropy = -(config_probs * torch.log(safe_config_probs)).sum(dim=-1)
            if mask is not None:
                entropy = entropy * mask.float()

            diagnostics = {
                'log_Z': log_Z,
                'config_probs': config_probs, 
                'energies': energies,
                'temperature': 1.0 / beta.item() if beta.numel() == 1 else (1.0/beta).tolist(), 
                'entropy': entropy
            }
            return out, diagnostics
        
        return out