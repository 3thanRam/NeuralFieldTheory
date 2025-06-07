#lossfunction.py
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict # Added for type hints

def nll_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = -100) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_idx 
    )

def entropy_regularizer(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor: # logits: (B,T,V)
    probs = F.softmax(logits, dim=-1)
    ent_per_token = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)  # (B,T)
    
    if mask is not None:
        ent_masked = ent_per_token * mask.float()
        norm = mask.float().sum().clamp_min(1.0) 
        return ent_masked.sum() / norm
    else:
        norm = float(ent_per_token.numel()) # Ensure float for division
        return ent_per_token.sum() / norm

def decorrelation_regularizer(hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # hidden: (B,T,C)
    B, T, C = hidden.shape
    if T <= 1:
        return torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)

    # Normalize hidden states along the feature dimension
    normed_hidden = hidden / (hidden.norm(dim=-1, keepdim=True).clamp_min(1e-10))
    
    # Compute pairwise cosine similarities (correlation matrix)
    # (B, T, C) @ (B, C, T) -> (B, T, T)
    corr_matrix = torch.matmul(normed_hidden, normed_hidden.transpose(1, 2))
    
    eye_mask = torch.eye(T, device=hidden.device, dtype=torch.bool).unsqueeze(0) # (1,T,T)
    off_diag_mask = ~eye_mask
    
    if mask is not None: 
        token_pair_mask = mask.unsqueeze(2).bool() & mask.unsqueeze(1).bool() # (B,T,T), True if both tokens in pair are valid
        final_mask = off_diag_mask & token_pair_mask
    else:
        final_mask = off_diag_mask

    # Only consider elements where final_mask is True
    off_diag_corr_sq = (corr_matrix.masked_select(final_mask)**2).sum()
    
    num_off_diag_elements = final_mask.float().sum().clamp_min(1.0)
    
    return off_diag_corr_sq / num_off_diag_elements

def gate_entropy_regularizer(
    gates_list: List[torch.Tensor],  # Remove Optional
    padding_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    total_entropy = 0.0
    num_blocks = len(gates_list)
    
    for config_probs_tensor in gates_list:
        # Compute entropy directly (no None check needed)
        entropy_per_token = -(config_probs_tensor * torch.log(config_probs_tensor.clamp_min(1e-10))).sum(dim=-1)
        
        if padding_mask is not None:
            valid_tokens = padding_mask.sum()
            total_entropy += (entropy_per_token * padding_mask).sum() / valid_tokens
        else:
            total_entropy += entropy_per_token.mean()
    
    return total_entropy / num_blocks


class CompositeCriterion:
    def __init__(self, λ_H: float = 0.01, λ_C: float = 0.1, λ_O: float = 0.001, pad_idx: int = -100):
        self.λ_H, self.λ_C, self.λ_O = λ_H, λ_C, λ_O
        self.pad_idx = pad_idx

    def __call__(self, 
                 logits: torch.Tensor, 
                 targets: torch.Tensor, 
                 hidden: torch.Tensor, # Final hidden states for decorrelation
                 gates_list: List[Optional[torch.Tensor]], # List of config_probs from MFI blocks
                 padding_mask: Optional[torch.Tensor] = None # (B,T) boolean, True for valid tokens
                 ) -> tuple[torch.Tensor, Dict[str, float]]:
        
        loss_nll = nll_loss(logits, targets, pad_idx=self.pad_idx)
        
        # Regularizers use padding_mask to consider only valid tokens
        # Ensure padding_mask is boolean for logical ops, float for multiplication
        # The regularizer functions themselves handle casting mask to float if needed.
        bool_padding_mask = padding_mask.bool() if padding_mask is not None else None

        loss_H = entropy_regularizer(logits, mask=bool_padding_mask)
        loss_C = decorrelation_regularizer(hidden, mask=bool_padding_mask)
        
        # Pass padding_mask to gate_entropy_regularizer for token-wise gate entropy
        loss_O = gate_entropy_regularizer(gates_list, padding_mask=bool_padding_mask) 

        loss_tot = (loss_nll +
                    self.λ_H * loss_H +
                    self.λ_C * loss_C +
                    self.λ_O * loss_O)
        
        logs = {
            "nll":   loss_nll.item(), # .item() directly as these are scalar losses
            "H_logits": loss_H.item(),
            "decor_hidden": loss_C.item(),
            "gateH_mfi": loss_O.item(),
            "total": loss_tot.item() # Also log total loss for convenience
        }
        return loss_tot, logs