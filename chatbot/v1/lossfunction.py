# lossfunction.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# ... (compute_nll_loss, compute_conservation_loss, compute_decorrelation_loss, compute_reversibility_loss, compute_jacobian_loss are unchanged) ...
def compute_nll_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)
def compute_conservation_loss(Q_i: torch.Tensor, P_i: torch.Tensor, Q_f: torch.Tensor, P_f: torch.Tensor) -> torch.Tensor:
    N_initial = (Q_i.pow(2) + P_i.pow(2)).sum(dim=-1).mean(); N_final = (Q_f.pow(2) + P_f.pow(2)).sum(dim=-1).mean()
    return F.mse_loss(N_final, N_initial)
def compute_decorrelation_loss(hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    B, T, C = hidden_states.shape
    if T <= 1: return torch.tensor(0.0, device=hidden_states.device)
    normed_hidden = F.normalize(hidden_states, p=2, dim=2)
    token_pair_mask = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
    eye_mask = ~torch.eye(T, device=hidden_states.device, dtype=torch.bool).unsqueeze(0)
    final_mask = token_pair_mask & eye_mask
    corr_matrices = torch.bmm(normed_hidden, normed_hidden.transpose(1, 2))
    total_corr_sq = corr_matrices.masked_select(final_mask).pow(2).sum()
    num_valid_pairs = final_mask.float().sum().clamp_min(1.0)
    return total_corr_sq / num_valid_pairs
def compute_reversibility_loss(model: nn.Module, ham_internals: Tuple) -> torch.Tensor:
    Q_i, P_i, Q_f, P_f = ham_internals; Q_rev, P_rev = Q_f, P_f
    for block in reversed(model.blocks): Q_rev, P_rev = model.leapfrog_update(Q_rev, P_rev, block, timestep=-model.timestep)
    loss_q = F.mse_loss(Q_rev, Q_i); loss_p = F.mse_loss(P_rev, P_i)
    return loss_q + loss_p
def compute_jacobian_loss(jac_internals: Tuple, padding_mask: torch.Tensor) -> torch.Tensor:
    log_s1, log_s2 = jac_internals; mask = padding_mask.unsqueeze(-1).float(); num_valid = padding_mask.sum().clamp_min(1.0)
    loss_s1 = (log_s1 * mask).abs().sum(); loss_s2 = (log_s2 * mask).abs().sum()
    return (loss_s1 + loss_s2) / num_valid

# --- NEW LOSS FUNCTION ---
def compute_momentum_consistency_loss(model: nn.Module, consistency_internals: Tuple, padding_mask: torch.Tensor) -> torch.Tensor:
    """Enforces that the momentum_net's mapping is consistent with the dynamics."""
    q_final, p_final = consistency_internals

    # Calculate the velocity of the final state
    dq_final = torch.zeros_like(q_final)
    dq_final[:, 1:, :] = q_final[:, 1:, :] - q_final[:, :-1, :]

    # Re-compute momentum from the final state
    momentum_net_input = torch.cat([q_final, dq_final], dim=-1)
    p_recomputed = model.momentum_net(momentum_net_input)

    # We only care about the error on non-padded tokens
    mask = padding_mask.unsqueeze(-1).float()
    
    # Calculate masked MSE
    error = (p_recomputed - p_final).pow(2) * mask
    loss = error.sum() / mask.sum().clamp_min(1.0)
    
    return loss

# In lossfunction.py

class CompositeCriterion:
    def __init__(self, conservation_weight: float = 0.01, decorrelation_weight: float = 0.1, reversibility_weight: float = 0.01, jacobian_weight: float = 0.01, momentum_consistency_weight: float = 0.01, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self.λ_conserve = conservation_weight
        self.λ_decorr = decorrelation_weight
        self.λ_reverse = reversibility_weight
        self.λ_jacobian = jacobian_weight
        self.λ_mom_const = momentum_consistency_weight

    def __call__(self, model: nn.Module, logits: torch.Tensor, targets: torch.Tensor, hidden_state: torch.Tensor, ham_internals: Tuple, jac_internals: Tuple, consistency_internals: Tuple, reversibility_loss_term: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # This part is fine
        loss_nll = compute_nll_loss(logits, targets, self.ignore_index)
        
        # Initialize all regularizers to zero tensors on the correct device
        # This is good practice
        device = logits.device
        loss_conserve = torch.tensor(0.0, device=device)
        loss_decorr = torch.tensor(0.0, device=device)
        loss_reverse = reversibility_loss_term # This is already a tensor from the model
        loss_jacobian = torch.tensor(0.0, device=device)
        loss_mom_const = torch.tensor(0.0, device=device)
        
        # Calculate the individual loss values if their weight is non-zero
        if self.λ_conserve > 0:
            loss_conserve = compute_conservation_loss(*ham_internals)
        if self.λ_decorr > 0:
            loss_decorr = compute_decorrelation_loss(hidden_state, padding_mask)
        if self.λ_jacobian > 0:
            loss_jacobian = compute_jacobian_loss(jac_internals, padding_mask)
        if self.λ_mom_const > 0:
            loss_mom_const = compute_momentum_consistency_loss(model, consistency_internals, padding_mask)
        
        # --- THIS IS THE FIX ---
        # Combine all losses using out-of-place addition to create a new computation graph
        total_loss = (loss_nll +
                      self.λ_conserve * loss_conserve +
                      self.λ_decorr * loss_decorr +
                      self.λ_reverse * loss_reverse +
                      self.λ_jacobian * loss_jacobian +
                      self.λ_mom_const * loss_mom_const)
        # --- END OF FIX ---
            
        logs = {
            "nll": loss_nll.item(),
            "conservation": loss_conserve.item(),
            "decorrelation": loss_decorr.item(),
            "reversibility": loss_reverse.item(),
            "jacobian": loss_jacobian.item(),
            "mom_const": loss_mom_const.item(),
            "total": total_loss.item()
        }
        
        return total_loss, logs