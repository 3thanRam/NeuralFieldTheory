# lossfunction.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeCriterion(nn.Module):
    # This __init__ now includes all the weights from your config
    def __init__(self, state_norm_weight=1.0, energy_conservation_weight=1.0, decorrelation_weight=1.0, 
                 reversibility_weight=1.0, jacobian_weight=0.1, 
                 momentum_consistency_weight=0.5, ignore_index=-100):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Store all weights
        self.state_norm_weight = state_norm_weight
        self.energy_conservation_weight = energy_conservation_weight
        self.decorrelation_weight = decorrelation_weight
        self.reversibility_weight = reversibility_weight
        self.jacobian_weight = jacobian_weight
        self.momentum_consistency_weight = momentum_consistency_weight

    # --- Assume all your helper compute_* functions exist here ---
    # e.g., compute_state_norm_loss, compute_energy_conservation_loss,
    # compute_decorrelation_loss, compute_jacobian_loss, compute_momentum_consistency_loss

    def compute_state_norm_loss(self, hamiltonian_internals):
        if not self.state_norm_weight > 0: return torch.tensor(0.0)
        Q_i, P_i, Q_f, P_f = hamiltonian_internals
        norm_initial = (Q_i.pow(2) + P_i.pow(2)).sum(dim=-1).mean()
        norm_final = (Q_f.pow(2) + P_f.pow(2)).sum(dim=-1).mean()
        return F.mse_loss(norm_final, norm_initial.detach())
        
    def compute_energy_conservation_loss(self, energy_internals):
        # ... (implementation from before) ...
        if not self.energy_conservation_weight > 0: return torch.tensor(0.0)
        energies_initial, energies_final = energy_internals
        if not energies_initial: return torch.tensor(0.0)
        total_loss = sum(F.mse_loss(h_f, h_i.detach()) for h_i, h_f in zip(energies_initial, energies_final))
        return total_loss / len(energies_initial)

    def compute_decorrelation_loss(self, hamiltonian_internals):
        if not self.decorrelation_weight > 0: return torch.tensor(0.0)
        _, P_i, _, P_f = hamiltonian_internals
        # Example: Penalize correlation between initial and final momentum
        P_i_flat = P_i.flatten(1)
        P_f_flat = P_f.flatten(1)
        corr = torch.abs(F.cosine_similarity(P_i_flat, P_f_flat, dim=-1)).mean()
        return corr # We want to minimize correlation, so this works as a loss

    def compute_jacobian_loss(self, jacobian_internals):
        if not self.jacobian_weight > 0: return torch.tensor(0.0)
        log_s1, log_s2 = jacobian_internals
        # The log-determinant of the Jacobian is the sum of the log scaling factors.
        # We want this to be close to zero for volume preservation.
        return (log_s1.abs().mean() + log_s2.abs().mean()) / 2

    def compute_momentum_consistency_loss(self, consistency_internals, targets):
        if not self.momentum_consistency_weight > 0: return torch.tensor(0.0)
        
        q_final_pred, p_final_from_inverse = consistency_internals
        
        # Create a 2D mask of shape [batch_size, seq_len]
        # This correctly selects the full embedding vectors where targets == pad_token_id
        pad_mask = (targets == self.cross_entropy_loss.ignore_index) 
        
        # Check if there are any padded elements to avoid division by zero if a batch has no padding
        if not pad_mask.any():
            return torch.tensor(0.0, device=p_final_from_inverse.device)

        # Index with the 2D mask. The result is a tensor of shape [num_padded_elements, embed_dim]
        momentum_at_pad = p_final_from_inverse[pad_mask]
        
        # Return the mean absolute value of these momentum components
        return momentum_at_pad.abs().mean()

    def forward(self, model_outputs, targets):
        (
            logits, _, hamiltonian_internals, jacobian_internals, 
            consistency_internals, reversibility_loss, energy_internals
        ) = model_outputs

        # 1. Main Language Modeling Loss
        ce_loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 2. Compute ALL auxiliary losses
        state_norm_loss = self.compute_state_norm_loss(hamiltonian_internals)
        energy_cons_loss = self.compute_energy_conservation_loss(energy_internals)
        decorr_loss = self.compute_decorrelation_loss(hamiltonian_internals)
        jac_loss = self.compute_jacobian_loss(jacobian_internals)
        mom_cons_loss = self.compute_momentum_consistency_loss(consistency_internals, targets)
        
        # 3. Combine them all with their weights
        total_loss = (
            ce_loss +
            self.state_norm_weight * state_norm_loss +
            self.energy_conservation_weight * energy_cons_loss +
            self.reversibility_weight * reversibility_loss +
            self.decorrelation_weight * decorr_loss +
            self.jacobian_weight * jac_loss +
            self.momentum_consistency_weight * mom_cons_loss
        )
        
        # 4. Create the complete dictionary for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'cross_entropy': ce_loss.item(),
            'state_norm': state_norm_loss.item(),
            'energy_conservation': energy_cons_loss.item(),
            'reversibility': reversibility_loss.item(),
            'decorrelation': decorr_loss.item(),
            'jacobian': jac_loss.item(),
            'mom_consistency': mom_cons_loss.item()
        }
        
        return total_loss, loss_components