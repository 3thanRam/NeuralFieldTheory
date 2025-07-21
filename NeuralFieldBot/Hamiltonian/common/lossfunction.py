# common/lossfunction.py - Defines the loss structures.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChatbotBaseLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions, targets):
        return self.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))

class StockbotBaseLoss(nn.Module):
    def __init__(self, primary_symbol_idx, huber_weight=0.7, direction_weight=0.3):
        super().__init__()
        self.primary_symbol_idx = primary_symbol_idx
        self.huber_weight = huber_weight
        self.direction_weight = direction_weight

    def forward(self, predictions, targets):
        start_idx = self.primary_symbol_idx * 5
        end_idx = start_idx + 5
        
        primary_preds = predictions[..., start_idx:end_idx]
        primary_targets = targets[..., start_idx:end_idx]

        magnitude_loss = F.smooth_l1_loss(primary_preds, primary_targets)
        directional_loss = torch.tensor(0., device=predictions.device)
        
        if predictions.size(1) > 1:
            pred_changes = primary_preds[:, 1:, 4] - primary_preds[:, :-1, 4]
            true_changes = primary_targets[:, 1:, 4] - primary_targets[:, :-1, 4]
            
            # --- THE FIX: Add a small epsilon for numerical stability ---
            directional_loss = 1 - F.cosine_similarity(
                pred_changes,
                true_changes,
                dim=1,
                eps=1e-8  # This prevents division by zero
            ).mean()
            
        return self.huber_weight * magnitude_loss + self.direction_weight * directional_loss

class CompositeLoss(nn.Module):
    def __init__(self, base_loss_fn, aux_weights=None):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.aux_weights = aux_weights if aux_weights is not None else {}

    def _state_norm_loss(self, internals):
        Q_i, P_i, Q_f, P_f = internals['hamiltonian']
        norm_initial = (Q_i.pow(2) + P_i.pow(2)).sum(-1).mean()
        norm_final = (Q_f.pow(2) + P_f.pow(2)).sum(-1).mean()
        return F.mse_loss(norm_final, norm_initial.detach())

    def _energy_conservation_loss(self, internals):
        energies_initial, energies_final = internals['energy']
        if not energies_initial:
            return torch.tensor(0.)
        loss = sum(F.mse_loss(h_f, h_i.detach()) for h_i, h_f in zip(energies_initial, energies_final))
        return loss / len(energies_initial)

    def forward(self, model_outputs, targets):
        predictions, internals = model_outputs
        base_loss = self.base_loss_fn(predictions, targets)
        state_loss = self._state_norm_loss(internals)
        energy_loss = self._energy_conservation_loss(internals)
        reversibility_loss = internals['reversibility_loss']
        
        total_loss = (base_loss +
                      self.aux_weights.get('state_norm', 0) * state_loss +
                      self.aux_weights.get('energy_conservation', 0) * energy_loss +
                      self.aux_weights.get('reversibility', 0) * reversibility_loss)
                      
        components = {
            'total': total_loss.item(),
            'base': base_loss.item(),
            'state': state_loss.item(),
            'energy': energy_loss.item(),
            'rev': reversibility_loss.item()
        }
        return total_loss, components