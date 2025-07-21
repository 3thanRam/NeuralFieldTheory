# common/lossfunction.py
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
        self.primary_symbol_idx = primary_symbol_idx; self.huber_weight=huber_weight; self.direction_weight=direction_weight
    def forward(self, predictions, targets):
        start_idx = self.primary_symbol_idx*5; end_idx = start_idx+5
        primary_preds = predictions[..., start_idx:end_idx]; primary_targets = targets[..., start_idx:end_idx]
        magnitude_loss = F.smooth_l1_loss(primary_preds, primary_targets)
        directional_loss = torch.tensor(0., device=predictions.device)
        if predictions.size(1) > 1:
            pred_changes = primary_preds[:, 1:, 4] - primary_preds[:, :-1, 4]
            true_changes = primary_targets[:, 1:, 4] - primary_targets[:, :-1, 4]
            directional_loss = 1 - F.cosine_similarity(pred_changes, true_changes, dim=1, eps=1e-8).mean()
        return self.huber_weight * magnitude_loss + self.direction_weight * directional_loss

class CompositeLoss(nn.Module):
    def __init__(self, base_loss_fn, aux_weights=None):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.aux_weights = aux_weights if aux_weights is not None else {}

    # --- NEW, ARCHITECTURE-SPECIFIC LOSS FUNCTIONS ---
    
    def _final_state_norm_loss(self, final_q, final_q_dot):
        # Penalizes the magnitude of the final evolved state
        if final_q is None: return torch.tensor(0.)
        q_norm = final_q.pow(2).mean()
        q_dot_norm = final_q_dot.pow(2).mean()
        return q_norm + q_dot_norm

    def _force_minimization_loss(self, forces_f, forces_g):
        # Encourages "gentle" dynamics by penalizing large forces
        if not forces_f: return torch.tensor(0.)
        f_loss = torch.mean(torch.stack([f.pow(2).mean() for f in forces_f]))
        g_loss = torch.mean(torch.stack([g.pow(2).mean() for g in forces_g]))
        return f_loss + g_loss

    def _force_decorrelation_loss(self, forces_f, forces_g):
        # Encourages the two parallel streams to learn different functions
        if not forces_f: return torch.tensor(0.)
        total_similarity = 0.0
        for f, g in zip(forces_f, forces_g):
            f_flat = f.reshape(f.shape[0], -1)
            g_flat = g.reshape(g.shape[0], -1)
            total_similarity += F.cosine_similarity(f_flat, g_flat, dim=-1).abs().mean()
        return total_similarity / len(forces_f)

    def forward(self, model_outputs, targets):
        predictions, internals = model_outputs
        base_loss = self.base_loss_fn(predictions, targets)
        
        # Get the new internals
        final_q = internals.get('final_q')
        final_q_dot = internals.get('final_q_dot')
        forces_f = internals.get('forces_f', [])
        forces_g = internals.get('forces_g', [])
        
        # Calculate the new auxiliary losses
        norm_loss = self._final_state_norm_loss(final_q, final_q_dot)
        force_loss = self._force_minimization_loss(forces_f, forces_g)
        decorr_loss = self._force_decorrelation_loss(forces_f, forces_g)
        
        total_loss = (base_loss +
                      self.aux_weights.get('norm_constraint', 0) * norm_loss +
                      self.aux_weights.get('force_minimization', 0) * force_loss +
                      self.aux_weights.get('force_decorrelation', 0) * decorr_loss)
                      
        components = {
            'total': total_loss.item(),
            'base': base_loss.item(),
            'norm': norm_loss.item(),
            'force': force_loss.item(),
            'decorr': decorr_loss.item(),
        }
        return total_loss, components