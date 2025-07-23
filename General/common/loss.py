# common/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Base (Task-Specific) Losses ---

class ChatbotBaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=kwargs.get('pad_idx'))
    def forward(self, predictions, targets, internals):
        return self.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))

class StockbotBaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.primary_symbol_idx = kwargs.get('primary_symbol_idx')
        self.huber_weight = kwargs.get('huber_weight', 0.7)
        self.direction_weight = kwargs.get('direction_weight', 0.3)
    def forward(self, predictions, targets, internals):
        start_idx = self.primary_symbol_idx * 5
        end_idx = start_idx + 5
        primary_preds = predictions[..., start_idx:end_idx]
        primary_targets = targets[..., start_idx:end_idx]
        mag_loss = F.smooth_l1_loss(primary_preds, primary_targets)
        dir_loss = torch.tensor(0., device=predictions.device)
        if predictions.size(1) > 1:
            pc = primary_preds[:, 1:, 4] - primary_preds[:, :-1, 4]
            tc = primary_targets[:, 1:, 4] - primary_targets[:, :-1, 4]
            dir_loss = 1 - F.cosine_similarity(pc, tc, dim=1, eps=1e-8).mean()
        return self.huber_weight * mag_loss + self.direction_weight * dir_loss

# --- Auxiliary (Model-Specific) Losses ---

class NormConstraintLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, predictions, targets, internals):
        final_q = internals.get('final_q')
        final_q_dot = internals.get('final_q_dot')
        if final_q is None: return torch.tensor(0.)
        q_norm = final_q.pow(2).mean()
        q_dot_norm = final_q_dot.pow(2).mean()
        return q_norm + q_dot_norm

class ForceMinimizationLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, predictions, targets, internals):
        forces_f = internals.get('forces_f', [])
        forces_g = internals.get('forces_g', [])
        if not forces_f: return torch.tensor(0.)
        f_loss = torch.mean(torch.stack([f.pow(2).mean() for f in forces_f]))
        g_loss = torch.mean(torch.stack([g.pow(2).mean() for g in forces_g]))
        return f_loss + g_loss

class ForceDecorrelationLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, predictions, targets, internals):
        forces_f = internals.get('forces_f', [])
        forces_g = internals.get('forces_g', [])
        if not forces_f: return torch.tensor(0.)
        total_similarity = 0.0
        for f, g in zip(forces_f, forces_g):
            f_flat = f.reshape(f.shape[0], -1)
            g_flat = g.reshape(g.shape[0], -1)
            total_similarity += F.cosine_similarity(f_flat, g_flat, dim=-1, eps=1e-8).abs().mean()
        return total_similarity / len(forces_f)


class RoundTripLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, predictions, targets, internals):
        return internals.get('round_trip_loss', torch.tensor(0.))

class EnergyRatioLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, predictions, targets, internals):
        final_q_dot = internals.get('final_q_dot')
        forces_f = internals.get('forces_f', [])
        if final_q_dot is None or not forces_f: return torch.tensor(0.)
        kinetic_energy = 0.5 * final_q_dot.pow(2).mean()
        potential_energy = forces_f[-1].pow(2).mean()
        return (kinetic_energy - potential_energy).pow(2)

# --- A simple factory to get loss instances by name ---
def get_loss_fn(name, **kwargs):
    if name == 'chatbot_base':
        return ChatbotBaseLoss(**kwargs)
    if name == 'stockbot_base':
        return StockbotBaseLoss(**kwargs)
    if name == 'norm_constraint':
        return NormConstraintLoss(**kwargs)
    if name == 'force_minimization':
        return ForceMinimizationLoss(**kwargs)
    if name == 'force_decorrelation':
        return ForceDecorrelationLoss(**kwargs)
    if name == 'round_trip':
        return RoundTripLoss(**kwargs)
    if name == 'energy_ratio':
        return EnergyRatioLoss(**kwargs)
    raise ValueError(f"Unknown loss function: {name}")