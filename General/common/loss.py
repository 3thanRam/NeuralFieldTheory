# common/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChatbotBaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=kwargs.get('pad_idx'))
    def forward(self, predictions, targets, internals):
        return self.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))

class StockbotBaseLoss(nn.Module):
    def __init__(self, primary_symbol_idx, **kwargs):
        super().__init__()
        self.primary_symbol_idx = primary_symbol_idx
        self.huber_weight = kwargs.get('huber_weight', 0.7)
        self.direction_weight = kwargs.get('direction_weight', 0.3)

    def forward(self, predictions_norm, targets_norm, internals):
        # Both predictions and targets are RETURNS (~1.0). Loss is stable.
        start_idx = self.primary_symbol_idx * 5
        end_idx = start_idx + 5
        
        primary_preds = predictions_norm[..., start_idx:end_idx]
        primary_targets = targets_norm[..., start_idx:end_idx]

        # Magnitude loss on the returns
        magnitude_loss = F.smooth_l1_loss(primary_preds, primary_targets)
        
        # Directional loss now compares if returns are >1 or <1
        dir_loss = torch.tensor(0., device=predictions_norm.device)
        if predictions_norm.size(1) > 1:
            # Tanh centers the returns around 0. (1.01 -> small pos, 0.99 -> small neg)
            pred_direction = torch.tanh((primary_preds[:, 1:, 3] - 1) * 10) # Index 3 is Close return
            true_direction = torch.tanh((primary_targets[:, 1:, 3] - 1) * 10)
            dir_loss = 1 - F.cosine_similarity(pred_direction, true_direction, dim=1, eps=1e-8).mean()
            
        return self.huber_weight * magnitude_loss + self.direction_weight * dir_loss

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

class CandleShapeLoss(nn.Module):
    """
    Penalizes the model for predicting unrealistic candlestick shapes.
    It compares the volatility (range) and body size of the predicted
    candles to the ground truth.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.primary_symbol_idx = kwargs.get('primary_symbol_idx')

    def forward(self, predictions, targets, internals):
        # Select the primary symbol's data
        start_idx = self.primary_symbol_idx * 5
        # We only need OHLC for this loss
        end_idx = start_idx + 4
        
        preds = predictions[..., start_idx:end_idx]
        truth = targets[..., start_idx:end_idx]

        # O, H, L, C indices are 0, 1, 2, 3 in this slice
        # Predicted range and body
        pred_range = preds[..., 1] - preds[..., 2] # High - Low
        pred_body = (preds[..., 0] - preds[..., 3]).abs() # |Open - Close|

        # True range and body
        true_range = truth[..., 1] - truth[..., 2]
        true_body = (truth[..., 0] - truth[..., 3]).abs()

        # Calculate the Mean Squared Error for both shape properties
        range_loss = F.mse_loss(pred_range, true_range)
        body_loss = F.mse_loss(pred_body, true_body)

        return range_loss + body_loss
    
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
    if name == 'candle_shape':
        return CandleShapeLoss(**kwargs)
    raise ValueError(f"Unknown loss function: {name}")