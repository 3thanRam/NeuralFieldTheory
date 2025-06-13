# highorderstats.py
import torch
from typing import Optional

def _expand_mask(mask: torch.Tensor, target_x_ndim: int) -> torch.Tensor:
    if mask is None:
        raise ValueError("_expand_mask called with None mask")
    ones_to_add = target_x_ndim - mask.ndim
    if ones_to_add < 0:
        raise ValueError(f"Mask ndim ({mask.ndim}) is greater than target_x_ndim ({target_x_ndim}).")
    if ones_to_add == 0:
        return mask.float()
    view_shape = mask.shape + (1,) * ones_to_add
    return mask.view(view_shape).float()

def skewness(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True, mask: Optional[torch.Tensor] = None, min_valid_points_for_skew: int = 3) -> torch.Tensor:
    if dim is None: 
        if mask is not None:
            num_valid = mask.float().sum()
            if num_valid < min_valid_points_for_skew: return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            x_masked = x[mask]
            if x_masked.numel() == 0: return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            mean_x, std_x = x_masked.mean(), x_masked.std(unbiased=unbiased).clamp_min(1e-9)
        else: 
            if x.numel() < min_valid_points_for_skew: return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            mean_x, std_x = x.mean(), x.std(unbiased=unbiased).clamp_min(1e-9)
        if std_x == 0: return torch.zeros_like(mean_x)
        source_tensor = x_masked if mask is not None and x_masked.numel() > 0 else x # Ensure x_masked is not empty before use
        if source_tensor.numel() == 0 : return torch.tensor(0.0, device=x.device, dtype=x.dtype) # If all masked out
        return torch.mean(((source_tensor - mean_x) / std_x)**3)
    else: 
        dim_tuple = (dim,) if not isinstance(dim, tuple) else dim
        
        # Determine output shape if all invalid upfront
        out_shape_list_for_zeros = list(x.shape)
        if keepdim:
            for d_val_idx in dim_tuple:
                out_shape_list_for_zeros[d_val_idx] = 1
        else:
            out_shape_list_for_zeros = [s for i, s in enumerate(x.shape) if i not in dim_tuple]
            if not out_shape_list_for_zeros: out_shape_list_for_zeros = [1] # for scalar output

        if mask is not None:
            num_valid_for_stats = mask.float().sum(dim=dim_tuple, keepdim=True)
            computation_valid_mask = (num_valid_for_stats >= min_valid_points_for_skew)
            if not computation_valid_mask.any():
                return torch.zeros(out_shape_list_for_zeros, device=x.device, dtype=x.dtype)

            num_valid_clamped = num_valid_for_stats.clamp_min(1.0)
            expanded_mask_for_x = _expand_mask(mask, x.ndim)
            x_times_mask = x * expanded_mask_for_x
            mean_x = x_times_mask.sum(dim=dim_tuple, keepdim=True) / num_valid_clamped
            mean_x_sq = (x_times_mask**2).sum(dim=dim_tuple, keepdim=True) / num_valid_clamped
            var_x_biased = (mean_x_sq - mean_x**2).clamp_min(0) # clamp_min(0) before sqrt
            var_x = var_x_biased * (num_valid_clamped / (num_valid_clamped - 1.0).clamp_min(1e-9)) if unbiased else var_x_biased
            std_x = torch.sqrt(var_x.clamp_min(0)).clamp_min(1e-9) # ensure var_x is non-negative
        else: 
            # Simplified check for non-masked case, assuming dim_tuple[0] is the primary reduction dim
            if x.shape[dim_tuple[0]] < min_valid_points_for_skew:
                computation_valid_mask = torch.zeros_like(x.mean(dim=dim_tuple, keepdim=True), dtype=torch.bool)
            else:
                computation_valid_mask = torch.ones_like(x.mean(dim=dim_tuple, keepdim=True), dtype=torch.bool)
            if not computation_valid_mask.any():
                return torch.zeros(out_shape_list_for_zeros, device=x.device, dtype=x.dtype)
            mean_x = torch.mean(x, dim=dim_tuple, keepdim=True)
            std_x = torch.std(x, dim=dim_tuple, keepdim=True, unbiased=unbiased).clamp_min(1e-9)

        zero_std_mask = (std_x == 0)
        std_x_safe = torch.where(zero_std_mask, torch.ones_like(std_x), std_x)
        skew_val_numerator = ((x - mean_x) / std_x_safe)**3
        if mask is not None:
            skew_val = (skew_val_numerator * expanded_mask_for_x).sum(dim=dim_tuple, keepdim=True) / num_valid_clamped
        else: 
            skew_val = torch.mean(skew_val_numerator, dim=dim_tuple, keepdim=True)
        
        final_valid_mask = computation_valid_mask & (~zero_std_mask)
        skew = torch.where(final_valid_mask, skew_val, torch.zeros_like(skew_val))
        if not keepdim:
            # Squeeze dimensions in reverse order of their original index to handle multiple reduced dims correctly
            for d_val in sorted(dim_tuple, reverse=True): 
                skew = skew.squeeze(d_val)
        return skew
        
def kurtosis(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True, excess: bool = True, mask: Optional[torch.Tensor] = None, min_valid_points_for_kurt: int = 4) -> torch.Tensor:
    kurt_zero_value = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    kurt_return_if_invalid = kurt_zero_value - (3.0 if excess else 0.0)

    if dim is None:
        num_elements = mask.float().sum() if mask is not None else x.numel()
        if num_elements < min_valid_points_for_kurt: return kurt_return_if_invalid # Will broadcast if x was multidim
        
        x_proc = x[mask] if mask is not None and x[mask].numel() > 0 else (x if mask is None else torch.empty(0, device=x.device, dtype=x.dtype))
        if x_proc.numel() < min_valid_points_for_kurt : return kurt_return_if_invalid

        mean_x, std_x = x_proc.mean(), x_proc.std(unbiased=unbiased).clamp_min(1e-9)
        if std_x == 0: kurt_val = torch.zeros_like(mean_x)
        else: kurt_val = torch.mean(((x_proc - mean_x) / std_x)**4)
    else:
        dim_tuple = (dim,) if not isinstance(dim, tuple) else dim
        
        out_shape_list_for_zeros = list(x.shape)
        if keepdim:
            for d_val_idx in dim_tuple:
                out_shape_list_for_zeros[d_val_idx] = 1
        else:
            out_shape_list_for_zeros = [s for i, s in enumerate(x.shape) if i not in dim_tuple]
            if not out_shape_list_for_zeros: out_shape_list_for_zeros = [1]
        
        kurt_return_if_invalid_shaped = torch.full(out_shape_list_for_zeros, 0.0, device=x.device, dtype=x.dtype) - (3.0 if excess else 0.0)


        if mask is not None:
            num_valid_for_stats = mask.float().sum(dim=dim_tuple, keepdim=True)
            computation_valid_mask = (num_valid_for_stats >= min_valid_points_for_kurt)
            if not computation_valid_mask.any(): return kurt_return_if_invalid_shaped
                
            num_valid_clamped = num_valid_for_stats.clamp_min(1.0); expanded_mask_for_x = _expand_mask(mask, x.ndim)
            x_times_mask = x * expanded_mask_for_x
            mean_x = x_times_mask.sum(dim=dim_tuple, keepdim=True) / num_valid_clamped
            var_x_biased = ((x_times_mask - mean_x)**2).sum(dim=dim_tuple, keepdim=True) / num_valid_clamped 
            var_x = var_x_biased * (num_valid_clamped / (num_valid_clamped - 1.0).clamp_min(1e-9)) if unbiased else var_x_biased
            std_x = torch.sqrt(var_x.clamp_min(0)).clamp_min(1e-9)
        else:
            if x.shape[dim_tuple[0]] < min_valid_points_for_kurt: computation_valid_mask = torch.zeros_like(x.mean(dim=dim_tuple, keepdim=True), dtype=torch.bool)
            else: computation_valid_mask = torch.ones_like(x.mean(dim=dim_tuple, keepdim=True), dtype=torch.bool)
            if not computation_valid_mask.any(): return kurt_return_if_invalid_shaped
                
            mean_x = torch.mean(x, dim=dim_tuple, keepdim=True)
            std_x = torch.std(x, dim=dim_tuple, keepdim=True, unbiased=unbiased).clamp_min(1e-9)

        zero_std_mask = (std_x == 0)
        std_x_safe = torch.where(zero_std_mask, torch.ones_like(std_x), std_x)
        kurt_val_numerator = ((x - mean_x) / std_x_safe)**4
        if mask is not None:
            kurt_val = (kurt_val_numerator * expanded_mask_for_x).sum(dim=dim_tuple, keepdim=True) / num_valid_clamped
        else:
            kurt_val = torch.mean(kurt_val_numerator, dim=dim_tuple, keepdim=True)
        
        final_valid_mask = computation_valid_mask & (~zero_std_mask)
        # If not valid, kurt_val should be 0 before subtracting excess
        kurt = torch.where(final_valid_mask, kurt_val, torch.zeros_like(kurt_val))
        
        if not keepdim:
            for d_val in sorted(dim_tuple, reverse=True): kurt = kurt.squeeze(d_val)
            
    return kurt - 3.0 if excess else kurt
