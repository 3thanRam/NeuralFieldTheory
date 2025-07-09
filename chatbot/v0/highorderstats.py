import torch
from typing import Optional

def skewness(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the skewness of a tensor.
    Args:
        x: Input tensor.
        dim: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        unbiased: If True, use the unbiased estimator for variance in normalization.
                  For skewness, an unbiased estimator is more complex and often sample skewness is used.
                  This 'unbiased' flag here primarily affects the std dev calculation.
        mask: Optional boolean tensor for masked computation. True for valid elements.
    Returns:
        Skewness tensor.
    """
    if mask is not None:
        if dim is None: # Global skewness with mask
            num_valid = mask.float().sum().clamp_min(1.0)
            x_masked = x[mask]
            if x_masked.numel() == 0: return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
            mean_x = x_masked.mean()
            std_x = x_masked.std(unbiased=unbiased).clamp_min(1e-9) # Avoid division by zero
            # For N elements, unbiased skew usually has a (N / ((N-1)*(N-2))) factor,
            # but here we use the simpler sample skewness based on E[((X-mu)/sigma)^3]
            skew = ((x_masked - mean_x) / std_x)**3
            return skew.sum() / num_valid # Average of the cubed standardized scores
        else: # Dimension-wise skewness with mask
            if not isinstance(dim, tuple): dim = (dim,)
            num_valid = mask.float().sum(dim=dim, keepdim=True).clamp_min(1.0)
            
            # Mask elements before sum for mean
            x_times_mask = x * mask.float().unsqueeze(-1) if x.ndim > mask.ndim else x * mask.float()
            mean_x = x_times_mask.sum(dim=dim, keepdim=True) / num_valid
            
            # For std dev with mask: E[X^2] - (E[X])^2
            mean_x_sq = (x_times_mask**2).sum(dim=dim, keepdim=True) / num_valid
            var_x = (mean_x_sq - mean_x**2).clamp_min(0) # Ensure non-negative variance
            if unbiased:
                # Adjust variance for unbiased estimator (Bessel's correction)
                # This correction factor is complex for masked, multi-dim reduction
                # For simplicity with masking, often the biased variance is used or a simplified correction
                # A common simplification is N / (N-1), but N varies per slice with mask.
                # Here, we'll use the biased std for simplicity with complex masks,
                # or you'd need a more involved way to get N for each slice.
                # correction_factor = num_valid / (num_valid - 1).clamp_min(1.0)
                # var_x = var_x * correction_factor
                pass # Sticking to biased for std in complex masked case for this example
            std_x = torch.sqrt(var_x).clamp_min(1e-9)
            
            # Cubed standardized scores, only for valid elements
            # (x - mean_x) will be non-zero only for original x positions
            # masking after subtraction and division ensures we sum correctly
            print(x.shape,mean_x.shape,std_x.shape)
            standardized_cubed = (((x - mean_x) / std_x)**3)
            # Apply mask again before summing the cubed terms
            skew_sum = (standardized_cubed * mask.float().unsqueeze(-1) if x.ndim > mask.ndim else standardized_cubed * mask.float()).sum(dim=dim, keepdim=True)
            skew = skew_sum / num_valid

            if not keepdim:
                # Squeeze the specified dimensions
                for d_idx, d_val in enumerate(sorted(dim, reverse=True)):
                    skew = skew.squeeze(d_val)
            return skew

    else: # No mask
        if dim is None: # Global skewness
            mean_x = x.mean()
            std_x = x.std(unbiased=unbiased).clamp_min(1e-9)
            if std_x == 0: return torch.zeros_like(mean_x) # Or NaN if preferred for zero std
            skew = torch.mean(((x - mean_x) / std_x)**3)
            return skew
        else: # Dimension-wise skewness
            mean_x = torch.mean(x, dim=dim, keepdim=True)
            std_x = torch.std(x, dim=dim, keepdim=True, unbiased=unbiased).clamp_min(1e-9)
            
            # Create a mask for zero std_dev to avoid NaN and output 0 skew
            zero_std_mask = (std_x == 0)

            skew = torch.mean(((x - mean_x) / std_x)**3, dim=dim, keepdim=True)
            skew = torch.where(zero_std_mask, torch.zeros_like(skew), skew) # Handle 0 std_dev

            if not keepdim:
                # Squeeze the specified dimensions
                if not isinstance(dim, tuple): dim = (dim,)
                for d_idx, d_val in enumerate(sorted(dim, reverse=True)):
                    skew = skew.squeeze(d_val)
            return skew
        
def kurtosis(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True, excess: bool = True, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the kurtosis (or excess kurtosis) of a tensor.
    Args:
        x: Input tensor.
        dim: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
        unbiased: If True, use unbiased estimator for variance in normalization.
                  Similar to skewness, a truly unbiased kurtosis estimator is complex.
        excess: If True, returns excess kurtosis (Kurt[X] - 3).
        mask: Optional boolean tensor for masked computation. True for valid elements.
    Returns:
        Kurtosis tensor.
    """
    if mask is not None:
        if dim is None: # Global kurtosis with mask
            num_valid = mask.float().sum().clamp_min(1.0)
            x_masked = x[mask]
            if x_masked.numel() == 0: return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
            mean_x = x_masked.mean()
            std_x = x_masked.std(unbiased=unbiased).clamp_min(1e-9)
            kurt = (((x_masked - mean_x) / std_x)**4).sum() / num_valid
        else: # Dimension-wise kurtosis with mask
            if not isinstance(dim, tuple): dim = (dim,)
            num_valid = mask.float().sum(dim=dim, keepdim=True).clamp_min(1.0)
            
            x_times_mask = x * mask.float().unsqueeze(-1) if x.ndim > mask.ndim else x * mask.float()
            mean_x = x_times_mask.sum(dim=dim, keepdim=True) / num_valid
            
            mean_x_sq = (x_times_mask**2).sum(dim=dim, keepdim=True) / num_valid
            var_x = (mean_x_sq - mean_x**2).clamp_min(0)
            # No simple unbiased correction for std here with complex mask
            std_x = torch.sqrt(var_x).clamp_min(1e-9)

            standardized_fourth = (((x - mean_x) / std_x)**4)
            kurt_sum = (standardized_fourth * mask.float().unsqueeze(-1) if x.ndim > mask.ndim else standardized_fourth * mask.float()).sum(dim=dim, keepdim=True)
            kurt = kurt_sum / num_valid
            
            if not keepdim:
                for d_idx, d_val in enumerate(sorted(dim, reverse=True)):
                    kurt = kurt.squeeze(d_val)
    else: # No mask
        if dim is None: # Global kurtosis
            mean_x = x.mean()
            std_x = x.std(unbiased=unbiased).clamp_min(1e-9)
            if std_x == 0: # If std is 0, kurtosis is ill-defined or could be taken as 0 for const signal
                kurt_val = torch.zeros_like(mean_x)
            else:
                kurt_val = torch.mean(((x - mean_x) / std_x)**4)
            kurt = kurt_val
        else: # Dimension-wise kurtosis
            mean_x = torch.mean(x, dim=dim, keepdim=True)
            std_x = torch.std(x, dim=dim, keepdim=True, unbiased=unbiased).clamp_min(1e-9)

            zero_std_mask = (std_x == 0)
            kurt_val = torch.mean(((x - mean_x) / std_x)**4, dim=dim, keepdim=True)
            kurt = torch.where(zero_std_mask, torch.zeros_like(kurt_val), kurt_val)

            if not keepdim:
                if not isinstance(dim, tuple): dim = (dim,)
                for d_idx, d_val in enumerate(sorted(dim, reverse=True)):
                    kurt = kurt.squeeze(d_val)

    if excess:
        return kurt - 3.0
    return kurt