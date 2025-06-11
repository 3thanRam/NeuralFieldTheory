# tuning.py
import numpy as np
import torch
import torch.optim as optim
from itertools import product
from torch.utils.data import Dataset, DataLoader
import math # For temperature decay

from config import myconfig
from lossfunction import CompositeCriterion # Assumes CompositeCriterion is updated
from training import get_training_texts,TextDataset # Keep TextDataset if used for dataloading
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up

import scipy
from scipy.optimize import minimize
from scipy.optimize import curve_fit # For exponential fitting
import warnings

car_return=_term_move_up() + '\r'
torch.autograd.set_detect_anomaly(True)
# Define the exponential decay function for fitting
def exp_decay_with_offset(t, A, k, C):
    """Exponential decay model: L(t) = A * exp(-k*t) + C"""
    t = np.array(t, dtype=float) # Ensure t is float for exp
    return A * np.exp(-k * t) + C

def test_config_fit(x, args): # x are lambdas, args is myconfig
    system:myconfig = args
    l_H,l_C,l_O, l_E_mfi=x # These are the parameters being optimized by minimize
    
    # Apply the lambdas being tested for this run
    system.lambda_H_logits = l_H
    system.lambda_C_hidden = l_C
    system.lambda_O_mfi = l_O
    system.mfi_energy_reg_lambda=l_E_mfi
    # Potentially, if mfi_energy_reg_lambda is also part of 'x', apply it here too.
    # For now, assume it's fixed in the 'args' (myconfig) or you add it to 'x'.
    # e.g., if x = (l_H, l_C, l_O, l_E_mfi), then system.mfi_energy_reg_lambda = l_E_mfi

    system.criterion = CompositeCriterion(
        λ_H=system.lambda_H_logits,
        λ_C=system.lambda_C_hidden,
        λ_O=system.lambda_O_mfi,
        λ_E_mfi=system.mfi_energy_reg_lambda, # Use the one from system config
        pad_idx=system.pad_idx
    )

    # Reset model parameters for each new configuration test for a fairer comparison.
    # This might be slow if model is large, but essential for tuning.
    # If not resetting, the model state from previous lambda tests will influence current test.
    system.model.reset_parameters() # Add reset_parameters method to your OverallLanguageModel
    system.model.to(system.device)

    system.optimizer = optim.AdamW(system.model.parameters(), lr=system.lr)
    # system.scheduler is not typically used for short tuning runs, but can be if desired.

    total_batches_for_this_tuning_run = int(system.num_tests) # num_tests is N batches

    batch_iterator = tqdm(system.train_dataloader,
                          total=total_batches_for_this_tuning_run,
                          desc=f"TuneFit Cfg:({l_H:.2f},{l_C:.2f},{l_O:.2f},{l_E_mfi:.2f})",
                          unit="batch", leave=False, position=0)
    batch_losses_for_this_run=[]

    for batch_idx, batch_data in enumerate(batch_iterator):
        if batch_idx >= total_batches_for_this_tuning_run:
            break

        # --- MFI Temperature and Sampling Mode Scheduling for this tuning run ---
        current_mfi_temperature_override = None
        if system.mfi_temperature_schedule_active:
            # Progress within this short tuning run (0 to 1)
            progress = min(1.0, batch_idx / max(1, total_batches_for_this_tuning_run - 1))
            current_mfi_temperature_override = system.mfi_initial_temperature - \
                                     (system.mfi_initial_temperature - system.mfi_final_temperature) * progress
            current_mfi_temperature_override = max(current_mfi_temperature_override, system.mfi_final_temperature)

        current_mfi_sampling_mode = "expectation"
        if system.mfi_sampling_schedule_active:
            # Here, mfi_sample_mode_until_epoch is interpreted as "until_batch_ratio"
            # e.g., if mfi_sample_mode_until_epoch is 0.3 (meaning 30% of epochs)
            # then for tuning, it means 30% of total_batches_for_this_tuning_run
            # This requires careful thought on how mfi_sample_mode_until_epoch is defined.
            # Simpler: use a fixed fraction of batches for 'sample' mode during tuning.
            sample_mode_until_batch = int(system.mfi_sample_mode_until_epoch * total_batches_for_this_tuning_run) \
                                      if system.mfi_sample_mode_until_epoch <=1 else system.mfi_sample_mode_until_epoch
            if batch_idx < sample_mode_until_batch : # Example: sample for first N batches or first X% of batches
                current_mfi_sampling_mode = "sample"
        # --- End Scheduling for tuning run ---

        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system.device), target_tensor.to(system.device)
        padding_mask = (input_tensor != system.pad_idx)
        system.optimizer.zero_grad(set_to_none=True)

        logits, hidden_states, gates_list, mfi_energies_list = system.model(
            input_tensor,
            mask=padding_mask,
            return_for_criterion=True,
            temperature_override=current_mfi_temperature_override,
            sampling_mode=current_mfi_sampling_mode
        )

        loss, loss_logs = system.criterion(
            logits,
            target_tensor,
            hidden_states,
            gates_list,
            mfi_energies_list,
            padding_mask=padding_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.model.parameters(), 1.0)
        system.optimizer.step()
        batch_losses_for_this_run.append(loss.item())

        postfix_dict = {
            "total": f"{loss.item():.3f}", "nll": f"{loss_logs['nll']:.3f}",
            # Add other relevant loss components if needed
        }
        if current_mfi_temperature_override is not None:
            postfix_dict["MFI_T"] = f"{current_mfi_temperature_override:.2f}"
        batch_iterator.set_postfix(postfix_dict)

    batch_iterator.close()
    # --- Exponential Fit ---
    if not batch_losses_for_this_run: # Handle case of no batches run
        print(f"{car_return}Cfg{x}: No batches run, returning inf.")
        return float('inf')

    x_data_observed = np.arange(1, len(batch_losses_for_this_run) + 1)
    y_data_observed = np.array(batch_losses_for_this_run)

    # Ensure y_data_observed values are positive for stable fitting, especially if loss can be negative.
    # Add a small constant if min is <= 0, then subtract later if needed, or ensure loss is always > 0.
    min_loss = y_data_observed.min()
    offset_for_fitting = 0
    if min_loss <= 1e-9: # If any loss is non-positive or too small
        offset_for_fitting = abs(min_loss) + 1e-6 # Add a bit more than the min
        y_data_observed = y_data_observed + offset_for_fitting


    initial_A = max(1e-6, y_data_observed[0] - y_data_observed[-1]) if len(y_data_observed) > 1 else y_data_observed[0]
    initial_k = 0.01
    initial_C = max(1e-9, y_data_observed[-1]) # Ensure C is positive (relative to potentially offsetted y_data)
    p0 = [initial_A, initial_k, initial_C]
    bounds = ([1e-9, 1e-9, 0], [np.inf, np.inf, np.inf]) # C must be >= 0

    C_fit_raw = float('inf') # This will be relative to the offsetted y_data
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=scipy.optimize.OptimizeWarning)
            params, covariance = curve_fit(
                exp_decay_with_offset,
                x_data_observed,
                y_data_observed,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
                ftol=1e-6, xtol=1e-6 # Add tolerance for convergence
            )
            A_fit, k_fit, C_fit_raw = params
    except RuntimeError: # If curve_fit fails
        print(f"{car_return}Cfg{x}: Exp fit failed. Using last observed loss or avg if C_fit_raw is bad.")
        # Fallback: use average of last few losses or just the last loss
        if len(y_data_observed) > 5:
            C_fit_raw = np.mean(y_data_observed[-5:])
        elif len(y_data_observed) > 0:
            C_fit_raw = y_data_observed[-1]
        else: # Should be caught by earlier check
            C_fit_raw = float('inf')


    # Adjust C_fit back if an offset was used for fitting
    C_fit_final = C_fit_raw - offset_for_fitting
    
    # Ensure C_fit is not unreasonably low (e.g., negative after offset removal)
    # The "true" loss should not be negative.
    C_fit_final = max(0.0, C_fit_final)


    print(f"{car_return}Cfg{x}: C_raw={C_fit_raw:.4e}, C_final={C_fit_final:.4e} (offset={offset_for_fitting:.2e})")
    return C_fit_final


# test_config_scalar can be modified similarly to test_config_fit
# Key changes:
# 1. Reset model parameters: system.model.reset_parameters()
# 2. Implement MFI scheduling logic inside the batch loop based on batch_idx/total_batches.
# 3. Pass temperature_override and sampling_mode to system.model(...).
# 4. Update criterion instantiation and call.

def test_config_scalar(x,args):
    l_H,l_C,l_O, l_E_mfi=x # Or (l_H,l_C,l_O, l_E_mfi) if tuning l_E_mfi
    system:myconfig=args
    system.lambda_H_logits = l_H
    system.lambda_C_hidden=l_C
    system.lambda_O_mfi=l_O
    system.mfi_energy_reg_lambda = l_E_mfi # If tuning

    system.criterion = CompositeCriterion(
        λ_H=system.lambda_H_logits,
        λ_C=system.lambda_C_hidden,
        λ_O=system.lambda_O_mfi,
        λ_E_mfi=system.mfi_energy_reg_lambda,
        pad_idx=system.pad_idx
    )
    
    system.model.reset_parameters() # Crucial for fair comparison
    system.model.to(system.device)
    system.optimizer = optim.AdamW(system.model.parameters(), lr=system.lr)
    
    total_batches_for_this_tuning_run = int(system.num_tests)
    batch_iterator = tqdm(system.train_dataloader, total=total_batches_for_this_tuning_run, desc=f"TuneScalar Cfg:({l_H:.2f},{l_C:.2f},{l_O:.2f})", unit="batch", leave=False, position=0)
    
    avg_loss_for_this_run=0
    completed_batches=0
    for batch_idx, batch_data in enumerate(batch_iterator):
        if batch_idx >= total_batches_for_this_tuning_run:
            break
        
        # --- MFI Scheduling for this run ---
        current_mfi_temperature_override = None
        if system.mfi_temperature_schedule_active:
            progress = min(1.0, batch_idx / max(1, total_batches_for_this_tuning_run - 1))
            current_mfi_temperature_override = system.mfi_initial_temperature - \
                                     (system.mfi_initial_temperature - system.mfi_final_temperature) * progress
            current_mfi_temperature_override = max(current_mfi_temperature_override, system.mfi_final_temperature)

        current_mfi_sampling_mode = "expectation"
        if system.mfi_sampling_schedule_active:
            sample_mode_until_batch = int(system.mfi_sample_mode_until_epoch * total_batches_for_this_tuning_run) \
                                      if system.mfi_sample_mode_until_epoch <=1 else system.mfi_sample_mode_until_epoch
            if batch_idx < sample_mode_until_batch:
                current_mfi_sampling_mode = "sample"
        # --- End Scheduling ---

        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system.device), target_tensor.to(system.device)
        padding_mask = (input_tensor != system.pad_idx) 
        system.optimizer.zero_grad(set_to_none=True)
        
        logits, hidden_states, gates_list, mfi_energies_list = system.model(
            input_tensor, 
            mask=padding_mask, 
            return_for_criterion=True,
            temperature_override=current_mfi_temperature_override,
            sampling_mode=current_mfi_sampling_mode
        )
        loss, loss_logs = system.criterion(
            logits, 
            target_tensor, 
            hidden_states, 
            gates_list, 
            mfi_energies_list,
            padding_mask=padding_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.model.parameters(), 1.0)
        system.optimizer.step()
        avg_loss_for_this_run += loss.item() 
        completed_batches+=1
        
        postfix_dict = {"total": f"{loss.item():.3f}", "nll": f"{loss_logs['nll']:.3f}"}
        if current_mfi_temperature_override is not None:
             postfix_dict["MFI_T"] = f"{current_mfi_temperature_override:.2f}"
        batch_iterator.set_postfix(postfix_dict)
    
    batch_iterator.close()
    if completed_batches == 0:
        print(f"{car_return}Cfg{x}: No batches run, returning inf.")
        return float('inf')
        
    avg_loss_for_this_run /= completed_batches
    print(f"{car_return}Cfg{x}: AvgLoss={avg_loss_for_this_run:.4e}")
    return avg_loss_for_this_run


def tuning(num_points_per_lambda=5, num_bestruns=10, num_tests=150): # Reduced num_tests for faster demo
    
    # It's good practice for 'tuning' to create its own config for isolation,
    # or ensure the passed 'system' object is correctly configured for tuning.
    system = myconfig(load=False, mode="train") # Ensure mode="train" if using train_dataloader
    system.num_tests = num_tests # num_tests is number of batches for each trial

    # Add OverallLanguageModel.reset_parameters() method if it doesn't exist:
    # class OverallLanguageModel(nn.Module):
    #     def reset_parameters(self):
    #         # Call reset_parameters on all submodules like embeddings, blocks, linear layers
    #         self.token_embedding.reset_parameters()
    #         for block in self.blocks:
    #             block.reset_parameters() # Ensure blocks have this method
    #         self.final_norm.reset_parameters()
    #         self.to_logits.reset_parameters()
    #         # Also reset PartitionFunctionMFI's custom parameters if they are not reset by block.reset_parameters()
    #         # PartitionFunctionMFI should also have reset_parameters to reset its log_beta and config_prior

    # Ensure blocks and MFI module have reset_parameters
    # class PartitionFunctionMFIBlock(nn.Module):
    #     def reset_parameters(self):
    #         self.mfi.reset_parameters()
    #         self.norm1.reset_parameters()
    #         # Reset MLP layers
    #         for layer in self.mlp:
    #             if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
    #         self.norm2.reset_parameters()

    # class PartitionFunctionMFI(nn.Module):
    #     def reset_parameters(self):
    #         # ... reset standard layers ...
    #         with torch.no_grad():
    #             self.log_beta.data.copy_(self._initial_log_beta) # Assuming you store initial values
    #             self.config_prior.data.copy_(self._initial_config_prior)


    train_texts, _ = get_training_texts(system) # We only need training data for tuning runs
    if not train_texts:
        print("No training data for tuning. Exiting.")
        return
    system.raw_training_texts = train_texts # Use only training data for these short runs

    device = system.device
    num_workers = 0
    pin_memory_flag = device.type == 'cuda'

    train_dataset = TextDataset(texts=system.raw_training_texts, tokenizer=system.tokenizer, max_len=system.max_seq_len)
    # Use a persistent DataLoader if num_workers > 0 and many calls to test_config_fit
    # For tuning, shuffle=False might give more stable loss curves for fitting, but True is also fine.
    system.train_dataloader = DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    system.set_scheduler(len(system.train_dataloader))
    # Define search space for lambdas
    # Example: x = [lambda_H, lambda_C, lambda_O, lambda_E_mfi]
    # initial_guess = [0.1, 0.1, 0.1, 0.001] # Add l_E_mfi if tuning it
    # bounds = [(0, 1), (0, 1), (0, 1), (0, 0.1)] # Add bounds for l_E_mfi

    initial_guess = [0.1, 0.1, 0.1,0.1]
    bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)] # Lambdas should generally be > 0

    print("Starting optimization with scipy.minimize (using L-BFGS-B by default with bounds)...")
    res = minimize(test_config_fit, initial_guess, args=(system,),
                 bounds=bounds, method='L-BFGS-B', # or 'Nelder-Mead' if bounds are problematic, or 'Powell'
                 options={'disp': True, 'maxiter': 50, 'eps': 1e-3}) # maxiter, eps control optimizer
    
    print("\nOptimization finished.")
    print("Best parameters found by minimize (e.g., [l_H, l_C, l_O,l_E]):")
    print(res.x)
    print(f"Corresponding minimum value (e.g., fitted C): {res.fun}")

    # The grid search part is commented out as minimize is now used.
    # If you want to use grid search, you'd iterate and call test_config_fit or test_config_scalar.

if __name__ == "__main__":
    tuning()