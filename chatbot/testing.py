# testing.py 
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader # Dataset is not directly used here
import math # For temperature decay

from config import myconfig
from lossfunction import CompositeCriterion # Assumes updated
from training import get_training_texts, TextDataset # For TextDataset
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

import subprocess
import cProfile
import pstats


car_return=_term_move_up() + '\r'
torch.autograd.set_detect_anomaly(True)

def moving_average(data, window_size):
    if len(data) < window_size: return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def exp_decay_with_offset(t, A, k, C):
    t = np.array(t, dtype=float)
    return A * np.exp(-k * t) + C

# The 'x' here are the fixed lambdas for this test run.
# 'args' is the myconfig object.
def test_config(x_lambdas, system_config: myconfig,
                plot_live=True, plot_window_size=10,
                plot_exp_fit=True,
                exp_extrapolate_steps=30, min_points_for_exp_fit=20,
                max_batches_to_run=101): # Renamed args to system_config

    l_H, l_C, l_O,l_E = x_lambdas # These are fixed for this visualization run
    
    # Apply the fixed lambdas for this test
    system_config.lambda_H_logits = l_H
    system_config.lambda_C_hidden = l_C
    system_config.lambda_O_mfi = l_O
    system_config.mfi_energy_reg_lambda = l_E
    # system_config.mfi_energy_reg_lambda can be set directly in myconfig before calling test()

    system_config.criterion = CompositeCriterion(
        λ_H=system_config.lambda_H_logits,
        λ_C=system_config.lambda_C_hidden,
        λ_O=system_config.lambda_O_mfi,
        λ_E_mfi=system_config.mfi_energy_reg_lambda, # Get from system_config
        pad_idx=system_config.pad_idx
    )

    # For testing.py, you might not want to reset parameters if you're continuing a training visualization.
    # If it's a fresh test of a config, then reset.
    # system_config.model.reset_parameters() # Optional: reset if it's a fresh test.
    system_config.model.to(system_config.device)
    system_config.optimizer = optim.AdamW(system_config.model.parameters(), lr=system_config.lr)

    batch_iterator = tqdm(system_config.train_dataloader,
                          total=max_batches_to_run,
                          desc=f"Plot Cfg:({l_H:.1e},{l_C:.1e},{l_O:.1e}, E:{system_config.mfi_energy_reg_lambda:.1e})",
                          unit="batch", leave=False, position=0)

    batch_losses_for_this_run=[]
    avg_loss_for_this_run=0
    completed_batches=0

    fig, ax = None, None
    line_raw, line_smooth, line_exp_fit = None, None, None

    if plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_yscale('log')
        line_raw, = ax.plot([], [], 'b-', alpha=0.4, label='Raw Batch Loss')
        line_smooth, = ax.plot([], [], 'r-', linewidth=2, label=f'Smoothed (win {plot_window_size})')
        if plot_exp_fit:
            line_exp_fit, = ax.plot([], [], 'g--', linewidth=2, label=f'Exp. Fit (extr {exp_extrapolate_steps} steps)')
        ax.set_xlabel("Batch Step")
        ax.set_ylabel("Loss (Log Scale)")
        ax.set_title(f"Live Loss: λH={l_H:.1e},λC={l_C:.1e},λO={l_O:.1e},λE={system_config.mfi_energy_reg_lambda:.1e}")
        ax.legend(loc='upper right')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        fig.canvas.manager.set_window_title(f"Loss Plot Config: {(l_H, l_C, l_O)}")


    for batch_idx, batch_data in enumerate(batch_iterator):
        if batch_idx >= max_batches_to_run:
            break

        # --- MFI Temperature and Sampling Mode Scheduling for this test run ---
        current_mfi_temperature_override = None
        if system_config.mfi_temperature_schedule_active:
            progress = min(1.0, batch_idx / max(1, max_batches_to_run - 1))
            current_mfi_temperature_override = system_config.mfi_initial_temperature - \
                                     (system_config.mfi_initial_temperature - system_config.mfi_final_temperature) * progress
            current_mfi_temperature_override = max(current_mfi_temperature_override, system_config.mfi_final_temperature)

        current_mfi_sampling_mode = "expectation"
        if system_config.mfi_sampling_schedule_active:
            # Example: sample for first 20% of batches in this test run
            sample_mode_until_batch = int(system_config.mfi_sample_mode_until_epoch * max_batches_to_run) \
                                      if system_config.mfi_sample_mode_until_epoch <=1 else system_config.mfi_sample_mode_until_epoch
            if batch_idx < sample_mode_until_batch:
                current_mfi_sampling_mode = "sample"
        # --- End Scheduling ---

        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system_config.device), target_tensor.to(system_config.device)
        padding_mask = (input_tensor != system_config.pad_idx)
        system_config.optimizer.zero_grad(set_to_none=True)

        logits, hidden_states, gates_list, mfi_energies_list = system_config.model(
            input_tensor,
            mask=padding_mask,
            return_for_criterion=True,
            temperature_override=current_mfi_temperature_override,
            sampling_mode=current_mfi_sampling_mode
        )

        loss, loss_logs = system_config.criterion(
            logits,
            target_tensor,
            hidden_states,
            gates_list,
            mfi_energies_list,
            padding_mask=padding_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system_config.model.parameters(), 1.0)
        system_config.optimizer.step()

        current_loss_item = loss.item()
        if current_loss_item <= 0: current_loss_item = 1e-9 # For log plot

        avg_loss_for_this_run += current_loss_item
        completed_batches+=1
        batch_losses_for_this_run.append(current_loss_item)

        postfix_dict = {
            "loss": f"{current_loss_item:.3e}", "nll": f"{loss_logs['nll']:.3e}",
            # Add other relevant components like E_mfi
        }
        if "E_mfi" in loss_logs: postfix_dict["Emfi"] = f"{loss_logs['E_mfi']:.3e}"
        if current_mfi_temperature_override is not None:
             postfix_dict["MFI_T"] = f"{current_mfi_temperature_override:.2f}"
        batch_iterator.set_postfix(postfix_dict)


        if plot_live and completed_batches > 0 and fig is not None:
            x_data_observed = np.arange(1, completed_batches + 1)
            y_data_observed = np.array(batch_losses_for_this_run)
            y_data_plot = np.maximum(y_data_observed, 1e-9) # Ensure positive for log plotting

            line_raw.set_xdata(x_data_observed)
            line_raw.set_ydata(y_data_plot)
            
            if completed_batches >= plot_window_size:
                smoothed_losses = moving_average(y_data_plot, plot_window_size)
                if smoothed_losses.size > 0:
                    x_data_smooth = np.arange(plot_window_size, completed_batches + 1)
                    line_smooth.set_xdata(x_data_smooth)
                    line_smooth.set_ydata(smoothed_losses)
            # ... (rest of plotting logic, including exp fit) ...
            # Ensure exp fit uses y_data_observed (original values) if they were offset for fitting,
            # or uses y_data_plot if that's what's intended.
            # The exp fit in tuning.py handles offsetting, a similar approach can be used here.
            # For simplicity here, assume y_data_observed is mostly positive.

            if plot_exp_fit and line_exp_fit is not None and completed_batches >= min_points_for_exp_fit:
                try:
                    # Simplified fit for plotting (tuning.py has more robust fitting)
                    y_to_fit = np.array(batch_losses_for_this_run) # use raw losses
                    min_loss_fit = y_to_fit.min()
                    offset_fit = 0
                    if min_loss_fit <= 1e-9:
                        offset_fit = abs(min_loss_fit) + 1e-6
                        y_to_fit = y_to_fit + offset_fit
                    
                    p0_fit = [max(1e-6, y_to_fit[0] - y_to_fit[-1]) if len(y_to_fit)>1 else y_to_fit[0], 0.01, max(1e-9, y_to_fit[-1])]
                    bounds_fit = ([1e-9, 1e-9, 0], [np.inf, np.inf, np.inf])
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        warnings.simplefilter("ignore", category=scipy.optimize.OptimizeWarning)
                        params_fit, _ = curve_fit(exp_decay_with_offset, x_data_observed, y_to_fit, p0=p0_fit, bounds=bounds_fit)
                    
                    A_f, k_f, C_f_raw = params_fit
                    C_f = C_f_raw - offset_fit # Adjust back
                    #C_f = max(0.0, C_f) # Ensure non-negative

                    x_exp_plot = np.arange(1, int(x_data_observed[-1]) + 1 + exp_extrapolate_steps)
                    y_exp_values = exp_decay_with_offset(x_exp_plot, A_f, k_f, C_f_raw) # Use C_f_raw for plotting the fitted curve
                    y_exp_plot = np.maximum(y_exp_values - offset_fit, 1e-9) # Adjust back and clamp for log plot

                    line_exp_fit.set_xdata(x_exp_plot)
                    line_exp_fit.set_ydata(y_exp_plot)
                    ax.legend(title=f"Fit: C ~ {C_f:.2e}", loc='upper right')
                except Exception as e:
                    if completed_batches % 10 == 0: tqdm.write(f"{car_return}Plot Exp. fit failed: {e}")
                    line_exp_fit.set_xdata([]); line_exp_fit.set_ydata([])
            
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            fig.canvas.flush_events()

    # ... (rest of test_config, closing plot, printing avg loss) ...
    batch_iterator.close()
    if completed_batches > 0: avg_loss_for_this_run /= completed_batches
    else: avg_loss_for_this_run = float('nan')
    tqdm.write(f"{car_return}Config ({l_H:.1e},{l_C:.1e},{l_O:.1e}, E:{system_config.mfi_energy_reg_lambda:.1e}): Avg Loss = {avg_loss_for_this_run:.4e} (over {completed_batches} batches)")

    if plot_live and fig is not None:
        plt.ioff()
        plt.show() # Keep plot window open


def test(max_b_run_in_test_config = 500): # Renamed from 'tuning' in the original testing.py
    system = myconfig(load=False, mode="train") # Ensure mode="train" for train_dataloader
    # Set MFI scheduling parameters in 'system' if you want to test them
    system.mfi_temperature_schedule_active = True
    system.mfi_initial_temperature = 2.0
    system.mfi_final_temperature = 0.5
    system.mfi_sampling_schedule_active = True
    system.mfi_sample_mode_until_epoch = 0.2 # Means 20% of batches in test_config

    system.mfi_energy_reg_lambda = 0.001 # Example, set this as desired

    if not hasattr(system, 'model') or system.model is None:
        raise AttributeError("system.model is not initialized.")

    train_texts, _ = get_training_texts(system)
    if not train_texts:
        print("No training data generated. Exiting.")
        return
    system.raw_training_texts = train_texts

    device=system.device
    num_workers = 0
    pin_memory_flag = device.type == 'cuda'

    train_dataset=TextDataset(texts=system.raw_training_texts, tokenizer=system.tokenizer, max_len=system.max_seq_len)
    system.train_dataloader =DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    system.set_scheduler(len(system.train_dataloader))
    system.model.to(system.device)
    # For testing.py, decide if you want to reset model params for each run of test()
    # system.model.reset_parameters()

    
    extrap_steps = max(20, int(0.3 * max_b_run_in_test_config))

    # Define the lambdas for this specific test run
    #fixed_lambdas_for_test = (5e-1, 1e-2, 2e-1,6e-1) # l_H, l_C, l_O
    fixed_lambdas_for_test = (0,0,0,0) 
    """
    Inspect the code with cProfile and snakeviz
    """
    #with cProfile.Profile() as pr:
    test_config(fixed_lambdas_for_test, system,
                plot_live=True,
                plot_window_size=15,
                plot_exp_fit=True,
                exp_extrapolate_steps=extrap_steps,
                min_points_for_exp_fit=25,
                max_batches_to_run=max_b_run_in_test_config)

    #    print('\n',"Finished Inspection")
    #    stats = pstats.Stats(pr).strip_dirs()
    #    stats.sort_stats(pstats.SortKey.TIME)
    #    stats.dump_stats(filename="statdump.prof")  # snakeviz ./statdump.prof
    #    subprocess.run(["snakeviz", "./statdump.prof"])
    

if __name__ == "__main__":
    test()