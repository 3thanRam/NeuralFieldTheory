#tuning.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Assuming these are in the same directory or Python path
from config import myconfig
from lossfunction import CompositeCriterion
from training import get_training_texts, TextDataset
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # For exponential fitting
import warnings

car_return=_term_move_up() + '\r'

def moving_average(data, window_size):
    """Computes moving average."""
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Define the exponential decay function for fitting
def exp_decay_with_offset(t, A, k, C):
    """Exponential decay model: L(t) = A * exp(-k*t) + C"""
    t = np.array(t, dtype=float) # Ensure t is float for exp
    return A * np.exp(-k * t) + C

def test_config(x, args,
                plot_live=True, plot_window_size=10,
                plot_exp_fit=True, # Changed from plot_poly_fit
                exp_extrapolate_steps=30, min_points_for_exp_fit=20,max_batches_to_run = 101): # Min points might need to be higher for stable exp fit
    l_H,l_C,l_O=x
    system:myconfig=args
    system.lambda_H_logits = l_H
    system.lambda_C_hidden=l_C
    system.lambda_O_mfi=l_O
    system.criterion = CompositeCriterion(
        λ_H=system.lambda_H_logits,
        λ_C=system.lambda_C_hidden,
        λ_O=system.lambda_O_mfi,
        pad_idx=system.pad_idx
    )
    
    system.model.to(system.device)
    system.optimizer = optim.AdamW(system.model.parameters(), lr=system.lr)
    
    
    
    batch_iterator = tqdm(system.train_dataloader, 
                          total=max_batches_to_run,
                          desc=f"Cfg:({l_H:.1e},{l_C:.1e},{l_O:.1e})", 
                          unit="batch", 
                          leave=False,
                          position=0)

    batch_losses_for_this_run=[]
    avg_loss_for_this_run=0
    completed_batches=0

    fig, ax = None, None
    line_raw, line_smooth, line_exp_fit = None, None, None

    if plot_live:
        plt.ion() 
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_yscale('log') # Set y-axis to logarithmic scale

        line_raw, = ax.plot([], [], 'b-', alpha=0.4, label='Raw Batch Loss') 
        line_smooth, = ax.plot([], [], 'r-', linewidth=2, label=f'Smoothed (win {plot_window_size})')
        if plot_exp_fit:
            line_exp_fit, = ax.plot([], [], 'g--', linewidth=2, label=f'Exp. Fit (A*exp(-kt)+C, extr {exp_extrapolate_steps} steps)')
        
        ax.set_xlabel("Batch Step")
        ax.set_ylabel("Loss (Log Scale)")
        ax.set_title(f"Live Training Loss for Config: {(l_H, l_C, l_O)}")
        ax.legend(loc='upper right')
        ax.grid(True, which="both", ls="-", alpha=0.5) # Grid for log scale
        fig.canvas.manager.set_window_title(f"Loss Plot Config: {(l_H, l_C, l_O)}")

    for batch_idx, batch_data in enumerate(batch_iterator):
        if batch_idx >= max_batches_to_run:
            break

        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system.device), target_tensor.to(system.device)
        padding_mask = (input_tensor != system.pad_idx) 
        system.optimizer.zero_grad(set_to_none=True)
        
        logits, hidden_states, gates_list = system.model(
            input_tensor, 
            mask=padding_mask, 
            return_for_criterion=True
        )
        
        loss, loss_logs = system.criterion(
            logits, 
            target_tensor, 
            hidden_states, 
            gates_list, 
            padding_mask=padding_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.model.parameters(), 1.0)
        system.optimizer.step()
        
        current_loss_item = loss.item()
        # Ensure loss is positive for log scale and fitting
        # (though usually loss should be positive anyway)
        if current_loss_item <= 0:
            current_loss_item = 1e-9 # A very small positive number to avoid log(0) issues
            if completed_batches % 10 == 0: # Log warning occasionally
                 tqdm.write(f"{car_return}Warning: Loss was <=0, clamped to {current_loss_item:.1e} for plotting/fitting.")


        avg_loss_for_this_run += current_loss_item 
        completed_batches+=1
        batch_losses_for_this_run.append(current_loss_item)
        
        batch_iterator.set_postfix(
            loss=f"{current_loss_item:.3e}", # Use scientific notation for loss
            nll=f"{loss_logs['nll']:.3e}",
            H=f"{loss_logs['H_logits']:.3e}",
            decor=f"{loss_logs['decor_hidden']:.3e}",
            mfi=f"{loss_logs['gateH_mfi']:.3e}"
        )

        if plot_live and completed_batches > 0 and fig is not None:
            x_data_observed = np.arange(1, completed_batches + 1)
            y_data_observed = np.array(batch_losses_for_this_run) # Use numpy array for easier handling

            line_raw.set_xdata(x_data_observed)
            line_raw.set_ydata(y_data_observed)
            
            if completed_batches >= plot_window_size:
                smoothed_losses = moving_average(y_data_observed, plot_window_size)
                if smoothed_losses.size > 0:
                    x_data_smooth = np.arange(plot_window_size, completed_batches + 1)
                    line_smooth.set_xdata(x_data_smooth)
                    line_smooth.set_ydata(smoothed_losses)
                else:
                    line_smooth.set_xdata([])
                    line_smooth.set_ydata([])
            else:
                line_smooth.set_xdata([])
                line_smooth.set_ydata([])

            if plot_exp_fit and line_exp_fit is not None:
                if completed_batches >= min_points_for_exp_fit:
                    try:
                        # Initial guesses for A, k, C
                        # A: Initial drop (y[0] - y[-1])
                        # k: Small positive decay rate
                        # C: Final observed loss
                        initial_A = max(1e-6, y_data_observed[0] - y_data_observed[-1]) if len(y_data_observed)>1 else y_data_observed[0]
                        initial_k = 0.01 
                        initial_C = max(1e-9, y_data_observed[-1]) # Ensure C is positive
                        p0 = [initial_A, initial_k, initial_C]
                        
                        # Bounds for parameters (A>0, k>0, C>=0)
                        bounds = ([1e-9, 1e-9, 0], [np.inf, np.inf, np.inf])

                        with warnings.catch_warnings(): # Suppress OptimizeWarning if fit is not perfect
                            warnings.simplefilter("ignore", category=RuntimeWarning) # For overflows in exp
                            warnings.simplefilter("ignore", category=scipy.optimize.OptimizeWarning)
                            params, covariance = curve_fit(
                                exp_decay_with_offset, 
                                x_data_observed, 
                                y_data_observed,
                                p0=p0,
                                bounds=bounds,
                                maxfev=5000 # Increase max function evaluations
                            )
                        
                        A_fit, k_fit, C_fit = params
                        
                        x_exp_plot = np.arange(max(1, int(x_data_observed[0])), int(x_data_observed[-1]) + 1 + exp_extrapolate_steps)
                        y_exp_plot = exp_decay_with_offset(x_exp_plot, A_fit, k_fit, C_fit)
                        
                        # Ensure y_exp_plot is positive for log scale
                        y_exp_plot = np.maximum(y_exp_plot, 1e-9) 

                        line_exp_fit.set_xdata(x_exp_plot)
                        line_exp_fit.set_ydata(y_exp_plot)
                        
                        # Update legend with fitted C value
                        ax.legend(title=f"Fit: C ~ {C_fit:.2e}", loc='upper right')


                    except RuntimeError as e: # curve_fit couldn't find parameters
                        if completed_batches % 10 == 0:
                            tqdm.write(f"{car_return}Warning: Exp. fit failed (batch {completed_batches}, cfg {x}): {e}")
                        line_exp_fit.set_xdata([])
                        line_exp_fit.set_ydata([])
                    except Exception as e: # Other potential errors
                        if completed_batches % 10 == 0:
                            tqdm.write(f"{car_return}Warning: Error during Exp. fit (batch {completed_batches}, cfg {x}): {e}")
                        line_exp_fit.set_xdata([])
                        line_exp_fit.set_ydata([])
                else:
                    line_exp_fit.set_xdata([])
                    line_exp_fit.set_ydata([])
            
            ax.relim() 
            ax.autoscale_view(True,True,True) 
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        
    if completed_batches > 0 :
        avg_loss_for_this_run /= completed_batches
    else:
        avg_loss_for_this_run = float('nan')

    batch_iterator.close()
    
    tqdm.write(f"{car_return}Config {l_H:.1e},{l_C:.1e},{l_O:.1e}: Avg Loss = {avg_loss_for_this_run:.4e} (over {completed_batches} batches)")
    
    if plot_live and fig is not None:
        plt.ioff() 
        # Optional: Add final fitted parameters to title or save them
        # if plot_exp_fit and 'C_fit' in locals(): # Check if C_fit was determined
        #     ax.set_title(f"Final Loss for Config: {(l_H, l_C, l_O)}\nEst. C (asymptote) = {C_fit:.3e}")
        #     fig.canvas.draw() # Redraw with updated title
        # plt.savefig(f"loss_plot_config_exp_{l_H}_{l_C}_{l_O}.png")
        plt.show()

def test(num_points_per_lambda=5,num_bestruns=10,num_tests=2e2):
    
    system = myconfig(load=False, mode="train")
    if not hasattr(system, 'model') or system.model is None:
        raise AttributeError("system.model is not initialized.")

    train_texts, val_texts = get_training_texts(system)
    if not train_texts:
        print("No training data generated. Exiting.")
        return
    
    system.raw_training_texts = train_texts

    device=system.device
    num_workers = 0 
    pin_memory_flag = device.type == 'cuda'

    train_dataset=TextDataset(texts=system.raw_training_texts, tokenizer=system.tokenizer, max_len=system.max_seq_len)
    system.train_dataloader =DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
        
    system.model.to(system.device)

    max_b_run_in_test_config = 101
    extrap_steps = max(20, int(0.3 * max_b_run_in_test_config))

    test_config((1e-1,1e-1,1e-1), system, 
                plot_live=True, 
                plot_window_size=15,  # Slightly larger window for smoother data for exp fit
                plot_exp_fit=True, 
                exp_extrapolate_steps=extrap_steps, 
                min_points_for_exp_fit=25,max_batches_to_run = 250) # Exponential fit often needs more points
    
if __name__ == "__main__":
    test()