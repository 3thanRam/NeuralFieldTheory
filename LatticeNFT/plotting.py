# plotting.py

import matplotlib.pyplot as plt
from collections import deque
import time

import matplotlib
matplotlib.use('Agg')

def plot_worker(queue, save_path='loss_plot.png', update_interval=50):
    """
    This worker saves a plot to a file using a symmetrical log scale for the y-axis,
    which can handle negative loss values.
    """
    history_len = 2000
    batch_iterations = deque(maxlen=history_len)
    losses = deque(maxlen=history_len)
    
    iteration = 0
    print("[Plotter] Worker started. Will save plot to file (using symlog scale).")

    while True:
        try:
            loss_value = queue.get()

            if loss_value is None:
                print("[Plotter] Received exit signal.")
                break

            batch_iterations.append(iteration)
            losses.append(loss_value)
            iteration += 1

            # --- Save plot to file periodically ---
            if iteration % update_interval == 0 and len(losses) > 1:
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(batch_iterations, losses)
                
                ax.set_title("Training Loss vs. Batch Iteration (Symlog Scale)")
                ax.set_xlabel("Batch Iteration")
                ax.set_ylabel("Loss (Contrastive)")
                
               
                ax.set_yscale('symlog', linthresh=1.0)
                # -----------------------------
                
                ax.grid(True, which='both') 
                
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

        except Exception as e:
            print(f"[Plotter] An error occurred: {e}")
            break
            
    # --- Create one final, high-quality plot ---
    if len(losses) > 1:
        print(f"[Plotter] Creating final plot at {save_path}")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(batch_iterations, losses)
        ax.set_title("Final Training Loss vs. Batch Iteration (Symlog Scale)")
        ax.set_xlabel("Batch Iteration")
        ax.set_ylabel("Loss (Contrastive)")
        
        ax.set_yscale('symlog', linthresh=1.0)
        
        ax.grid(True, which='both')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print("[Plotter] Final plot saved. Shutting down.")