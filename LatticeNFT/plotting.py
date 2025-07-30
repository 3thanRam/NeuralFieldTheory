# In plotter.py

import matplotlib.pyplot as plt
from collections import deque
import time

# Use a non-interactive backend for saving files on a server
import matplotlib
matplotlib.use('Agg')

def plot_worker(queue, save_path='loss_plot.png', update_interval=50):
    """
    This worker saves a plot to a file instead of displaying it live.
    Perfect for remote servers.
    """
    history_len = 2000 # Can store more history now
    batch_iterations = deque(maxlen=history_len)
    losses = deque(maxlen=history_len)
    
    iteration = 0
    print("[Plotter] Worker started. Will save plot to file.")

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
            if iteration % update_interval == 0:
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(batch_iterations, losses)
                
                ax.set_title("Training Loss vs. Batch Iteration")
                ax.set_xlabel("Batch Iteration")
                ax.set_ylabel("Loss (Contrastive)")
                ax.grid(True)
                
                # Save the figure. `bbox_inches='tight'` cleans up whitespace.
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig) # IMPORTANT: Close the figure to free up memory

        except Exception as e:
            print(f"[Plotter] An error occurred: {e}")
            break
            
    # Create one final, high-quality plot at the very end
    print(f"[Plotter] Creating final plot at {save_path}")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(batch_iterations, losses)
    ax.set_title("Final Training Loss vs. Batch Iteration")
    ax.set_xlabel("Batch Iteration")
    ax.set_ylabel("Loss (Contrastive)")
    ax.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print("[Plotter] Final plot saved. Shutting down.")