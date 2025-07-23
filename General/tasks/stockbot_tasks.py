# tasks/stockbot_tasks.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from types import SimpleNamespace




def _run_stockbot_test(config, model, val_loader):
    """ Runs the autoregressive prediction test and plots the result. """
    
    full_sequence, _ = next(iter(val_loader))
    hist_len, pred_len = 40, 20
    if full_sequence.shape[1] < hist_len + pred_len:
        raise ValueError(f"Sequence from DataLoader is too short for test.")
    
    initial_sequence = full_sequence[0:1, :hist_len, :]
    ground_truth = full_sequence[0, hist_len:hist_len + pred_len, :]
    
    predictions = model.generate(initial_sequence, max_new_tokens=pred_len)
    
    os.makedirs(os.path.join(config.project_directory,"test_result"), exist_ok=True)
    _plot_candlestick_comparison(
        initial_sequence.squeeze(0).cpu().numpy(),
        ground_truth.cpu().numpy(),
        predictions.cpu().numpy(),
         os.path.join(config.project_directory,"test_result","stock_comparison_plot.png"),
        config
    )

def _plot_candlestick_comparison(history, ground_truth, predictions, save_path, config):
    """ Plots historical, ground truth, and predicted candlesticks for the primary symbol. """
    print(f"Plotting candlestick comparison and saving to {save_path}...")
    fig, ax = plt.subplots(figsize=(18, 9))
    hist_len = len(history)
    num_primary_features = 5

    def draw_candles(data, start_index, color_up, color_down):
        for i, ohlcv in enumerate(data):
            idx = start_index + i; o, h, l, c, _ = ohlcv
            color = color_up if c >= o else color_down
            ax.plot([idx, idx], [l, h], color=color, linewidth=1)
            ax.add_patch(Rectangle((idx - 0.4, min(o, c)), 0.8, abs(c - o), facecolor=color))

    draw_candles(history[:, :num_primary_features], 0, 'green', 'red')
    draw_candles(ground_truth[:, :num_primary_features], hist_len, 'green', 'red')
    draw_candles(predictions[:, :num_primary_features], hist_len, 'cyan', 'orange')

    handles = [Rectangle((0,0),1,1,facecolor=c) for c in ['green','red','cyan','orange']]
    labels = ['True Up','True Down','Pred Up','Pred Down']
    ax.legend(handles, labels)
    ax.axvline(x=hist_len - 0.5, color='k', linestyle='--')
    ax.set_title(f"Stock Price Prediction for {config.primary_symbol}: History vs. Ground Truth")
    ax.set_ylabel('Price ($)'); ax.set_xlabel('Time (Days)'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close(fig); print("Plot saved.")