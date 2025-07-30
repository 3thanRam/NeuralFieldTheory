# tasks/stockbot_tasks.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from types import SimpleNamespace




def _run_stockbot_test(config, model, val_loader):
    """ Runs the autoregressive prediction test and plots the result. """
    # Get a batch of raw input data for testing
    full_sequence_raw, _ = next(iter(val_loader))
    
    hist_len = config.test_history_length
    pred_len = config.test_prediction_length
    
    if full_sequence_raw.shape[1] < hist_len + pred_len:
        raise ValueError(
            f"DataLoader sequence_length ({full_sequence_raw.shape[1]}) is too short for the test. "
            f"It must be at least test_history_length + test_prediction_length ({hist_len + pred_len})."
        )
    
    # --- Correctly slice the RAW data for history, ground truth, and model input ---
    # We take the first sample from the batch for our test case
    history_and_truth_period_raw = full_sequence_raw[0, :hist_len + pred_len, :]

    # This is the historical data that will be plotted
    history_for_plot = history_and_truth_period_raw[:hist_len, :]
    
    # This is the ground truth that we will compare our predictions against
    ground_truth = history_and_truth_period_raw[hist_len:, :]
    
    # The input to the model's generate function is the most recent `sequence_length` points of history
    initial_sequence_for_model = history_for_plot[None, -config.sequence_length:, :]
    
    # Generate predictions. The model.generate method now correctly handles the
    # internal normalization and outputs raw-scale price predictions.
    predictions = model.generate(initial_sequence_for_model, max_new_tokens=pred_len)
    
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{config.task}_comparison_plot.png")
    
    _plot_candlestick_comparison(
        history_for_plot.cpu().numpy(),
        ground_truth.cpu().numpy(),
        predictions.cpu().numpy(),
        save_path,
        config
    )

def _plot_candlestick_comparison(history, ground_truth, predictions, save_path, config):
    """ Plots historical, ground truth, and predicted candlesticks for the primary symbol. """
    print(f"Plotting candlestick comparison and saving to {save_path}...")
    fig, ax = plt.subplots(figsize=(24, 12))
    hist_len = len(history)
    num_primary_features = 5

    def draw_candles(data, start_index, color_up, color_down):
        for i, ohlcv in enumerate(data):
            idx = start_index + i
            # Correct unpacking for OHLCV format
            o, h, l, c, v = ohlcv
            color = color_up if c >= o else color_down
            ax.plot([idx, idx], [l, h], color=color, linewidth=1)
            ax.add_patch(Rectangle((idx - 0.4, min(o, c)), 0.8, abs(c - o), facecolor=color, edgecolor='black', linewidth=0.5))

    # For plotting, we only care about the primary symbol's data, which is the first 5 features
    draw_candles(history[:, :num_primary_features], 0, 'green', 'red')
    draw_candles(ground_truth[:, :num_primary_features], hist_len, 'green', 'red')
    draw_candles(predictions[:, :num_primary_features], hist_len, 'cyan', 'orange')

    handles = [
        Rectangle((0,0), 1, 1, facecolor='green', label='True Up'),
        Rectangle((0,0), 1, 1, facecolor='red', label='True Down'),
        Rectangle((0,0), 1, 1, facecolor='cyan', label='Pred Up'),
        Rectangle((0,0), 1, 1, facecolor='orange', label='Pred Down')
    ]
    ax.legend(handles=handles)
    ax.axvline(x=hist_len - 0.5, color='k', linestyle='--', linewidth=1.5)
    ax.set_title(f"Stock Price Prediction for {config.primary_symbol}: Truth vs Prediction", fontsize=16)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print("Plot saved successfully.")