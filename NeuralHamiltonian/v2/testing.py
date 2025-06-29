# testing.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import config
from data_handling import prepare_dataset_from_api

def plot_candlesticks(ax, data, start_index, label, color_up, color_down, alpha=1.0):
    """
    Helper function to plot candlestick data on a given axes.
    
    Args:
        ax: The matplotlib axes to plot on.
        data: A numpy array of shape (N, 4) with columns (o, h, l, c).
        start_index: The starting x-coordinate for this data series.
        label: The label for the legend.
        color_up: The color for a day where close >= open.
        color_down: The color for a day where close < open.
        alpha: The transparency of the candles.
    """
    # Create a proxy artist for the legend
    proxy_patch = mpatches.Patch(color=color_up, label=label, alpha=alpha)

    for i, ohlc in enumerate(data):
        # The OHLC order from the model is (o, c, h, l). Let's be explicit.
        # But our `prepare_dataset_from_api` returns (o, c, h, l), and the model predicts 4 values.
        # Let's assume the order is [Open, Close, High, Low] for clarity in plotting.
        # Note: If your data is ordered differently, adjust here. Let's assume prediction output is (O, C, H, L).
        # Based on your previous code, prediction[:, 1] was Close. Let's assume [O, C, H, L] for now.
        # It's crucial this matches your data structure.
        # The provided code has close as index 1, so let's stick to that: O=0, C=1, H=2, L=3
        # BUT this is not standard OHLC. Standard is O,H,L,C.
        # Let's assume your data is ordered [Open, High, Low, Close] for standard candlestick plotting.
        # We'll need to re-order the prediction output if it's different.
        # The `generate` method returns (1, 1, 4) -> (O, C, H, L) based on the loss function.
        # So we'll map: o=pred[0], c=pred[1], h=pred[2], l=pred[3]
        
        o, c, h, l = ohlc[0], ohlc[1], ohlc[2], ohlc[3]
        
        idx = start_index + i
        color = color_up if c >= o else color_down
        
        # Plot the high-low wick
        ax.plot([idx, idx], [l, h], color=color, linewidth=1, alpha=alpha)
        
        # Plot the open-close body
        body_height = abs(c - o)
        if body_height == 0:
            body_height = 0.01 # Make flat bars visible
        rect = plt.Rectangle(
            (idx - 0.4, min(o, c)), 0.8, body_height, 
            facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha
        )
        ax.add_patch(rect)
        
    return proxy_patch


def testing(model1, model2, model1_name="Hamiltonian", model2_name="Transformer"):
    """
    Tests two models side-by-side and plots their predictions as candlestick charts.
    """
    device = torch.device(config["device"])
    model1.to(device).eval()
    model2.to(device).eval()
    print(f"Generating a comparison test sample for {model1_name} and {model2_name}...")

    # Generate a common test dataset
    test_X, test_Y = prepare_dataset_from_api(
        symbols=config["symbols"],
        primary_symbol=config["primary_symbol"],
        years_of_data=2
    )
    
    test_sample_X = torch.tensor(test_X[-1:], dtype=torch.float32).to(device)
    ground_truth_ohlc = test_Y[-1]
    
    pred_len = ground_truth_ohlc.shape[0]

    # --- Generate full OHLC predictions from both models ---
    with torch.no_grad():
        predictions1 = model1.generate(test_sample_X, n_to_pred=pred_len)
        predictions2 = model2.generate(test_sample_X, n_to_pred=pred_len)
    
    # --- Prepare plot data ---
    seq_len = config["sequence_length"]
    
    # Extract the historical OHLC for the primary symbol from the input
    primary_idx = config["symbols"].index(config["primary_symbol"])
    start_col, end_col = primary_idx * 4, primary_idx * 4 + 4
    historical_ohlc = test_sample_X[0, :, start_col:end_col].cpu().numpy()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(18, 9))
    legend_handles = []

    # Plot Historical Data (Solid)
    handle = plot_candlesticks(ax, historical_ohlc, 0, 'Historical', 'darkgray', 'lightgray', alpha=0.9)
    legend_handles.append(handle)

    # Plot Actual Future Data (Solid, Green/Red)
    handle = plot_candlesticks(ax, ground_truth_ohlc, seq_len, 'Actual Future', 'green', 'red', alpha=1.0)
    legend_handles.append(handle)

    # Plot Prediction 1 (Transparent, Orange/Blue)
    handle = plot_candlesticks(ax, predictions1, seq_len, f'Predicted ({model1_name})', 'darkorange', 'deepskyblue', alpha=0.7)
    legend_handles.append(handle)
    
    # Plot Prediction 2 (Transparent, Purple/Yellow)
    handle = plot_candlesticks(ax, predictions2, seq_len, f'Predicted ({model2_name})', 'purple', 'gold', alpha=0.7)
    legend_handles.append(handle)
    
    # --- Final Touches ---
    ax.axvline(x=seq_len - 0.5, color='k', linestyle='--', label='Prediction Start')
    legend_handles.append(ax.get_lines()[-1]) # Add the axvline to the legend
    
    ax.set_title(f"{config['primary_symbol']} Candlestick Prediction Comparison", fontsize=16)
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(handles=legend_handles, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = config["comparison_plot_path"]
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison candlestick plot saved to {plot_path}")
    plt.close()