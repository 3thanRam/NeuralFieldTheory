# testing.py

import os
import torch
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from config import config
from data_handling import prepare_dataset_from_api # We only need this one

def plot_candlesticks(ax, data, start_index=0, color_real='darkgreen', color_pred='lime', is_prediction=False):
    """Helper function to plot candlestick data."""
    for i, bar in enumerate(data):
        # Unpack OHLC, bar might have other elements like time
        o, c, h, l = bar[1], bar[2], bar[3], bar[4]
        
        idx = start_index + i
        color = 'g' if c >= o else 'r'
        
        # Line for high-low
        ax.plot([idx, idx], [l, h], color=color, linewidth=1)
        # Rectangle for open-close
        body_height = abs(c - o)
        if body_height < 0.001: body_height = 0.001 # Make flat bars visible
        rect = plt.Rectangle((idx - 0.4, min(o, c)), 0.8, body_height, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)


def testing(model):
    device = torch.device(config["device"])
    model.to(device)
    model.eval()
    print("Generating a test sample for plotting...")

    # Generate test data - using same method as training
    test_X, test_Y = prepare_dataset_from_api(
        symbols=config["symbols"],
        primary_symbol=config["primary_symbol"],
        years_of_data=1  # Only need 1 year for testing
    )
    
    # Take the most recent complete sequence
    test_sample_X = torch.tensor(test_X[-1:], dtype=torch.float32).to(device)
    ground_truth = test_Y[-1]  # Corresponding target sequence
    
    # Generate predictions - match prediction length to ground truth
    pred_len = ground_truth.shape[0]  # Use full sequence length
    with torch.no_grad():
        predictions = model.generate(test_sample_X, n_to_pred=pred_len)
    
    # Verify shapes
    assert predictions.shape[0] == ground_truth.shape[0], \
        f"Prediction length {predictions.shape[0]} != ground truth {ground_truth.shape[0]}"
    
    # Prepare plot data
    seq_len = config["sequence_length"]
    total_points = seq_len + pred_len
    
    # Create time indices
    history_indices = np.arange(seq_len)
    pred_indices = np.arange(seq_len, seq_len + pred_len)
    
    # Extract close prices (index 1 in OHLC)
    historical_close = test_sample_X[0, :, config["symbols"].index(config["primary_symbol"])*4 + 1].cpu()
    predicted_close = predictions[:, 1]  # Close prices from predictions
    actual_future_close = ground_truth[:, 1]  # Close prices from ground truth
    
    # --- Plotting ---
    plt.figure(figsize=(15, 7))
    
    # Plot historical data
    plt.plot(history_indices, historical_close, 'b-', label='Historical Prices')
    
    # Plot actual future
    plt.plot(pred_indices, actual_future_close, 'g-', label='Actual Future')
    
    # Plot predictions
    plt.plot(pred_indices, predicted_close, 'r--', label='Predicted Future')
    
    plt.axvline(x=seq_len - 0.5, color='k', linestyle='--', label='Prediction Start')
    plt.title(f"{config['primary_symbol']} Price Prediction")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(config["model_save_path"]), "test_results.png")
    plt.savefig(plot_path)
    print(f"Test plot saved to {plot_path}")
    plt.close()