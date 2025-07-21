# stockbot/tasks.py
import torch, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common.config import StockbotConfig
from common.trainer import train_model
from .data import prepare_stock_loaders

def run_training(args):
    config = StockbotConfig(args)
    train_loader, val_loader = prepare_stock_loaders(config)
    model, criterion = config.get_model_and_criterion()
    train_model(config, model, criterion, train_loader, val_loader)

def run_testing(args):
    config = StockbotConfig(args)
    if not os.path.exists(config.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {config.ckpt_path}.")
    
    _, val_loader = prepare_stock_loaders(config)
    model, _ = config.get_model_and_criterion()
    
    checkpoint = torch.load(config.ckpt_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device).eval()
    print("--- Model loaded for testing. ---")

    full_sequence, _ = next(iter(val_loader))
    hist_len, pred_len = 40, 20
    if full_sequence.shape[1] < hist_len + pred_len:
        raise ValueError("Sequence from DataLoader is too short for this test.")

    initial_sequence = full_sequence[0:1, :hist_len, :]
    ground_truth = full_sequence[0, hist_len:hist_len + pred_len, :]
    
    # Call the new unified generate method
    predictions = model.generate(initial_sequence, max_new_tokens=pred_len)
    
    os.makedirs("test_results", exist_ok=True)
    _plot_candlestick_comparison(
        initial_sequence.squeeze(0).cpu().numpy(),
        ground_truth.cpu().numpy(),
        predictions.cpu().numpy(),
        os.path.join(config.outputdir, "stock_comparison_plot.png")
    )

def _autoregressive_predict(model, initial_sequence, num_steps, device):
    model.eval()
    predicted_steps = []
    current_sequence = initial_sequence.clone().to(device)
    print(f"Generating {num_steps} future predictions...")

    for _ in range(num_steps):
        with torch.enable_grad():
            # Model predicts the next state for ALL features
            predictions, _ = model(current_sequence, return_internals=True)
        
        # Take the last predicted vector (all features)
        next_step_pred = predictions[:, -1:, :] # Shape [1, 1, num_features]
        
        # Store the full prediction
        predicted_steps.append(next_step_pred.detach().cpu())
        
        # The update is a simple concatenation because the model's output
        # shape now matches its input shape.
        current_sequence = torch.cat([current_sequence[:, 1:, :], next_step_pred], dim=1)
        
    return torch.cat(predicted_steps, dim=1).squeeze(0)

def _plot_candlestick_comparison(history, ground_truth, predictions, save_path):
    print(f"Plotting comparison and saving to {save_path}...")
    fig, ax = plt.subplots(figsize=(18, 9))
    hist_len = len(history)

    def draw_candles(data, start_index, color_up, color_down):
        for i, ohlcv in enumerate(data):
            idx = start_index + i
            # OHLCV -> O, H, L, C is indices 0, 1, 2, 4 in Alpaca data
            # Assuming your data format is O, H, L, C, V...
            o, h, l, c = ohlcv[0], ohlcv[1], ohlcv[2], ohlcv[3]
            color = color_up if c >= o else color_down
            ax.plot([idx, idx], [l, h], color=color, linewidth=1)
            ax.add_patch(Rectangle((idx - 0.4, min(o, c)), 0.8, abs(c - o), facecolor=color, edgecolor='none'))

    draw_candles(history[:, :5], 0, 'gray', 'black') # Plot primary symbol history
    draw_candles(ground_truth[:, :5], hist_len, 'lime', 'green') # Plot primary symbol ground truth
    draw_candles(predictions[:, :5], hist_len, 'cyan', 'blue') # Plot primary symbol predictions

    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ['gray', 'black', 'lime', 'green', 'cyan', 'blue']]
    labels = ['Hist Up', 'Hist Down', 'True Up', 'True Down', 'Pred Up', 'Pred Down']
    ax.legend(handles, labels)
    ax.axvline(x=hist_len - 0.5, color='k', linestyle='--', linewidth=1.5)
    ax.set_title('Stock Price: History vs. Ground Truth vs. Prediction')
    ax.set_ylabel('Price ($)')
    ax.set_xlabel('Time (Days)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print("Plot saved successfully.")