# main.py
import torch
from types import SimpleNamespace
import os

def main():
    # ==================================================================
    #                       MAIN CONTROL PANEL
    # ==================================================================
    TASK = 'stockbot'  # 'chatbot' or 'stockbot'
    MODE = 'test'      # 'train' or 'test'
    LOAD_CHECKPOINT = True
    EPOCHS = 100
    # ==================================================================

    print(f"--- Task: {TASK} | Mode: {MODE} ---")
    torch.manual_seed(42)
    args = SimpleNamespace(load=LOAD_CHECKPOINT, epochs=EPOCHS)
    torch.autograd.set_detect_anomaly(True)

    # --- Delegate to the appropriate task runner ---
    if TASK == 'chatbot':
        from chatbot.tasks import run_training, run_testing
        if MODE == 'train':
            run_training(args)
        elif MODE == 'test':
            run_testing(args)

    elif TASK == 'stockbot':
        from stockbot.tasks import run_training, run_testing
        if MODE == 'train':
            run_training(args)
        elif MODE == 'test':
            run_testing(args)

if __name__ == "__main__":
    main()