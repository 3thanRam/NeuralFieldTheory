#training.py
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import math

from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
car_return = _term_move_up() + '\r'
from torch.utils.data import Dataset, DataLoader
#from torch.amp import GradScaler, autocast
from config import system_config
from lossfunction import CompositeCriterion

def get_training_texts(system: system_config):
    corpus_content = system.corpus_content
    if not corpus_content and system.mode == "train":
        print("Warning: Corpus content is empty. No training data will be generated.")
        return [], []

    training_texts_list_full = []
    validation_split_ratio = system.validation_split_ratio
    text_chunk_len = system.max_seq_len - 2  # For SOS/EOS tokens
    if text_chunk_len <= 0:
        print(f"Warning: max_seq_len ({system.max_seq_len}) too small. Setting chunk len to 1.")
        text_chunk_len = 1
    stride = max(1, text_chunk_len // 2)

    if len(corpus_content) < text_chunk_len:
        if corpus_content.strip(): training_texts_list_full.append(corpus_content)
    else:
        for i in range(0, len(corpus_content) - text_chunk_len + 1, stride):
            chunk = corpus_content[i: i + text_chunk_len]
            if chunk.strip(): training_texts_list_full.append(chunk)

    if not training_texts_list_full:
        print("No text segments generated for training/validation.")
        return [], []
    print(f"Generated {len(training_texts_list_full)} raw text segments.")

    if validation_split_ratio > 0.0 and len(training_texts_list_full) > 1:
        random.shuffle(training_texts_list_full)
        num_val_samples = max(1, int(len(training_texts_list_full) * validation_split_ratio))
        if len(training_texts_list_full) - num_val_samples < 1: num_val_samples = len(training_texts_list_full) - 1
        if num_val_samples >= len(training_texts_list_full):
            print(f"Warning: Not enough data for validation split. Using all for training.")
            return training_texts_list_full, []
        split_idx = len(training_texts_list_full) - num_val_samples
        train_texts, val_texts = training_texts_list_full[:split_idx], training_texts_list_full[split_idx:]
        print(f"Split data: {len(train_texts)} train, {len(val_texts)} validation segments.")
        return train_texts, val_texts
    else:
        return training_texts_list_full, []


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(
            f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}",
            max_length=self.max_len, truncation=True, padding="max_length"
        )
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        target_ids[input_ids == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        return input_ids, target_ids


def epoch_loop(system: system_config, epoch_idx: int) -> bool:
    tokenizer = system.tokenizer; device = system.device; pin_memory_flag = device.type == 'cuda'
    model: nn.Module = system.model
    if not system.raw_training_texts:
        print(f"Epoch {epoch_idx + 1}: No training data. Skipping training phase.")
        return True

    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
    train_dataset = TextDataset(texts=system.raw_training_texts, tokenizer=tokenizer, max_len=system.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    val_dataloader = None
    if system.raw_validation_texts:
        val_dataset = TextDataset(texts=system.raw_validation_texts, tokenizer=tokenizer, max_len=system.max_seq_len)
        val_dataloader = DataLoader(val_dataset, batch_size=system.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag)

    optimizer: optim.Optimizer = system.optimizer; scheduler = system.scheduler; criterion: CompositeCriterion = system.criterion
    scaler = torch.amp.grad_scaler.GradScaler(enabled=(device.type == 'cuda'))
    epoch_total_loss_sum = 0; processed_batches_train = 0; stop_training_flag = False

    log_interval = 10
    model.train()
    # Use enumerate to get the batch index for periodic logging
    batch_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Train Epoch {epoch_idx + 1}/{system.num_epochs}", unit="batch", leave=False)
    
    try:
        for batch_idx, batch_data in batch_iterator:
            input_tensor, target_tensor = batch_data[0].to(device), batch_data[1].to(device)
            padding_mask = (input_tensor != system.pad_idx)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast_mode.autocast(device_type=device.type,enabled=(device.type == 'cuda')):
                outputs = model(input_tensor, mask=padding_mask, return_internals=True)
                loss, loss_logs = criterion(outputs, target_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_total_loss_sum += loss_logs['total_loss']
            processed_batches_train += 1
            
            # --- CHANGE 1: Simplify the postfix to show only key metrics ---
            # This ensures the main progress bar line stays clean and doesn't truncate.
            postfix_dict = {
                'loss': f"{loss_logs['total_loss']:.2f}",
                'nll': f"{loss_logs['cross_entropy']:.2f}"
            }
            batch_iterator.set_postfix(postfix_dict)

            # --- CHANGE 2: Periodically print a detailed, multi-line summary ---
            if (batch_idx + 1) % log_interval == 0:
                # Construct the detailed string
                log_str = (
                    f"  [Batch {batch_idx + 1}/{len(train_dataloader)}] "
                    f"Total: {loss_logs['total_loss']:.3f} | NLL: {loss_logs['cross_entropy']:.3f}\n"
                    f"  Aux -> SN: {loss_logs['state_norm']:.3f} | "
                    f"EC: {loss_logs['energy_conservation']:.3f} | "
                    f"Rev: {loss_logs['reversibility']:.3f}\n"
                    f"         Dec: {loss_logs['decorrelation']:.3f} | "
                    f"Jac: {loss_logs['jacobian']:.3f} | "
                    f"Mom: {loss_logs['mom_consistency']:.3f}"
                )
                # Use tqdm.write to print above the progress bar without disturbing it
                batch_iterator.write(log_str)

    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user."); stop_training_flag = True
    finally: batch_iterator.close()

    if processed_batches_train == 0:
        print(f"Epoch {epoch_idx + 1}: No batches processed. Stopping.")
        return True
    avg_train_total_loss = epoch_total_loss_sum / processed_batches_train
    print(f"Epoch {epoch_idx + 1} Train Avg Loss: {avg_train_total_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    current_val_loss = avg_train_total_loss
    if val_dataloader:
        model.eval()
        epoch_val_total_loss_sum = 0.0; processed_batches_val = 0
        val_iterator = tqdm(val_dataloader, desc=f"Validate Epoch {epoch_idx + 1}", unit="batch", leave=False)
        # In validation, we still need the grad context for the model's forward pass
        with torch.enable_grad():
            try:
                for batch_data_val in val_iterator:
                    input_tensor_val, target_tensor_val = batch_data_val[0].to(device), batch_data_val[1].to(device)
                    padding_mask_val = (input_tensor_val != system.pad_idx)
                    
                    with torch.amp.autocast_mode.autocast(device_type=device.type,enabled=(device.type == 'cuda')):
                        # --- CHANGE 1 (repeated): Unpack outputs cleanly ---
                        outputs_val = model(input_tensor_val, mask=padding_mask_val, return_internals=True)
                    
                    # --- CHANGE 2 (repeated): Call criterion cleanly ---
                    # The loss calculation itself doesn't need gradients
                    with torch.no_grad():
                        loss_val, loss_logs_val = criterion(outputs_val, target_tensor_val)

                    epoch_val_total_loss_sum += loss_logs_val['total_loss']
                    processed_batches_val += 1
                    val_iterator.set_postfix({"loss": f"{loss_logs_val['total_loss']:.4f}"})
            except KeyboardInterrupt: print(f"\nValidation interrupted by user.")
            finally: val_iterator.close()
        if processed_batches_val > 0:
            current_val_loss = epoch_val_total_loss_sum / processed_batches_val
            print(f"Epoch {epoch_idx + 1} Val Avg Loss: {current_val_loss:.4f}")

    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

    if current_val_loss < system.best_val_loss:
        system.best_val_loss = current_val_loss; system.epochs_no_improve = 0
        print(f"** New best model found (Val Loss: {system.best_val_loss:.4f}), saving checkpoint... **")
        torch.save({'epoch': epoch_idx, 'best_val_loss': system.best_val_loss, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None}, system.ckpt)
    else:
        system.epochs_no_improve += 1
        print(f"Val loss did not improve for {system.epochs_no_improve} epoch(s). Best: {system.best_val_loss:.4f}")

    if system.epochs_no_improve >= system.lr_patience * 2:
        print(f"\nEarly stopping triggered after {system.epochs_no_improve} epochs with no improvement.")
        stop_training_flag = True
    return stop_training_flag


def train_model(system: system_config):
    if not system.raw_training_texts and not system.load:
        print("No training texts available and not loading a model. Aborting training.")
        return
    for epoch_idx in range(system.start_epoch, system.num_epochs):
        print(f"\n--- Starting Epoch {epoch_idx + 1}/{system.num_epochs} ---")
        try:
            if epoch_loop(system, epoch_idx):
                print("Stopping training...")
                break
        except Exception as e:
            print(f"{car_return}\nAn error occurred during epoch {epoch_idx + 1}: {e}"); import traceback; traceback.print_exc()
            break
    print(f"{car_return}Training finished.")