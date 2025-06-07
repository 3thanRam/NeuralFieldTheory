#training.py
import os
import random 
import torch
import torch.optim as optim 
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
car_return=_term_move_up() + '\r'
from torch.utils.data import Dataset, DataLoader
from config import myconfig
# from lossfunction import CompositeCriterion # Not needed here, it's in config

# ... (get_training_texts and TextDataset remain the same) ...
def get_training_texts(config:myconfig):
    corpus_content = config.corpus_content
    if not corpus_content and config.mode == "train" and not config.load: 
        print("Warning: Corpus content is empty. No training data will be generated.")
        return [], []
    if not corpus_content and config.mode == "train" and config.load: 
        print("Error: Corpus content is empty when loading a model for training. Text data needs to be available.")
        return [], []

    training_texts_list_full = []
    validation_split_ratio=config.validation_split_ratio
    if not corpus_content:
        return [], []

    text_chunk_len = config.max_seq_len - 2 
    if text_chunk_len <= 0:
        print(f"Warning: max_seq_len ({config.max_seq_len}) too small, resulting in non-positive chunk length. Setting to 1.")
        text_chunk_len = 1
    
    stride = max(1, text_chunk_len // 2 if len(corpus_content) > 2 * text_chunk_len else 1)

    if len(corpus_content) < text_chunk_len:
        if corpus_content.strip(): 
            training_texts_list_full.append(corpus_content)
    else:
        for i in range(0, len(corpus_content) - text_chunk_len + 1, stride):
            chunk = corpus_content[i: i + text_chunk_len]
            if chunk.strip(): 
                 training_texts_list_full.append(chunk)
    
    if not training_texts_list_full and corpus_content.strip(): 
        training_texts_list_full.append(corpus_content[:text_chunk_len])

    if not training_texts_list_full:
        print("No text segments generated for training/validation.")
        return [], []

    print(f"Generated {len(training_texts_list_full)} raw text segments in total.")

    if validation_split_ratio > 0.0 and len(training_texts_list_full) > 1:
        random.shuffle(training_texts_list_full) 
        
        num_val_samples = max(1, int(len(training_texts_list_full) * validation_split_ratio))
        if len(training_texts_list_full) - num_val_samples < 1 and len(training_texts_list_full) > num_val_samples : 
             num_val_samples = len(training_texts_list_full) - 1

        if num_val_samples >= len(training_texts_list_full): 
            print(f"Warning: Not enough data ({len(training_texts_list_full)} samples) to create a validation split with ratio {validation_split_ratio}. Using all for training.")
            return training_texts_list_full, []

        split_idx = len(training_texts_list_full) - num_val_samples 

        train_texts = training_texts_list_full[:split_idx]
        val_texts = training_texts_list_full[split_idx:]
        
        if not train_texts or not val_texts: 
            print(f"Warning: Splitting resulted in empty train or validation set. Using all {len(training_texts_list_full)} for training.")
            return training_texts_list_full, []

        print(f"Split data: {len(train_texts)} train, {len(val_texts)} validation segments.")
        return train_texts, val_texts
    else:
        if validation_split_ratio > 0.0 :
             print(f"Warning: Not enough data ({len(training_texts_list_full)} samples) or zero split ratio. Using all for training.")
        return training_texts_list_full, []


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.max_len,
            truncation=True, padding="max_length"
        )
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, target_ids

def epoch_loop(config:myconfig, epoch_idx: int) -> bool: 
    tokenizer = config.tokenizer
    device=config.device
    num_workers = 0 
    pin_memory_flag = device.type == 'cuda'

    if not config.raw_training_texts:
        print(f"Epoch {epoch_idx+1}: No training data. Skipping training phase.")
        return True 

    train_dataset=TextDataset(texts=config.raw_training_texts, tokenizer=tokenizer, max_len=config.max_seq_len)
    train_dataloader =DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    
    val_dataloader = None
    if config.raw_validation_texts:
        val_dataset = TextDataset(texts=config.raw_validation_texts, tokenizer=tokenizer, max_len=config.max_seq_len)
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=max(0, num_workers // 2), pin_memory=pin_memory_flag 
        )

    model=config.model
    optimizer = config.optimizer
    scheduler = config.scheduler
    criterion = config.criterion # This is now CompositeCriterion
    
    epoch_total_loss_sum = 0.0
    # Store sums for individual loss components if you want to average them over epoch
    epoch_nll_loss_sum = 0.0
    epoch_h_loss_sum = 0.0
    epoch_decor_loss_sum = 0.0
    epoch_gateh_loss_sum = 0.0

    processed_batches_train = 0
    stop_training_flag = False 

    model.train()
    batch_iterator_desc = f"Train Epoch {epoch_idx+1}/{config.num_epochs}"
    batch_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=batch_iterator_desc, unit="batch", leave=False)

    try:
        for batch_idx, batch_data in enumerate(batch_iterator): # Added batch_idx for detailed logging
            input_tensor, target_tensor = batch_data
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            # padding_mask is (B,T) boolean, True for valid tokens
            padding_mask = (input_tensor != config.pad_idx) 

            optimizer.zero_grad(set_to_none=True)

            # Model forward pass to get outputs for CompositeCriterion
            # return_for_criterion=True
            logits, hidden_states, gates_list = model(
                input_tensor, 
                mask=padding_mask, 
                return_for_criterion=True
            )

            # Loss calculation using CompositeCriterion
            loss, loss_logs = criterion(
                logits, 
                target_tensor, 
                hidden_states, 
                gates_list, 
                padding_mask=padding_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total_loss_sum += loss.item() # This is loss_logs["total"]
            epoch_nll_loss_sum += loss_logs["nll"]
            epoch_h_loss_sum += loss_logs["H_logits"]
            epoch_decor_loss_sum += loss_logs["decor_hidden"]
            epoch_gateh_loss_sum += loss_logs["gateH_mfi"]
            processed_batches_train += 1

            # Update tqdm postfix with detailed losses
            batch_iterator.set_postfix(
                total=f"{loss.item():.3f}", 
                nll=f"{loss_logs['nll']:.3f}",
                H=f"{loss_logs['H_logits']:.3f}",
                decor=f"{loss_logs['decor_hidden']:.3f}",
                gateH=f"{loss_logs['gateH_mfi']:.3f}"
            )
            # Optional: More frequent detailed logging to console (can be verbose)
            # if (batch_idx + 1) % 50 == 0: # Log every 50 batches
            #     log_str = f"[E{epoch_idx+1} B{batch_idx+1}] "
            #     log_str += " ".join([f"{k}:{v:.3f}" for k,v in loss_logs.items()])
            #     batch_iterator.write(car_return + log_str)
    except KeyboardInterrupt: 
        print("\nTraining stopped early by user (KeyboardInterrupt in batch_iterator loop).")
        stop_training_flag = True 

         

    batch_iterator.close()

    if processed_batches_train == 0:
        print(f"Epoch {epoch_idx+1}: No batches were processed during training. Stopping.")
        return True 

    avg_train_total_loss = epoch_total_loss_sum / processed_batches_train
    avg_train_nll_loss = epoch_nll_loss_sum / processed_batches_train
    # ... average other components similarly if needed for epoch summary
    lr_str = f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    print(f"Epoch {epoch_idx+1} Train Avg Total Loss: {avg_train_total_loss:.4f} (NLL: {avg_train_nll_loss:.4f}) | {lr_str}")

    current_val_loss = avg_train_total_loss # Default if no validation or val fails (use total loss for scheduler)
    if val_dataloader and (hasattr(val_dataloader, 'dataset') and len(val_dataloader.dataset) > 0):
        model.eval()
        epoch_val_total_loss_sum = 0.0
        epoch_val_nll_loss_sum = 0.0 # Track NLL for validation too
        processed_batches_val = 0
        val_iterator_desc = f"Validate Epoch {epoch_idx+1}"
        val_iterator = tqdm(val_dataloader, total=len(val_dataloader), desc=val_iterator_desc, unit="batch", leave=False)
        with torch.no_grad():
            for batch_data_val in val_iterator:
                input_tensor_val, target_tensor_val = batch_data_val
                input_tensor_val, target_tensor_val = input_tensor_val.to(device), target_tensor_val.to(device)
                padding_mask_val = (input_tensor_val != config.pad_idx)
                
                logits_val, hidden_states_val, gates_list_val = model(
                    input_tensor_val, 
                    mask=padding_mask_val, 
                    return_for_criterion=True
                )
                
                loss_val, loss_logs_val = criterion(
                    logits_val, 
                    target_tensor_val, 
                    hidden_states_val, 
                    gates_list_val, 
                    padding_mask=padding_mask_val
                )
                epoch_val_total_loss_sum += loss_val.item()
                epoch_val_nll_loss_sum += loss_logs_val["nll"]
                processed_batches_val += 1
                val_iterator.set_postfix(total_loss=f"{loss_val.item():.4f}", nll=f"{loss_logs_val['nll']:.4f}")
            
        val_iterator.close()
        if processed_batches_val > 0:
            current_val_loss = epoch_val_total_loss_sum / processed_batches_val # Use total loss for best model tracking
            avg_val_nll_loss = epoch_val_nll_loss_sum / processed_batches_val
            print(f"Epoch {epoch_idx+1} Val Avg Total Loss: {current_val_loss:.4f} (NLL: {avg_val_nll_loss:.4f})")
        else:
            print(f"Epoch {epoch_idx+1}: No batches processed during validation, using train loss for scheduler.")
            current_val_loss = avg_train_total_loss 
    
    if scheduler:
        # Scheduler typically monitors the main validation loss metric (total loss here)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss) 
        else: 
            scheduler.step()
    
    if current_val_loss < config.best_val_loss:
        config.best_val_loss = current_val_loss
        config.epochs_no_improve = 0 
        print(f"** New best model found at epoch {epoch_idx+1} with Val Total Loss: {config.best_val_loss:.4f} **")
        checkpoint = {
            'epoch': epoch_idx, 
            'best_val_loss': config.best_val_loss,
            'epochs_no_improve': config.epochs_no_improve,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
        }
        torch.save(checkpoint, config.ckpt)
    else:
        config.epochs_no_improve += 1
        print(f"Val total loss ({current_val_loss:.4f}) did not improve for {config.epochs_no_improve} epoch(s). Best: {config.best_val_loss:.4f}")

    if config.epochs_no_improve >= config.lr_patience: 
        print(f"\nEarly stopping triggered after {config.lr_patience} epochs with no improvement on validation loss.")
        stop_training_flag = True 

    return stop_training_flag 

def train_model(config:myconfig):
    # ... (train_model loop structure remains the same)
    if not config.raw_training_texts and not config.load :
         print("No training texts available and not loading a model. Aborting training.")
         return

    for epoch_idx in range(config.start_epoch, config.num_epochs):
        print(f"\n--- Starting Epoch {epoch_idx+1}/{config.num_epochs} ---")
        
        try:
            should_stop = epoch_loop(config, epoch_idx)
            if should_stop: 
                print("Stopping training based on epoch_loop signal (e.g., early stopping or no data).")
                break
        except KeyboardInterrupt: 
            print("\nEpoch Training stopped by user (KeyboardInterrupt in epoch_idx loop).")
            break 
        except Exception as e:
            print(f"\nAn error occurred during epoch {epoch_idx+1}: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to stop training gracefully.")
            break

    print("Training finished.")