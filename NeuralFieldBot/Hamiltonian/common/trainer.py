# common/trainer.py - Contains the generic training loop.
import torch
import os
from tqdm.auto import tqdm

def epoch_loop(config, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch_idx):
    stop_training = False
    device = config.device
    log_interval = 20
    
    model.train()
    total_train_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{config.num_epochs} [Train]", leave=False)
    
    try:
        for i, (batch_X, batch_Y) in enumerate(train_iterator):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            model_outputs = model(batch_X, return_internals=True)
            loss, loss_components = criterion(model_outputs, batch_Y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if (i + 1) % log_interval == 0:
                log_str = " | ".join([f"{k.capitalize()}:{v:.3f}" for k, v in loss_components.items()])
                train_iterator.write(log_str)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        stop_training = True
    finally:
        train_iterator.close()
        
    if stop_training:
        return float('nan'), float('nan'), True
        
    avg_train_loss = total_train_loss / len(train_loader) if train_loader else 0
    print(f"\nEpoch {epoch_idx+1} Train Avg Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    model.eval()
    avg_val_loss = avg_train_loss
    if val_loader:
        total_val_loss = 0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch_idx+1} [Val]", leave=False)
        try:
            with torch.no_grad():
                for batch_X_val, batch_Y_val in val_iterator:
                    batch_X_val, batch_Y_val = batch_X_val.to(device), batch_Y_val.to(device)
                    outputs_val = model(batch_X_val, return_internals=True)
                    loss_val, _ = criterion(outputs_val, batch_Y_val)
                    total_val_loss += loss_val.item()
                    val_iterator.set_postfix({'loss': f"{loss_val.item():.4f}"})
        except KeyboardInterrupt:
            print("\nValidation interrupted by user.")
            stop_training = True
        finally:
            val_iterator.close()
            
        if stop_training:
            return avg_train_loss, float('nan'), True
        avg_val_loss = total_val_loss / len(val_loader) if val_loader else avg_train_loss
        print(f"Epoch {epoch_idx+1} Val Avg Loss: {avg_val_loss:.4f}")
        
    scheduler.step(avg_val_loss)
    return avg_train_loss, avg_val_loss, stop_training

def train_model(config, model, criterion, train_loader, val_loader):
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    if config.load and os.path.exists(config.ckpt_path):
        ckpt = torch.load(config.ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt.get('scheduler_state_dict'))
        config.start_epoch = ckpt.get('epoch', 0)
        config.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {config.start_epoch}. Best val loss: {config.best_val_loss:.4f}")
        
    try:
        for epoch_idx in range(config.start_epoch, config.num_epochs):
            _, avg_val_loss, stop_training = epoch_loop(config, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch_idx)
            if stop_training:
                break
            if avg_val_loss < config.best_val_loss:
                config.best_val_loss = avg_val_loss
                print(f"** New best model (Val Loss: {avg_val_loss:.4f}), saving checkpoint... **")
                state = {
                    'epoch': epoch_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': config.best_val_loss,
                    'scheduler_state_dict': scheduler.state_dict()
                }
                os.makedirs(os.path.dirname(config.ckpt_path), exist_ok=True)
                torch.save(state, config.ckpt_path)
    except KeyboardInterrupt:
        print("\n--- Training stopped by user. ---")
        
    print("\nTraining finished.")