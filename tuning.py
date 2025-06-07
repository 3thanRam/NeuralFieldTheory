#tuning.py
import numpy as np
import torch
import torch.optim as optim
from itertools import product
from torch.utils.data import Dataset, DataLoader

from config import myconfig
from lossfunction import CompositeCriterion
from training import get_training_texts,TextDataset
from tqdm.auto import tqdm
from tqdm._utils import _term_move_up
car_return=_term_move_up() + '\r'

def tuning(num_points_per_lambda=5,num_bestruns=10):
    system = myconfig(load=False, mode="train")
    train_texts, val_texts = get_training_texts(system)
    system.raw_training_texts = train_texts
    system.raw_validation_texts = val_texts
    if not train_texts:
        print("No training data generated. Exiting.")
        return
    
    device=system.device
    num_workers = 0 
    pin_memory_flag = device.type == 'cuda'


    train_dataset=TextDataset(texts=system.raw_training_texts, tokenizer=system.tokenizer, max_len=system.max_seq_len)
    train_dataloader =DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
        
    
    range_l_H = np.logspace(-2, 0, num_points_per_lambda)
    range_l_C = np.logspace(-2, 0, num_points_per_lambda)
    range_l_O = np.logspace(-2, 0, num_points_per_lambda)
    run_counter = 0
    total_combinations = len(range_l_H) * len(range_l_C) * len(range_l_O)
    #best_overall_val_loss=float('inf')
    bestruns=[]
    for l_H, l_C, l_O in product(range_l_H, range_l_C, range_l_O):
        run_counter+=1
        system.lambda_H_logits = l_H
        system.lambda_C_hidden=l_C
        system.lambda_O_mfi=l_O

        system.criterion = CompositeCriterion(
            λ_H=system.lambda_H_logits,
            λ_C=system.lambda_C_hidden,
            λ_O=system.lambda_O_mfi,
            pad_idx=system.pad_idx
        )
        
        system.optimizer = optim.AdamW(system.model.parameters(), lr=system.lr)
        system.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            system.optimizer, 'min', 
            patience=system.lr_patience, 
            factor=system.lr_factor
        )
        system.start_epoch = 0
        system.best_val_loss = float('inf')
        system.epochs_no_improve = 0
        
        ##################
        batch_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Tuning config n{run_counter}:{(l_H, l_C, l_O)}", unit="batch", leave=False)
        batch_losses_for_this_run=[]
        for batch_idx, batch_data in enumerate(batch_iterator):
            input_tensor, target_tensor = batch_data
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            # padding_mask is (B,T) boolean, True for valid tokens
            padding_mask = (input_tensor != system.pad_idx) 

            system.optimizer.zero_grad(set_to_none=True)

            # Model forward pass to get outputs for CompositeCriterion
            # return_for_criterion=True
            logits, hidden_states, gates_list = system.model(
                input_tensor, 
                mask=padding_mask, 
                return_for_criterion=True
            )

            # Loss calculation using CompositeCriterion
            loss, loss_logs = system.criterion(
                logits, 
                target_tensor, 
                hidden_states, 
                gates_list, 
                padding_mask=padding_mask
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.model.parameters(), 1.0)
            system.optimizer.step()

            avg_loss_for_this_run += loss.item() # loss_logs["total"]
            batch_losses_for_this_run.append(loss.item())
            #epoch_nll_loss_sum += loss_logs["nll"]
            #epoch_h_loss_sum += loss_logs["H_logits"]
            #epoch_decor_loss_sum += loss_logs["decor_hidden"]
            #epoch_gateh_loss_sum += loss_logs["gateH_mfi"]
            

            batch_iterator.set_postfix(
                total=f"{loss.item():.3f}", 
                nll=f"{loss_logs['nll']:.3f}",
                H=f"{loss_logs['H_logits']:.3f}",
                decor=f"{loss_logs['decor_hidden']:.3f}",
                gateH=f"{loss_logs['gateH_mfi']:.3f}"
            )
        ###################
        current_result_package = (
        avg_loss_for_this_run,
        {"lambda_H": l_H, "lambda_C": l_C, "lambda_O": l_O},
        batch_losses_for_this_run
        )
        if (bestruns is None) or (len(bestruns)<num_bestruns) or avg_loss_for_this_run<bestruns[0][0]:
            bestruns.append(current_result_package)
            bestruns.sort(key=lambda x: x[0])
        print(f"Tested {run_counter}/{total_combinations} configurations")
    

    print(f"\n{num_bestruns} Best hyperparameters found:")
    for (Avloss,params) in bestruns:
        print(f"lambda_H:{params["lambda_H"]},lambda_C:{params["lambda_H"]},lambda_C:{params["lambda_O"]},loss{Avloss}")




if __name__ == "__main__":
    tuning()