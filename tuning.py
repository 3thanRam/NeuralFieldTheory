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

from scipy.optimize import minimize

car_return=_term_move_up() + '\r'

def test_config_vector(system:myconfig,l_H,l_C,l_O):

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
    batch_iterator = tqdm(system.train_dataloader, total=len(system.train_dataloader), desc=f"Testing config :{(l_H, l_C, l_O)}", unit="batch", leave=False)
    batch_losses_for_this_run=[]
    for batch_idx, batch_data in enumerate(batch_iterator):
        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system.device), target_tensor.to(system.device)
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
    if (system.bestruns is None) or (len(system.bestruns)<system.num_bestruns) or avg_loss_for_this_run<system.bestruns[0][0]:
        system.bestruns.append(current_result_package)
        system.bestruns.sort(key=lambda x: x[0])
        if len(system.bestruns)>system.num_bestruns:
            system.bestruns=system.bestruns[:10]
    return system.bestruns

def test_config_scalar(x,args):
    l_H,l_C,l_O=x
    system:myconfig=args
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
    
    ##################
    batch_iterator = tqdm(system.train_dataloader, total=len(system.train_dataloader), desc=f"Testing config :{(l_H, l_C, l_O)}", unit="batch", leave=False)
    batch_losses_for_this_run=[]
    avg_loss_for_this_run=0
    completed_batches=0
    for batch_idx, batch_data in enumerate(batch_iterator):
        input_tensor, target_tensor = batch_data
        input_tensor, target_tensor = input_tensor.to(system.device), target_tensor.to(system.device)
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
        completed_batches+=1
        batch_losses_for_this_run.append(loss.item())
        
        batch_iterator.set_postfix(
            total=f"{loss.item():.3f}", 
            nll=f"{loss_logs['nll']:.3f}",
            H=f"{loss_logs['H_logits']:.3f}",
            decor=f"{loss_logs['decor_hidden']:.3f}",
            gateH=f"{loss_logs['gateH_mfi']:.3f}"
        )
        if batch_idx>100:
            break
    ###################
    #current_result_package = (
    #avg_loss_for_this_run,
    #{"lambda_H": l_H, "lambda_C": l_C, "lambda_O": l_O},
    #batch_losses_for_this_run
    #)
    avg_loss_for_this_run/=completed_batches
    print(f"{(l_H,l_C,l_O)}:avg_loss_for_this_run")
    return avg_loss_for_this_run
    
    


def tuning(num_points_per_lambda=5,num_bestruns=10,num_tests=2e2):
    
    system = myconfig(load=False, mode="train")
    system.num_bestruns=num_bestruns
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
    system.train_dataloader =DataLoader(train_dataset, batch_size=system.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
        
    
    range_l_H = np.logspace(-2, 0, num_points_per_lambda)
    range_l_C = np.logspace(-2, 0, num_points_per_lambda)
    range_l_O = np.logspace(-2, 0, num_points_per_lambda)
    run_counter = 0
    total_combinations = len(range_l_H) * len(range_l_C) * len(range_l_O)
    #best_overall_val_loss=float('inf')
    system.bestruns=[]
    res=minimize(test_config_scalar, [0.1,0.1,0.1],system, method='SLSQP', 
                 bounds=[(0, None), (0, None), (0, None)])
    print(res.x)
    #for l_H, l_C, l_O in product(range_l_H, range_l_C, range_l_O):
    #    run_counter+=1
    #    #test_config(system,l_H,l_C,l_O)
    #    
    #    print(f"Tested {run_counter}/{total_combinations} configurations")
        
        

    #print(f"\n{system.num_bestruns} Best hyperparameters found:")
    #for (Avloss,params) in system.bestruns:
    #    print(f"lambda_H:{params["lambda_H"]},lambda_C:{params["lambda_H"]},lambda_C:{params["lambda_O"]},loss{Avloss}")




if __name__ == "__main__":
    tuning()