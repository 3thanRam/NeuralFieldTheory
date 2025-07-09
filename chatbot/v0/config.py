# config.py
import os
from transformers import GPT2TokenizerFast,get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch 
import torch.optim as optim

from lossfunction import CompositeCriterion
from datasets import load_dataset
from network import OverallLanguageModel 

FILEPATH = os.path.dirname(os.path.realpath(__file__))

class myconfig:
    def __init__(self,mode="train",max_seq_len=60,batch_size=8,num_epochs=10,
                 max_order=5,embed_dim=128,num_configs=32,mlp_ratio=6,
                 validation_split_ratio=0.005,lr=5e-4,lr_patience=2,lr_factor=0.5,
                 start_epoch=0,load=False, num_model_blocks=1,history_len=10**4,
                 # Hyperparameters for CompositeCriterion
                 lambda_H_logits=1e-1, lambda_C_hidden=1e-1, lambda_O_mfi=1e-1,
                 mfi_temperature_schedule_active: bool = True, # Turn on/off
                 mfi_initial_temperature: float = 2.0,
                 mfi_final_temperature: float = 0.5, # Don't go too low (e.g. 0.1) initially
                 mfi_temperature_decay_epochs: int = 0, # If 0, decay over all num_epochs
                 mfi_energy_reg_lambda: float = 0.001,  # Strength for MFI energy regularization
                 mfi_sampling_schedule_active: bool = False, # Turn on/off
                 mfi_sample_mode_until_epoch: int = 0, # Epoch until which 'sample' mode is used, then 'expectation'
                 warmup_steps_ratio=0.1
                 ):

        self.load=load
        self.ckpt=os.path.join(FILEPATH, "model_data","checkpoint.pth.tar")
        self.hyperparams_filepath=os.path.join(FILEPATH, "model_data","hyperparams.json")
        self.tokenizer_dir=os.path.join(FILEPATH, "model_data","tokenizer")
        self.corpus_filepath = os.path.join(FILEPATH, "model_data", "corpus.txt")
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(os.path.join(FILEPATH, "model_data"), exist_ok=True)
        
        self.mode=mode
        self.embed_dim=embed_dim
        self.batch_size=batch_size
        self.start_epoch=start_epoch
        self.num_epochs=num_epochs
        self.best_val_loss=float('inf')
        self.epochs_no_improve = 0 
        self.max_seq_len=max_seq_len
        self.max_order=max_order
        self.num_configs=num_configs
        self.mlp_ratio=mlp_ratio
        self.num_model_blocks = num_model_blocks
        self.validation_split_ratio=validation_split_ratio
        self.lr=lr
        self.lr_patience=lr_patience
        self.lr_factor=lr_factor
        self.history_len=history_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Composite Criterion lambdas
        self.lambda_H_logits = lambda_H_logits
        self.lambda_C_hidden = lambda_C_hidden
        self.lambda_O_mfi = lambda_O_mfi

        self.mfi_temperature_schedule_active = mfi_temperature_schedule_active
        self.mfi_initial_temperature = mfi_initial_temperature
        self.mfi_final_temperature = mfi_final_temperature
        self.mfi_temperature_decay_epochs = mfi_temperature_decay_epochs if mfi_temperature_decay_epochs > 0 else num_epochs
        self.mfi_energy_reg_lambda = mfi_energy_reg_lambda
        self.mfi_sampling_schedule_active = mfi_sampling_schedule_active
        self.mfi_sample_mode_until_epoch = mfi_sample_mode_until_epoch
        self.warmup_steps_ratio=warmup_steps_ratio
        self.corpus_content=""
        
        if load:
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_dir)
            with open(self.corpus_filepath, "r", encoding="utf-8") as f:
                self.corpus_content = f.read()
        else:
            base_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            special_tokens = {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>", "unk_token": "<UNK>"}
            base_tokenizer.add_special_tokens(special_tokens)
            tokenizer = base_tokenizer
            tokenizer.save_pretrained(self.tokenizer_dir)
            
            #hf_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
            #corpus_texts_list = sorted([ex["text"] for ex in hf_dataset["train"] if ex["text"].strip()])
            #self.corpus_content = "\n".join(corpus_texts_list)
            #with open(self.corpus_filepath, "w", encoding="utf-8") as f:
            #    f.write(self.corpus_content)
            print("Loading Penn Treebank dataset...")
            # Use dataset specific split, e.g. 'train', 'validation', 'test'
            hf_dataset_ptb = load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)
            # Concatenate train, validation, test for a larger corpus, or just use train
            corpus_texts_list = []
            for split in ['train', 'validation', 'test']:
                 corpus_texts_list.extend([ex["sentence"] for ex in hf_dataset_ptb[split] if ex["sentence"].strip()])
            self.corpus_content = "\n".join(corpus_texts_list)
            print(f"Loaded Penn Treebank corpus with {len(corpus_texts_list)} sentences.")
            with open(self.corpus_filepath, "w", encoding="utf-8") as f:
                f.write(self.corpus_content)

        self.tokenizer = tokenizer
        self.stoi = tokenizer.get_vocab()
        self.itos = {idx: tok for tok, idx in self.stoi.items()}
        self.vocab_size = len(tokenizer.get_vocab())
        self.pad_idx = tokenizer.pad_token_id
        if self.pad_idx is None:
            raise ValueError("tokenizer.pad_token_id is None. Ensure '<PAD>' token is correctly added.")

        self.model = OverallLanguageModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            pad_idx=self.pad_idx,
            max_order=self.max_order,
            num_configs=self.num_configs,
            mlp_ratio=self.mlp_ratio,
            num_blocks=self.num_model_blocks
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.myscheduler_fct=get_linear_schedule_with_warmup
        self.scheduler = None#optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.lr_patience, factor=self.lr_factor)
        
        if load:
            if os.path.exists(self.ckpt):
                checkpoint = torch.load(self.ckpt, map_location=self.device)
                self.start_epoch = checkpoint['epoch'] + 1 
                self.best_val_loss = checkpoint['best_val_loss']
                self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Best val loss: {self.best_val_loss:.4f}")
            else:
                print(f"Warning: Load=True but checkpoint file not found at {self.ckpt}. Starting from scratch.")
                self.start_epoch = 0
        
        # --- Criterion ---
        self.criterion = CompositeCriterion(
            λ_H=self.lambda_H_logits,
            λ_C=self.lambda_C_hidden,
            λ_O=self.lambda_O_mfi,
            λ_E_mfi=self.mfi_energy_reg_lambda, 
            pad_idx=self.pad_idx
        )
        
        self.raw_training_texts = [] 
        self.raw_validation_texts = []
    def set_scheduler(self,len_trainingdata):
        #if config.scheduler is None:
        num_training_steps_total = self.num_epochs * len_trainingdata # train_dataloader defined earlier
        # If using gradient accumulation:
        # num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        # num_training_steps_total = config.num_epochs * num_update_steps_per_epoch

        num_warmup_steps = int(self.warmup_steps_ratio * num_training_steps_total)

        self.scheduler = self.myscheduler_fct( # or get_cosine_schedule_with_warmup
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps_total
        )
        print(f"Initialized LR scheduler with {num_warmup_steps} warmup steps and {num_training_steps_total} total training steps.")
        return self.scheduler