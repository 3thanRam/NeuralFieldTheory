# config.py
import os
import torch
import torch.optim as optim
from transformers import GPT2TokenizerFast
from datasets import load_dataset

from lossfunction import CompositeCriterion
from network import HamiltonianModel
from transformer import TransformerModel

FILEPATH = os.path.dirname(os.path.realpath(__file__))

class system_config:
    def __init__(self, load=False, mode="train", model_type="hamiltonian",
                 num_epochs=50, batch_size=32, embed_dim=256, start_epoch=0, max_seq_len=128,
                 validation_split_ratio=0.1, lr=1e-4, lr_patience=3,
                 # Hamiltonian-specific params
                 num_blocks=2,
                 dropout=0.1,
                 momentum_noise_sigma=0.1,
                 # Attention-specific params
                 n_head=4,
                 num_subspaces=4,
                 subspace_dim=64,
                 # Loss function weights
                 state_norm_weight=1e-6, 
                 energy_conservation_weight=1e-6,
                 decorrelation_weight=1,
                 reversibility_weight=1,
                 jacobian_weight=0.1,
                 momentum_consistency_weight=0.5,
                 ):

        self.ckpt = os.path.join(FILEPATH, "model_data", "checkpoint.pth.tar")
        self.hyperparams_filepath = os.path.join(FILEPATH, "model_data", "hyperparams.json")
        self.tokenizer_dir = os.path.join(FILEPATH, "model_data", "tokenizer")
        self.corpus_filepath = os.path.join(FILEPATH, "model_data", "corpus.txt")
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(os.path.join(FILEPATH, "model_data"), exist_ok=True)

        self.load = load
        self.mode = mode
        self.model_type = model_type
        self.raw_training_texts = []
        self.raw_validation_texts = []
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        # These will be overridden by load_checkpoint if loading
        self.start_epoch = start_epoch
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.max_seq_len = max_seq_len
        self.validation_split_ratio = validation_split_ratio
        self.lr = lr
        self.lr_patience = lr_patience

        # Store all hyperparameters for model and loss
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.n_head = n_head
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.state_norm_weight = state_norm_weight
        self.energy_conservation_weight = energy_conservation_weight
        self.decorrelation_weight = decorrelation_weight
        self.reversibility_weight = reversibility_weight
        self.jacobian_weight = jacobian_weight
        self.momentum_consistency_weight = momentum_consistency_weight 
        self.momentum_noise_sigma=momentum_noise_sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.get_tokeniser()

        self.criterion = CompositeCriterion(
            state_norm_weight=self.state_norm_weight,
            energy_conservation_weight=self.energy_conservation_weight,
            decorrelation_weight=self.decorrelation_weight,
            reversibility_weight=self.reversibility_weight,
            jacobian_weight=self.jacobian_weight,
            momentum_consistency_weight=self.momentum_consistency_weight,
            ignore_index=self.pad_idx
        )

        self.get_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.lr_patience)
        
        # --- NEW: Call the checkpoint loading logic at the end of initialization ---
        self.load_checkpoint()

        if self.pad_idx is None:
            raise ValueError("tokenizer.pad_token_id is None. Ensure '<PAD>' token is correctly added.")

    # --- NEW: A dedicated method to handle loading from a checkpoint ---
    def load_checkpoint(self):
        """Loads model, optimizer, and training state from a checkpoint file."""
        if not self.load:
            print("`load` is False. Initializing model from scratch.")
            return

        if not os.path.exists(self.ckpt):
            print(f"WARNING: Checkpoint file not found at {self.ckpt}. Starting from scratch.")
            return

        print(f"Loading checkpoint from {self.ckpt}...")
        # Use map_location to handle moving between devices (e.g., GPU-trained to CPU-inference)
        checkpoint = torch.load(self.ckpt, map_location=self.device)

        # Load the model's learned weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # If in training mode, also load optimizer and scheduler states
        if self.mode == "train":
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Warning: Optimizer state not found in checkpoint. Initializing new optimizer.")

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Warning: Scheduler state not found in checkpoint. Initializing new scheduler.")

        # Restore the training progress
        self.start_epoch = checkpoint['epoch'] + 1  # We start from the next epoch
        self.best_val_loss = checkpoint['best_val_loss']
        # Use .get() for backward compatibility with old checkpoints that might not have this
        self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        
        print(f"Checkpoint loaded successfully. Resuming from epoch {self.start_epoch}.")
        print(f"Best validation loss from previous run: {self.best_val_loss:.4f}")

    def get_tokeniser(self):
        # ... (this method is unchanged) ...
        if self.load and os.path.exists(os.path.join(self.tokenizer_dir, "vocab.json")):
            print(f"Loading tokenizer from {self.tokenizer_dir}...")
            tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_dir)
            if os.path.exists(self.corpus_filepath):
                with open(self.corpus_filepath, "r", encoding="utf-8") as f:
                    self.corpus_content = f.read()
            else:
                self.corpus_content = ""
                print("Warning: corpus.txt not found, but it's needed for training.")
        else:
            print("Initializing new tokenizer and downloading corpus...")
            base_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            special_tokens = {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>", "unk_token": "<UNK>"}
            base_tokenizer.add_special_tokens(special_tokens)
            tokenizer = base_tokenizer
            tokenizer.save_pretrained(self.tokenizer_dir)

            print("Loading Penn Treebank dataset...")
            hf_dataset_ptb = load_dataset("ptb_text_only", "penn_treebank", trust_remote_code=True)
            corpus_texts_list = []
            for split in ['train', 'validation', 'test']:
                 corpus_texts_list.extend([ex["sentence"] for ex in hf_dataset_ptb[split] if ex["sentence"].strip()])
            self.corpus_content = "\n".join(corpus_texts_list)
            print(f"Loaded Penn Treebank corpus with {len(corpus_texts_list)} sentences.")
            with open(self.corpus_filepath, "w", encoding="utf-8") as f:
                f.write(self.corpus_content)

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.pad_idx = tokenizer.pad_token_id
        print(f"Tokenizer loaded. Vocab size: {self.vocab_size}")

    def get_model(self):
        """Instantiates the selected model."""
        # This part is correct: we always need to build the model architecture first.
        # The weights will be loaded into this architecture by load_checkpoint().
        if self.model_type == "transformer":
            print("Initializing Standard TransformerModel...")
            model = TransformerModel(
                vocab_size=self.vocab_size, embed_dim=self.embed_dim, nhead=self.n_head,
                num_encoder_layers=self.num_blocks, # Reuse num_blocks for layer count
                dim_feedforward=self.embed_dim * 4, dropout=self.dropout, pad_idx=self.pad_idx
            )
        elif self.model_type == "hamiltonian":
            print("Initializing HamiltonianModel...")
            model_kwargs = {
                'dropout': self.dropout,
                'n_head': self.n_head,
                'num_subspaces': self.num_subspaces,
                'subspace_dim': self.subspace_dim
            }
            model = HamiltonianModel(
                num_blocks=self.num_blocks, input_dim=self.embed_dim, d_embedding=self.embed_dim,
                d_hidden_dim=self.embed_dim * 4, output_dim=self.vocab_size,
                vocab_size=self.vocab_size, embed_dim=self.embed_dim, pad_idx=self.pad_idx,momentum_noise_sigma=self.momentum_noise_sigma,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model = model.to(self.device)
        print(f"Model '{self.model_type}' created with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters.")