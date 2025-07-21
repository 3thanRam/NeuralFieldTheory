# common/config.py - Defines configurations for each task.
import os
import torch
from .NeuralFieldNetwork import LanguageModel, TimeSeriesPredictor
from .lossfunction import CompositeLoss, ChatbotBaseLoss, StockbotBaseLoss

class BaseConfig:
    def __init__(self, args):
        self.load = args.load
        self.num_epochs = args.epochs
        self.batch_size = 32
        self.embed_dim = 128
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.lr = 5e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_blocks = 1
        self.d_hidden_dim = self.embed_dim * 4
        self.dropout = 0.1
        self.n_head = 4
        self.subspace_dim = self.embed_dim // self.n_head
        self.kernel_size = 3
        print(f"Using device: {self.device}")

class ChatbotConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = "chatbot"
        self.ckpt_path = os.path.join(os.path.dirname(__file__), "..", "model_data", "chatbot_checkpoint.pth.tar")
        self.vocab_size = None
        self.pad_idx = None
        self.aux_weights = {'state_norm': 1.0, 'energy_conservation': 1e-3, 'reversibility': 1.0}

    def get_model_and_criterion(self):
        model_kwargs = {
            'd_hidden_dim': self.d_hidden_dim,
            'num_blocks': self.num_blocks,
            'dropout': self.dropout,
            'n_head': self.n_head,
            'subspace_dim': self.subspace_dim,
            'kernel_size': self.kernel_size
        }
        model = LanguageModel(self.vocab_size, self.embed_dim, self.pad_idx, **model_kwargs)
        base_criterion = ChatbotBaseLoss(self.pad_idx)
        return model, CompositeLoss(base_criterion, self.aux_weights)

class StockbotConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = "stockbot"
        self.ckpt_path = os.path.join(os.path.dirname(__file__), "..", "model_data", "stockbot_checkpoint.pth.tar")
        self.symbols = ["SPY", "AAPL", "MSFT", "GOOGL"]
        self.primary_symbol = "SPY"
        self.years_of_data = 5
        self.sequence_length = 60
        self.val_split_ratio = 0.1
        self.num_input_features = len(self.symbols) * 5
        self.num_output_predictions = self.num_input_features
        self.primary_symbol_idx = self.symbols.index(self.primary_symbol)
        self.huber_weight = 0.7
        self.direction_weight = 0.3
        self.aux_weights = {'state_norm': 1, 'energy_conservation': 1e-3, 'reversibility': 1}

    def get_model_and_criterion(self):
        model_kwargs = {
            'd_hidden_dim': self.d_hidden_dim,
            'num_blocks': self.num_blocks,
            'dropout': self.dropout,
            'n_head': self.n_head,
            'subspace_dim': self.subspace_dim,
            'kernel_size': self.kernel_size
        }
        model = TimeSeriesPredictor(
            num_input_features=self.num_input_features,
            num_output_predictions=self.num_output_predictions,
            embed_dim=self.embed_dim,
            **model_kwargs
        )
        base_loss = StockbotBaseLoss(
            primary_symbol_idx=self.primary_symbol_idx,
            huber_weight=self.huber_weight,
            direction_weight=self.direction_weight
        )
        criterion = CompositeLoss(base_loss_fn=base_loss, aux_weights=self.aux_weights)
        return model, criterion