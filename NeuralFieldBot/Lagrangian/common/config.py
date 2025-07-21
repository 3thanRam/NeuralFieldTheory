# common/config.py
import os
import torch
from .NeuralFieldNetwork import UniversalModel
from .lossfunction import CompositeLoss, ChatbotBaseLoss, StockbotBaseLoss

class BaseConfig:
    def __init__(self, args):
        self.load = args.load
        self.num_epochs = args.epochs
        self.batch_size = 64
        self.embed_dim = 256
        self.outputdir = os.path.join(os.path.dirname(__file__), "..","output")
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.lr = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_blocks = 4
        self.d_hidden_dim = self.embed_dim * 4
        self.dropout = 0.1
        self.damping_factor = 0.99
        print(f"Using device: {self.device}")

class ChatbotConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = "chatbot"
        self.ckpt_path = os.path.join(self.outputdir,".." ,"model_data", "chatbot_checkpoint.pth.tar")
        self.vocab_size = None; self.pad_idx = None
        self.aux_weights = {
            'norm_constraint': 0.1,   # Keep the state from exploding
            'force_minimization': 1e1, # Encourage simple/smooth dynamics
            'force_decorrelation': 1e3  # Encourage the two streams to be different
        }

    def get_model_and_criterion(self):
        model_kwargs = {
            'mode': 'nlp', 'embed_dim': self.embed_dim, 'd_hidden_dim': self.d_hidden_dim,
            'num_blocks': self.num_blocks, 'vocab_size': self.vocab_size, 'pad_idx': self.pad_idx,'damping_factor': self.damping_factor,
            'dropout': self.dropout
        }
        #model = LNN(**model_kwargs)
        model = UniversalModel(**model_kwargs)
        base_criterion = ChatbotBaseLoss(self.pad_idx)
        return model, CompositeLoss(base_criterion, self.aux_weights)

class StockbotConfig(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = "stockbot"
        self.ckpt_path = os.path.join(os.path.dirname(__file__), "..", "model_data", "stockbot_checkpoint.pth.tar")
        self.symbols = ["GLD", "AAPL", "TSLA", "SPY", "TLT","NVDA","AMZN","MSFT","GOOGL","META","AVGO","BRK.B","TSM"]; self.primary_symbol = "GLD"
        self.years_of_data = 5; self.sequence_length = 60; self.val_split_ratio = 0.1
        self.num_input_features = len(self.symbols) * 5
        self.num_output_predictions = self.num_input_features
        self.primary_symbol_idx = self.symbols.index(self.primary_symbol)
        self.huber_weight = 0.7; self.direction_weight = 0.3
        self.aux_weights = {
            'norm_constraint': 0.1,   # Keep the state from exploding
            'force_minimization': 1e1, # Encourage simple/smooth dynamics
            'force_decorrelation': 1e3  # Encourage the two streams to be different
        }

    def get_model_and_criterion(self):
        model_kwargs = {
            'mode': 'timeseries', 'embed_dim': self.embed_dim, 'd_hidden_dim': self.d_hidden_dim,
            'num_blocks': self.num_blocks, 'num_input_features': self.num_input_features,
            'num_output_predictions': self.num_output_predictions, 'dropout': self.dropout,'damping_factor': self.damping_factor
        }
        #model = LNN(**model_kwargs)
        model = UniversalModel(**model_kwargs)
        base_criterion = StockbotBaseLoss(self.primary_symbol_idx, self.huber_weight, self.direction_weight)
        return model, CompositeLoss(base_criterion, self.aux_weights)