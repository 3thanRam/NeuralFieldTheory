# sampler.py

import torch
from tqdm import tqdm

class DPMSolver:
    """
    A fast ODE solver for score-based models (DPM-Solver++).
    This sampler is used for inference/generation ONLY.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def generate(self, batch_size, seq_len, num_steps, tokenizer=None):
        self.model.eval()
        
        # --- 1. Define the noise schedule ---
        # A simple linear noise schedule is a good starting point.
        betas = torch.linspace(1e-4, 0.02, 1000, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # --- 2. Initialize with pure noise ---
        # Start from a sample from the prior distribution (a standard normal)
        x_t = torch.randn(
            batch_size, seq_len, self.config['embed_dim'], device=self.device
        )
        
        # --- 3. The DPM-Solver++ sampling loop ---
        timesteps = torch.linspace(999, 0, num_steps + 1, device=self.device).long()
        
        pbar = tqdm(range(num_steps), desc="Generating with DPM-Solver++")
        for i in pbar:
            t = timesteps[i]
            t_prev = timesteps[i+1]
            
            # The current state
            lambda_t = torch.log(alphas_cumprod[t]**0.5 / (1 - alphas_cumprod[t])**0.5)
            lambda_t_prev = torch.log(alphas_cumprod[t_prev]**0.5 / (1 - alphas_cumprod[t_prev])**0.5) if t_prev >= 0 else -torch.inf
            
            # Predict the noise (which is related to the score)
            with torch.no_grad():
                predicted_noise = self.model(x_t, t.repeat(batch_size))
            
            # DPM-Solver++ 2nd order update step
            h = lambda_t_prev - lambda_t
            r = h / 2.0
            
            s = t.repeat(batch_size)
            s = s - 1.0 / r if r != 0 else s
            
            with torch.no_grad():
                D_r = (predicted_noise - self.model(x_t * torch.exp(r) + (torch.exp(r) - 1) * predicted_noise, s)) / r if r != 0 else torch.zeros_like(x_t)
            
            x_t = torch.exp(h) * x_t - (torch.exp(h) - 1) * (predicted_noise + D_r)
        
        # --- 4. Discretize the final state (Unchanged) ---
        embedding_matrix = self.model.input_head.weight
        q_flat = x_t.view(-1, self.config['embed_dim'])
        distances = torch.cdist(q_flat, embedding_matrix)
        final_token_ids = torch.argmin(distances, dim=1).view(batch_size, seq_len)
        
        if tokenizer:
            return [tokenizer.decode(ids, skip_special_tokens=True) for ids in final_token_ids]
        return final_token_ids