import torch
import torch.nn as nn
import torch.nn.functional as F



class GaugeFieldGenerator(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim * embed_dim)
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)

    def forward(self, phi_i, phi_j):
        combined = torch.cat([phi_i, phi_j], dim=-1)
        elements = self.mlp(combined)
        return elements.view(-1, self.embed_dim, self.embed_dim)

class GaugeConvolutionBlock(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.gauge_generator = GaugeFieldGenerator(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q):
        phi = q
        batch_size, seq_len, embed_dim = phi.shape
        
        # --- Create all pairs ---
        # This part is correct
        phi_i_pairs = phi.unsqueeze(2).expand(-1, -1, seq_len, -1)
        phi_j_pairs = phi.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # --- Generate all U_ij matrices ---
        # This part is correct
        U_all_flat = self.gauge_generator(
            phi_i_pairs.reshape(-1, 1, self.embed_dim),
            phi_j_pairs.reshape(-1, 1, self.embed_dim)
        )
        U_all = U_all_flat.view(batch_size, seq_len, seq_len, self.embed_dim, self.embed_dim)

        # --- Transport and Aggregate Information (THE FIX IS HERE) ---
        
        # OLD, INCORRECT EINSUM:
        # transported_phi = torch.einsum('bnjde,bnjd->bnie', U_all, phi_j_pairs)
        # aggregated_info = torch.sum(transported_phi, dim=2)

        # NEW, CORRECTED EINSUM:
        # This single line performs the matrix-vector product for all pairs
        # and sums over the 'j' dimension simultaneously.
        # 'bijde' is U_all, where i and j are seq dims, d and e are embed dims
        # 'bjd' is phi, where j is the seq dim to be multiplied and summed over
        # 'bie' is the output, the aggregated info for each token i
        aggregated_info = torch.einsum('bijde,bjd->bie', U_all, phi)
        # The output 'aggregated_info' will correctly have the shape (B, N, D)
        
        # --- The rest of the block remains the same ---
        phi = self.norm1(phi + self.dropout(aggregated_info))
        mlp_output = self.mlp(phi)
        q_final = self.norm2(phi + self.dropout(mlp_output))
        return q_final