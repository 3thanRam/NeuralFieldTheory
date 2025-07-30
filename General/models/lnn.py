# models/lnn.py
import torch
import torch.nn as nn
from common.modules.base_modules import PositionalEncoding, LearnableFrFT, ParallelForceBlock,MultiHeadForceBlock

# --- This is the missing PotentialBlock class ---
class PotentialBlock(nn.Module):
    """
    The SLOW, autograd-based force calculator from a single scalar potential.
    This is the non-parallel alternative to ParallelForceBlock.
    """
    def __init__(self, embed_dim, hidden_dim, **kwargs): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar V
        )
    
    def forward(self, q):
        # This returns the total potential energy of the system, a single scalar.
        return self.net(q).sum()
    
    def get_force(self, q):
        """ Calculates the force F = -∇V using automatic differentiation. """
        # method called when parallel_force=False
        if not q.requires_grad:
            q = q.requires_grad_(True)
        V = self.forward(q)
        force = -torch.autograd.grad(V, q, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        return force

# --- The LNN class, now complete ---
class LNN(nn.Module):
    def __init__(self, embed_dim, d_hidden_dim, num_blocks, reversible=False, parallel_force=True, **kwargs):
        super().__init__()
        self.dt = kwargs.get('dt', 0.1)
        self.reversible = reversible
        self.parallel_force = parallel_force
        
        # --- Shared Components ---
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.q_shift = nn.Parameter(torch.zeros(embed_dim))
        self.coord_transform = LearnableFrFT(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- Dynamically Build Force Blocks ---
        kernel_size = kwargs.get('kernel_size', 3) 
        num_heads = kwargs.get('num_heads', 3) 
        # --- Dynamically Build Force Blocks ---
        if self.parallel_force:
            # Pass kernel_size to the constructor
            ForceBlock = lambda dim, d_hidden_dim: MultiHeadForceBlock(num_heads,dim, d_hidden_dim, kernel_size=kernel_size,dropout_rate=kwargs.get('dropout', 0.1))
        else:
            ForceBlock = lambda dim, d_hidden_dim: PotentialBlock(dim, d_hidden_dim)
        
        # Pass hidden_dim to the block constructors
        if self.reversible:
            if embed_dim % 2 != 0: raise ValueError("embed_dim must be even for reversible.")
            self.F_blocks = nn.ModuleList([ForceBlock(embed_dim // 2, d_hidden_dim) for _ in range(num_blocks)])
            self.G_blocks = nn.ModuleList([ForceBlock(embed_dim // 2, d_hidden_dim) for _ in range(num_blocks)])
        else:
            self.F_blocks = nn.ModuleList([ForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])
            self.G_blocks = nn.ModuleList([ForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])

    def _get_force(self, block, q):
        """ Helper to abstract away the force calculation method. """
        if self.parallel_force:
            return block(q) # ParallelForceBlock returns force directly
        else: # PotentialBlock
            return block.get_force(q) # We need to call its .get_force() method

    def _forward_reversible(self, Q, Q_dot, return_internals):
        q1, q2 = torch.chunk(Q, 2, dim=-1)
        q_dot1, q_dot2 = torch.chunk(Q_dot, 2, dim=-1)
        
        internals = {'forces_f': [], 'forces_g': [], 'round_trip_loss': []}

        for f_block, g_block in zip(self.F_blocks, self.G_blocks):
           
            accel_f = self._get_force(f_block, q2)
            # Update position q1
            q1_new = q1 + q_dot1 * self.dt + 0.5 * accel_f * (self.dt ** 2)

            accel_f = self._get_force(f_block, q2)
            q_dot1_update = accel_f * self.dt
            q1_update = q_dot1_update * self.dt
            
            # Additive update
            q1_new = q1 + q1_update
            q_dot1_new = q_dot1 + q_dot1_update

            # --- Second half of the RevNet update (G) ---
            accel_g = self._get_force(g_block, q1_new)
            q_dot2_update = accel_g * self.dt
            q2_update = q_dot2_update * self.dt
            
            # Additive update
            q2_new = q2 + q2_update
            q_dot2_new = q_dot2 + q_dot2_update

            # Update state for the next loop
            q1, q2, q_dot1, q_dot2 = q1_new, q2_new, q_dot1_new, q_dot2_new

            if return_internals:
                internals['forces_f'].append(accel_f)
                internals['forces_g'].append(accel_g)
                # Round trip loss is not well-defined here
                internals['round_trip_loss'].append(torch.tensor(0.))

        Q_final, Q_dot_final = torch.cat([q1, q2], -1), torch.cat([q_dot1, q_dot2], -1)
        return Q_final, Q_dot_final, internals

    def _forward_non_reversible(self, Q, Q_dot, return_internals):
        internals = {'forces_f': [], 'forces_g': [], 'round_trip_loss': []}
        for f_block, g_block in zip(self.F_blocks, self.G_blocks):
            accel_f = self._get_force(f_block, Q)
            if return_internals:
                Q_probed = Q.detach() + accel_f.detach() * self.dt
                accel_g_probe = self._get_force(g_block, Q_probed)
                rtl = (accel_f.detach() + accel_g_probe).pow(2).mean()
                internals['round_trip_loss'].append(rtl)
                internals['forces_f'].append(accel_f)
            
            acceleration = accel_f
            Q = Q + Q_dot * self.dt + 0.5 * acceleration * (self.dt ** 2)
            next_acceleration = self._get_force(g_block, Q)
            Q_dot = Q_dot + 0.5 * (acceleration + next_acceleration) * self.dt
            if return_internals: internals['forces_g'].append(next_acceleration) 
        return Q, Q_dot, internals

    def forward(self, initial_vectors, return_internals=False):
        q_0 = self.dropout(self.pos_encoder(initial_vectors) + self.q_shift)
        q_dot_0 = torch.zeros_like(q_0); q_dot_0[:, 1:] = (q_0[:, 1:] - q_0[:, :-1]) / self.dt
        Q, Q_dot = self.coord_transform(q_0, q_dot_0)
        
        if self.reversible:
            Q_final, Q_dot_final, loop_internals = self._forward_reversible(Q, Q_dot, return_internals)
        else:
            Q_final, Q_dot_final, loop_internals = self._forward_non_reversible(Q, Q_dot, return_internals)

        q_final, _ = self.coord_transform.inverse(Q_final, Q_dot_final)
        final_state = self.final_norm(q_final + q_0)
        
        if return_internals:
            internals = {
                'final_q': Q_final, 'final_q_dot': Q_dot_final,
                'forces_f': loop_internals['forces_f'], 'forces_g': loop_internals['forces_g'],
                'round_trip_loss': torch.mean(torch.stack(loop_internals['round_trip_loss'])) if loop_internals['round_trip_loss'] else torch.tensor(0.)
            }
            return final_state, internals
            
        return final_state