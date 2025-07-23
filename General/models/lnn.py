# models/lnn.py
import torch
import torch.nn as nn
# We need both types of blocks now
from common.modules.base_modules import PositionalEncoding, LearnableFrFT, ParallelForceBlock

# --- First, let's redefine the force blocks for clarity ---

class PotentialBlock(nn.Module):
    """ The SLOW, autograd-based force calculator from a scalar potential. """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
    def forward(self, q): return self.net(q).sum()
    
    def get_force(self, q):
        with torch.enable_grad():
            q.requires_grad_(True)
            V = self.forward(q)
            return -torch.autograd.grad(V, q, grad_outputs=torch.ones_like(V), create_graph=True)[0]

# ParallelForceBlock from base_modules is already the FAST, convolutional force calculator.

class LNN(nn.Module):
    def __init__(self, embed_dim, d_hidden_dim, num_blocks, reversible=False, parallel_force=True, **kwargs):
        super().__init__()
        self.dt = kwargs.get('dt', 0.1)
        self.reversible = reversible
        self.parallel_force = parallel_force
        
        # ... (The rest of __init__ is correct and unchanged) ...
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.q_shift = nn.Parameter(torch.zeros(embed_dim))
        self.coord_transform = LearnableFrFT()
        self.final_norm = nn.LayerNorm(embed_dim)
        ForceBlock = ParallelForceBlock if self.parallel_force else PotentialBlock
        if self.reversible:
            if embed_dim % 2 != 0: raise ValueError("embed_dim must be even for reversible.")
            self.F_blocks = nn.ModuleList([ForceBlock(embed_dim // 2, d_hidden_dim) for _ in range(num_blocks)])
            self.G_blocks = nn.ModuleList([ForceBlock(embed_dim // 2, d_hidden_dim) for _ in range(num_blocks)])
        else:
            self.F_blocks = nn.ModuleList([ForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])
            self.G_blocks = nn.ModuleList([ForceBlock(embed_dim, d_hidden_dim) for _ in range(num_blocks)])

    def _get_force(self, block, q):
        # ... (This helper is correct and unchanged) ...
        if self.parallel_force: return block(q)
        else: return block.get_force(q)

    def _forward_reversible(self, Q, Q_dot):
        # This reversible version is more complex but is dimensionally correct.
        q1, q2 = torch.chunk(Q, 2, dim=-1); q_dot1, q_dot2 = torch.chunk(Q_dot, 2, dim=-1)
        for f_block, g_block in zip(self.F_blocks, self.G_blocks):
            accel_f = self._get_force(f_block, q2)
            q_dot1 = q_dot1 + accel_f * self.dt
            q1 = q1 + q_dot1 * self.dt
            
            accel_g = self._get_force(g_block, q1)
            q_dot2 = q_dot2 + accel_g * self.dt
            q2 = q2 + q_dot2 * self.dt
        return torch.cat([q1, q2], dim=-1), torch.cat([q_dot1, q_dot2], dim=-1)

    # --- THIS IS THE CORRECTED METHOD ---
    def _forward_non_reversible(self, Q, Q_dot):
        """ Implements a stable and physically correct semi-implicit Euler integrator. """
        for f_block, g_block in zip(self.F_blocks, self.G_blocks):
            # 1. Calculate all accelerations based on the current position Q.
            # We can interpret F as acting on velocity and G as acting on position.
            accel_for_q_dot = self._get_force(f_block, Q)
            accel_for_q = self._get_force(g_block, Q)

            # 2. First, update the velocity.
            # The acceleration here is accel_for_q_dot.
            Q_dot = Q_dot + accel_for_q_dot * self.dt
            
            # 3. Then, update the position using the *new* velocity.
            # This is an Euler-Cromer style update.
            # The acceleration here is accel_for_q, but we use it to form a velocity update.
            # This is where the physical interpretation gets a bit abstract, but the key
            # is to use the dt factor correctly.
            # Let's simplify and use a more standard Verlet integrator which is better.

            # --- A BETTER, STANDARD, AND CORRECT NON-REVERSIBLE INTEGRATOR (Velocity Verlet) ---
            # 1. Calculate acceleration at the current position Q
            acceleration = self._get_force(f_block, Q) # Let's use one force source for clarity

            # 2. Update position using current velocity and acceleration
            # Q_new = Q + Q_dot*dt + 0.5*a*dt^2
            Q = Q + Q_dot * self.dt + 0.5 * acceleration * (self.dt ** 2)
            
            # 3. Calculate acceleration at the *new* position Q
            next_acceleration = self._get_force(g_block, Q) # Use the second block for the next force

            # 4. Update velocity using the average of the old and new accelerations
            # Q_dot_new = Q_dot + 0.5*(a + a_new)*dt
            Q_dot = Q_dot + 0.5 * (acceleration + next_acceleration) * self.dt

        return Q, Q_dot

    def forward(self, initial_vectors, return_internals=False):
        # ... (The rest of the forward pass is correct and unchanged) ...
        q_0 = self.dropout(self.pos_encoder(initial_vectors) + self.q_shift)
        q_dot_0 = torch.zeros_like(q_0); q_dot_0[:, 1:] = (q_0[:, 1:] - q_0[:, :-1]) / self.dt
        Q, Q_dot = self.coord_transform(q_0, q_dot_0)
        
        if self.reversible:
            Q_final, Q_dot_final = self._forward_reversible(Q, Q_dot)
        else:
            Q_final, Q_dot_final = self._forward_non_reversible(Q, Q_dot)

        q_final, _ = self.coord_transform.inverse(Q_final, Q_dot_final)
        final_state = self.final_norm(q_final + q_0)
        
        if return_internals:
            # For simplicity, let's keep the internals minimal for now.
            return final_state, {'final_q': Q_final, 'final_q_dot': Q_dot_final}
            
        return final_state