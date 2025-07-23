# common/modules/reversible_modules.py
import torch
import torch.nn as nn
from .base_modules import ParallelForceBlock

class ReversibleLNNBlock(nn.Module):
    """
    A reversible dynamics block using the RevNet architecture.
    It uses two ParallelForceBlocks (F and G) to update the two halves of the state.
    """
    def __init__(self, embed_dim, hidden_dim, dt=0.1):
        super().__init__()
        self.dt = dt
        # The F and G functions are our parallel force calculators
        self.F = ParallelForceBlock(embed_dim // 2, hidden_dim)
        self.G = ParallelForceBlock(embed_dim // 2, hidden_dim)

    def forward(self, q1, q2, q_dot1, q_dot2):
        """ The sequential (Gauss-Seidel) update that guarantees reversibility. """
        # Update first half based on the original second half
        accel_f = self.F(q2)
        q_dot1_new = q_dot1 + accel_f * self.dt
        q1_new = q1 + q_dot1_new * self.dt
        
        # Update second half based on the NEW first half
        accel_g = self.G(q1_new)
        q_dot2_new = q_dot2 + accel_g * self.dt
        q2_new = q2 + q_dot2_new * self.dt
        
        return q1_new, q2_new, q_dot1_new, q_dot2_new

    def inverse(self, y1, y2, y_dot1, y_dot2):
        """ The exact algebraic inverse of the forward pass. """
        # Invert the second step first
        accel_g = self.G(y1)
        q2_intermediate = y2 - y_dot2 * self.dt
        q_dot2_intermediate = y_dot2 - accel_g * self.dt
        
        # Invert the first step second
        accel_f = self.F(q2_intermediate)
        q1_intermediate = y1 - y_dot1 * self.dt
        q_dot1_intermediate = y_dot1 - accel_f * self.dt
        
        return q1_intermediate, q2_intermediate, q_dot1_intermediate, q_dot2_intermediate