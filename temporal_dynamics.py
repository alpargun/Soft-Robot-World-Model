import torch
import torch.nn as nn

class ResidualConvGRUCell(nn.Module):
    """
    An upgraded ConvGRU with a residual path to help gradients flow 
    through long 60-frame sequences without vanishing.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        # Gates: Reset (r) and Update (z)
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        # Candidate hidden state (h~)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        # LayerNorm helps stabilize recurrent training on Apple Silicon/MPS
        self.ln = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_can(combined_can))
        
        # Standard GRU update equation: h_t = (1-z) * h_{t-1} + z * h~
        h_new = (1 - update_gate) * h_prev + update_gate * h_candidate
        
        return self.ln(h_new)

class TriPlaneDynamics(nn.Module):
    def __init__(self, feature_dim=32, action_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Action-to-FiLM Generator
        # We generate a Scale (gamma) and Shift (beta) for every feature channel
        self.film_generator = nn.Sequential(
            nn.Linear(action_dim, feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim * 2, feature_dim * 2) # Outputs [gamma, beta]
        )
        
        # 2. Shared Physics Engine
        # We process the modulated planes through a Residual ConvGRU
        self.dynamics_rnn = ResidualConvGRUCell(
            input_dim=feature_dim, 
            hidden_dim=feature_dim
        )

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        """
        Inputs:
            tri_planes_t: Dict {'xy', 'xz', 'yz'} each [B, C, H, W]
            action_t: Pressures [B, 3]
            hidden_states_prev: Dict of previous memory states
        """
        B, C, H, W = tri_planes_t['xy'].shape
        
        # --- FiLM Modulation ---
        # Generate gamma (scale) and beta (shift) from pressures
        # Equation: Output = gamma * Features + beta
        action_params = self.film_generator(action_t) # [B, 2*C]
        gamma, beta = action_params.chunk(2, dim=1)
        
        # Reshape for broadcasting across spatial H, W
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # Initialize hidden states if necessary
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.zeros_like(v) for k, v in tri_planes_t.items()}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        # Process planes
        for plane_key in ['xy', 'xz', 'yz']:
            plane_features = tri_planes_t[plane_key]
            
            # STEP 1: Apply FiLM (This forces the AI to listen to the pressure)
            # If pressures change, the entire feature map is mathematically forced to shift
            modulated_input = (gamma * plane_features) + beta
            
            # STEP 2: Step the Physics RNN
            # The hidden state captures the path (hysteresis)
            h_new = self.dynamics_rnn(modulated_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new