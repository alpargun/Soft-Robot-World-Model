import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        # Gates: Reset (r) and Update (z)
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        # Candidate hidden state (h~)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        self.ln = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_can(combined_can))
        
        # GRU update: h_t = (1-z) * h_{t-1} + z * h~
        h_new = (1 - update_gate) * h_prev + update_gate * h_candidate
        
        return self.ln(h_new)

class TriPlaneDynamics(nn.Module):
    def __init__(self, feature_dim=32, action_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Action-to-FiLM Generator (DECOUPLED)
        # Generate 6 sets of parameters: [gamma_xy, beta_xy, gamma_xz, beta_xz, gamma_yz, beta_yz]
        self.film_generator = nn.Sequential(
            nn.Linear(action_dim, feature_dim * 4), # Widened hidden layer to support larger output
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim * 4, feature_dim * 6) # Outputs 6 * C
        )
        
        # 2. Shared Physics Engine
        # The ConvGRU still shares weights across planes to maintain translation invariance
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim, 
            hidden_dim=feature_dim
        )

        # 3. Dynamic Initial State Generator
        # Projects the visual resting frame into a physical resting memory state
        self.h0_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        """
        Inputs:
            tri_planes_t: Dict {'xy', 'xz', 'yz'} each [B, C, H, W]
            action_t: Pressures [B, 3]
            hidden_states_prev: Dict of previous memory states
        """
        B, C, H, W = tri_planes_t['xy'].shape
        
        # --- Decoupled FiLM Modulation ---
        action_params = self.film_generator(action_t) # [B, 6*C]
        
        # Chunk into 6 distinct parameter blocks
        params = action_params.chunk(6, dim=1)
        
        # Map specific gammas and betas to their respective planes and reshape for spatial broadcasting
        film_params = {
            'xy': (params[0].view(B, C, 1, 1), params[1].view(B, C, 1, 1)),
            'xz': (params[2].view(B, C, 1, 1), params[3].view(B, C, 1, 1)),
            'yz': (params[4].view(B, C, 1, 1), params[5].view(B, C, 1, 1))
        }
        
        # Initialize hidden states if not provided (i.e., at the first time step)
        if hidden_states_prev is None:
            # Use tanh to keep the memory state bounded [-1, 1] like the GRU expects
            hidden_states_prev = {
                k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()
            }
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        # Process planes
        for plane_key in ['xy', 'xz', 'yz']:
            plane_features = tri_planes_t[plane_key]
            
            # Retrieve the specific affine shifts for THIS orthogonal plane
            gamma, beta = film_params[plane_key]
            
            # STEP 1: Apply Directional FiLM
            modulated_input = (gamma * plane_features) + beta
            
            # STEP 2: Step the Physics RNN
            h_new = self.dynamics_rnn(modulated_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new