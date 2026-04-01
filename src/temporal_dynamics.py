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
        
    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_can(combined_can))
        
        # GRU update: h_t = (1-z) * h_{t-1} + z * h~
        h_new = (1 - update_gate) * h_prev + update_gate * h_candidate
        
        return h_new

class TriPlaneDynamics(nn.Module):
    def __init__(self, feature_dim=128, action_dim=3, action_embed_dim=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        
        # 1. Action Embedding 
        # Processes the 3 raw pressure values into a richer feature representation
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, self.action_embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.action_embed_dim, self.action_embed_dim),
            nn.Tanh()
        )
        
        # 2. Hard-Coupled Physics Engine
        # The ConvGRU is forced to look at the geometry and the action simultaneously
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + self.action_embed_dim, 
            hidden_dim=feature_dim
        )

        # 3. Dynamic Initial State Generator
        self.h0_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        """
        Inputs:
            tri_planes_t: Dict {'xy', 'xz', 'yz'} each [B, C, H, W]
            action_t: Pressures [B, 3]
            hidden_states_prev: Dict of previous memory states
        """
        B, C, H, W = tri_planes_t['xy'].shape
        
        # --- Spatial Action Concatenation ---
        # 1. Embed the raw pressures
        act_embed = self.action_mlp(action_t) # [B, action_embed_dim]
        
        # 2. Expand the 1D embedding into a 2D spatial grid [B, action_embed_dim, H, W]
        act_spatial = act_embed.view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        if hidden_states_prev is None:
            hidden_states_prev = {
                k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()
            }
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            plane_features = tri_planes_t[plane_key]
            
            # STEP 1: The Hard-Coupling Glue
            # We explicitly stack the geometry and the pressure together along the channel dimension.
            # Shape becomes [B, C + action_embed_dim, H, W]
            coupled_input = torch.cat([plane_features, act_spatial], dim=1)
            
            # STEP 2: Step the Physics RNN
            h_new = self.dynamics_rnn(coupled_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new