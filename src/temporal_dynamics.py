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
        
        # 1. Plane-Specific Action Embedding 
        # Output is 48 channels (16 for each of the 3 planes)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, self.action_embed_dim * 3),
            nn.LeakyReLU(0.2),
            nn.Linear(self.action_embed_dim * 3, self.action_embed_dim * 3),
            nn.Tanh()
        )
        
        # 2. Physics Engine
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + self.action_embed_dim, # Looks at the geometry and the action simultaneously
            hidden_dim=feature_dim
        )

        # 3. Dynamic Initial State Generator
        self.h0_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        
        # --- Axis-Isolated Concatenation ---
        act_embed = self.action_mlp(action_t) # [B, action_embed_dim*3]
        
        # Slice into three unique 16-channel instruction sets
        act_chunks = act_embed.chunk(3, dim=1) # 3 items of shape [B, action_embed_dim]
        
        # Route specific chunks to specific orthogonal planes
        plane_actions = {
            'xy': act_chunks[0].view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W),
            'xz': act_chunks[1].view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W),
            'yz': act_chunks[2].view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        }
        
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            plane_features = tri_planes_t[plane_key]
            
            # Grab ONLY the action instructions for this specific geometric plane
            act_spatial = plane_actions[plane_key] 
            
            coupled_input = torch.cat([plane_features, act_spatial], dim=1)
            h_new = self.dynamics_rnn(coupled_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new