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
    def __init__(self, feature_dim=64, action_dim=3, action_embed_dim=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_embed_dim = action_embed_dim
        
        # Give each orthogonal plane its own dedicated translator.
        # Each one looks at ALL 3 pressures (P1, P2, P3) and figures out 
        # what that specific plane needs to do to maintain the 3D volume.
        
        def build_projection_head():
            return nn.Sequential(
                nn.Linear(action_dim, action_embed_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(action_embed_dim, action_embed_dim),
                nn.Tanh() # to keep the action instructions bounded and smooth for the dynamics engine
            )
            
        self.mlp_xy = build_projection_head()
        self.mlp_xz = build_projection_head()
        self.mlp_yz = build_projection_head()
        
        # 2. Shared Physics Engine
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + action_embed_dim, 
            hidden_dim=feature_dim
        )

        self.h0_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        
        # Generate custom instructions for each plane using the FULL action vector
        act_xy = self.mlp_xy(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_xz = self.mlp_xz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_yz = self.mlp_yz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        plane_actions = {'xy': act_xy, 'xz': act_xz, 'yz': act_yz}
        
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            coupled_input = torch.cat([tri_planes_t[plane_key], plane_actions[plane_key]], dim=1)
            h_new = self.dynamics_rnn(coupled_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new