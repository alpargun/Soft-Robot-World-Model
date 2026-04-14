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
    def __init__(self, feature_dim=64, action_dim=3, action_embed_dim=32, spatial_size=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_embed_dim = action_embed_dim
        
        # --- MULTIPLICATIVE GATE ---
        # Initialized to 1.0 so gradients don't vanish, forcing the network 
        # to use this as a strict routing map for the pressure values.
        self.spatial_attention_xy = nn.Parameter(torch.ones(1, action_embed_dim, spatial_size, spatial_size))
        self.spatial_attention_xz = nn.Parameter(torch.ones(1, action_embed_dim, spatial_size, spatial_size))
        self.spatial_attention_yz = nn.Parameter(torch.ones(1, action_embed_dim, spatial_size, spatial_size))

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
            hidden_dim=action_embed_dim
        )

        # Projects 64-dim visual planes down to 32-dim memory for initialization
        self.h0_proj = nn.Conv2d(feature_dim, action_embed_dim, kernel_size=1)
        
        # Projects 32-dim memory up to 64-dim visual planes for the decoder
        self.plane_proj = nn.Conv2d(action_embed_dim, feature_dim, kernel_size=1)

        # Add visual dropout to encourage the model to use the hidden state for memory, not just the visual features
        self.memory_dropout = nn.Dropout2d(p=0.20)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        
        act_xy = self.mlp_xy(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_xz = self.mlp_xz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_yz = self.mlp_yz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)

        # Constrain the attention parameters to [0, 1]
        att_xy = torch.sigmoid(self.spatial_attention_xy)
        att_xz = torch.sigmoid(self.spatial_attention_xz)
        att_yz = torch.sigmoid(self.spatial_attention_yz)
        
        # Apply bounded routing
        act_xy = act_xy * att_xy
        act_xz = act_xz * att_xz
        act_yz = act_yz * att_yz

        plane_actions = {'xy': act_xy, 'xz': act_xz, 'yz': act_yz}
        
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            coupled_input = torch.cat([tri_planes_t[plane_key], plane_actions[plane_key]], dim=1)
            
            # Degrade the memory to force reliance on the action tensor
            h_prev_dropped = self.memory_dropout(hidden_states_prev[plane_key])
            
            h_new = self.dynamics_rnn(coupled_input, h_prev_dropped)
            
            # Decouple the visual plane from the bounded hidden state
            tri_planes_next[plane_key] = self.plane_proj(h_new)
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new