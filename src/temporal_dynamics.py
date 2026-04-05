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
    def __init__(self, feature_dim=64, action_dim=3, action_embed_dim=32):
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
        
        # ---------------------------------------------------------
        # --- Dynamic Attention Generators ---
        # Instead of static parameters, these mini-networks look at the 
        # CURRENT visual shape and generate a custom mask for this exact frame.
        # ---------------------------------------------------------
        def build_attention_generator():
            return nn.Sequential(
                # feature_dim (Visual) + action_embed_dim (Action) + feature_dim (Memory) + 2 (X, Y Coords)
                nn.Conv2d((feature_dim * 2) + action_embed_dim + 2, action_embed_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(action_embed_dim, action_embed_dim, kernel_size=3, padding=1),
                nn.Sigmoid() 
            )
            
        self.attention_gen_xy = build_attention_generator()
        self.attention_gen_xz = build_attention_generator()
        self.attention_gen_yz = build_attention_generator()

        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + action_embed_dim, 
            hidden_dim=feature_dim
        )

        self.h0_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # Separates the clamped hidden state memory from the unbounded visual features
        self.plane_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # Pre-compute the 32x32 spatial coordinate grid and save it as a model buffer.
        # This automatically moves to the correct device and prevents recreation overhead.
        spatial_dim = 32
        y_coords = torch.linspace(-1, 1, spatial_dim).view(1, 1, spatial_dim, 1).expand(1, 1, spatial_dim, spatial_dim)
        x_coords = torch.linspace(-1, 1, spatial_dim).view(1, 1, 1, spatial_dim).expand(1, 1, spatial_dim, spatial_dim)
        coord_grid = torch.cat([y_coords, x_coords], dim=1) # Shape: [1, 2, 32, 32]
        self.register_buffer('coord_grid', coord_grid)

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        
        # Step 1: Translate the 1D action vector into 3D instructions
        act_xy_base = self.mlp_xy(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_xz_base = self.mlp_xz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_yz_base = self.mlp_yz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        # Step 2: Initialize hidden states on the first forward pass using the initial visual state as a reference
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
        
        # Expand the pre-computed grid to match the current Batch size
        batch_coords = self.coord_grid.expand(B, -1, -1, -1)
        
        # Step 3: HYSTERESIS-AWARE & SPATIALLY-ANCHORED DYNAMIC MASKS
        # Now passing 162 channels: Visual(64) + Action(32) + Memory(64) + Coords(2)
        mask_xy = self.attention_gen_xy(torch.cat([tri_planes_t['xy'], act_xy_base, hidden_states_prev['xy'], batch_coords], dim=1))
        mask_xz = self.attention_gen_xz(torch.cat([tri_planes_t['xz'], act_xz_base, hidden_states_prev['xz'], batch_coords], dim=1))
        mask_yz = self.attention_gen_yz(torch.cat([tri_planes_t['yz'], act_yz_base, hidden_states_prev['yz'], batch_coords  ], dim=1))

        # Step 4: Apply Masks
        act_xy = act_xy_base * mask_xy
        act_xz = act_xz_base * mask_xz
        act_yz = act_yz_base * mask_yz

        plane_actions = {'xy': act_xy, 'xz': act_xz, 'yz': act_yz}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            coupled_input = torch.cat([tri_planes_t[plane_key], plane_actions[plane_key]], dim=1)
            h_new = self.dynamics_rnn(coupled_input, hidden_states_prev[plane_key])
            
            # Decouple the visual plane from the bounded hidden state
            tri_planes_next[plane_key] = self.plane_proj(h_new)
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new