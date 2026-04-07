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
    def __init__(self, feature_dim=64, action_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_embed_dim = feature_dim 
        
        # Give each orthogonal plane its own dedicated translator.
        def build_projection_head():
            return nn.Sequential(
                nn.Linear(action_dim, self.action_embed_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.action_embed_dim, self.action_embed_dim),
                nn.Tanh() # to keep the action instructions bounded and smooth for the dynamics engine
            )
            
        self.mlp_xy = build_projection_head()
        self.mlp_xz = build_projection_head()
        self.mlp_yz = build_projection_head()
        
        # ---------------------------------------------------------
        # --- Dynamic Attention Generators ---
        # ---------------------------------------------------------
        def build_attention_generator():
            return nn.Sequential(
                # feature_dim (Visual) + action_embed_dim (Action) + feature_dim (Memory) + 2 (Coords)
                nn.Conv2d((self.feature_dim * 2) + self.action_embed_dim + 2, self.action_embed_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(self.action_embed_dim, self.action_embed_dim, kernel_size=3, padding=1),
                nn.Sigmoid() 
            )
            
        self.attention_gen_xy = build_attention_generator()
        self.attention_gen_xz = build_attention_generator()
        self.attention_gen_yz = build_attention_generator()

        # ==========================================
        # 1. THE SHARED BODY: Universal Physics
        # ==========================================
        self.dynamics_rnn = ConvGRUCell(
            input_dim=self.feature_dim, 
            hidden_dim=self.feature_dim
        )

        self.h0_proj = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        
        # ==========================================
        # 2. THE MULTI-HEAD: Independent Geometric Decoders
        # ==========================================
        def build_plane_head():
            return nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1),
                nn.Tanh() 
            )
            
        self.plane_projs = nn.ModuleDict({
            'xy': build_plane_head(),
            'xz': build_plane_head(),
            'yz': build_plane_head()
        })

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        device = tri_planes_t['xy'].device
        
        # Step 1: Translate the 1D action vector into 3D instructions for each plane
        act_xy_base = self.mlp_xy(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_xz_base = self.mlp_xz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        act_yz_base = self.mlp_yz(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        # Step 2: Initialize hidden states on the first forward pass
        if hidden_states_prev is None:
            hidden_states_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
        
        # ==========================================
        # Cached Dynamic Coordinate Grid
        # ==========================================
        # Check if the grid exists, matches the current dimensions, and is on the correct device.
        # If not (e.g., on the very first step, if resolution changes, or device changes), generate it.
        if not hasattr(self, 'cached_grid') or self.cached_grid.shape[-1] != W or self.cached_grid.shape[-2] != H or self.cached_grid.device != device:
            y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
            x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
            # Save it to the instance so it is never recalculated again
            self.cached_grid = torch.cat([y_coords, x_coords], dim=1)
            
        # Expand the pre-computed grid to match the current Batch size
        batch_coords = self.cached_grid.expand(B, -1, -1, -1)
        
        # Step 3: HYSTERESIS-AWARE & SPATIALLY-ANCHORED DYNAMIC MASKS
        # Now passing 194 channels: Visual(64) + Action(64) + Memory(64) + Coords(2)
        mask_xy = self.attention_gen_xy(torch.cat([tri_planes_t['xy'], act_xy_base, hidden_states_prev['xy'], batch_coords], dim=1))
        mask_xz = self.attention_gen_xz(torch.cat([tri_planes_t['xz'], act_xz_base, hidden_states_prev['xz'], batch_coords], dim=1))
        mask_yz = self.attention_gen_yz(torch.cat([tri_planes_t['yz'], act_yz_base, hidden_states_prev['yz'], batch_coords], dim=1))

        # FiLM (Feature-wise Linear Modulation)
        mod_xy = tri_planes_t['xy'] * (1.0 + (act_xy_base * mask_xy))
        mod_xz = tri_planes_t['xz'] * (1.0 + (act_xz_base * mask_xz))
        mod_yz = tri_planes_t['yz'] * (1.0 + (act_yz_base * mask_yz))

        plane_mods = {'xy': mod_xy, 'xz': mod_xz, 'yz': mod_yz}
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        for plane_key in ['xy', 'xz', 'yz']:
            # All planes route through the SHARED physics engine
            h_new = self.dynamics_rnn(plane_mods[plane_key], hidden_states_prev[plane_key])
            
            # Each plane decodes the physics using its INDEPENDENT head
            delta = self.plane_projs[plane_key](h_new) # Residual Prediction: predict the deformation (delta), which is added to the current frame.
            
            tri_planes_next[plane_key] = tri_planes_t[plane_key] + delta
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new