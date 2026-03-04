import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    """ A standard ConvGRU Cell to process spatial features over time. """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        # Gates: Reset and Update (Z and R)
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        # Candidate hidden state
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_can(combined_can))
        
        h_new = (1 - update_gate) * h_prev + update_gate * h_candidate
        return h_new

class TriPlaneDynamics(nn.Module):
    def __init__(self, feature_dim=32, action_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Action Projector: Maps the [B, 3] pressures to a spatial channel dimension
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Shared ConvGRU for all three planes to enforce symmetric physics rules
        # Input dim = plane features + action features
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim * 2, 
            hidden_dim=feature_dim
        )

    def forward(self, tri_planes_t, action_t, hidden_states_prev=None):
        """
        Steps the world model forward by one time step.
        Inputs:
            tri_planes_t: Dict of 'xy', 'xz', 'yz' planes at time t. Shape: [B, feature_dim, H, W]
            action_t: Pressures at time t. Shape: [B, 3]
            hidden_states_prev: Dict of previous memory states.
        Outputs:
            tri_planes_next: Predicted planes at t+1.
            hidden_states_new: Updated memory states.
        """
        B, C, H, W = tri_planes_t['xy'].shape
        
        # 1. Spatially expand the 1D action to match the 2D plane dimensions
        action_embedded = self.action_mlp(action_t) # [B, feature_dim]
        action_spatial = action_embedded.view(B, C, 1, 1).expand(B, C, H, W) # [B, feature_dim, H, W]
        
        # Initialize hidden states at t=0 if not provided
        if hidden_states_prev is None:
            hidden_states_prev = {
                'xy': torch.zeros_like(tri_planes_t['xy']),
                'xz': torch.zeros_like(tri_planes_t['xz']),
                'yz': torch.zeros_like(tri_planes_t['yz'])
            }
            
        tri_planes_next = {}
        hidden_states_new = {}
        
        # 2. Process each plane through the physics dynamics
        for plane_key in ['xy', 'xz', 'yz']:
            plane_features = tri_planes_t[plane_key]
            
            # Fuse the physical state with the current pressure commands
            fused_input = torch.cat([plane_features, action_spatial], dim=1) # [B, feature_dim*2, H, W]
            
            # Predict the next spatial state
            h_new = self.dynamics_rnn(fused_input, hidden_states_prev[plane_key])
            
            tri_planes_next[plane_key] = h_new
            hidden_states_new[plane_key] = h_new
            
        return tri_planes_next, hidden_states_new

# --- Quick Dimension Check ---
if __name__ == "__main__":
    B, C, H, W = 2, 32, 128, 128
    dummy_planes = {
        'xy': torch.randn(B, C, H, W),
        'xz': torch.randn(B, C, H, W),
        'yz': torch.randn(B, C, H, W)
    }
    dummy_action = torch.randn(B, 3) # Pressures for this exact frame
    
    dynamics_module = TriPlaneDynamics(feature_dim=C, action_dim=3)
    
    next_planes, next_hidden = dynamics_module(dummy_planes, dummy_action)
    print(f"Predicted XY Plane Shape at t+1: {next_planes['xy'].shape}") # Expected: [B, feature_dim, H, W]
    print(f"Predicted XZ Plane Shape at t+1: {next_planes['xz'].shape}") # Expected: [B, feature_dim, H, W]
    print(f"Predicted YZ Plane Shape at t+1: {next_planes['yz'].shape}")