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

class DynamicsTriplane(nn.Module):
    def __init__(self, feature_dim=64, action_dim=3, action_embed_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_embed_dim = action_embed_dim
        
        # Give each orthogonal plane its own dedicated head network.
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.Tanh() # to keep the action instructions bounded and smooth for the dynamics engine
        )
        
        # 2. Shared Physics Engine
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + action_embed_dim, 
            hidden_dim=action_embed_dim
        )

        # Projects visual planes to memory for initialization
        self.h0_proj = nn.Conv2d(feature_dim, action_embed_dim, kernel_size=1)
        
        # Projects memory to visual planes for the decoder
        self.feature_proj = nn.Conv2d(action_embed_dim, feature_dim, kernel_size=1)

        torch.nn.init.zeros_(self.feature_proj.weight)
        torch.nn.init.zeros_(self.feature_proj.bias)
        
        # Add visual dropout to encourage the model to use the hidden state for memory
        self.memory_dropout = nn.Dropout2d(p=0.20)
        
        # Sequence inverse dynamics head
        self.history_len = 5 # Number of past actions to predict to account for hysteresis
        
        self.inverse_head = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 32, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            # Output dimension is now (3 pressures * 5 frames) = 15
            nn.Linear(32, action_dim * self.history_len),
            nn.Sigmoid()
        )

    def forward(self, tri_planes_t, action_t, hidden_prev=None):
        B, C, H, W = tri_planes_t['xy'].shape
        act_emb = self.action_mlp(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        if hidden_prev is None:
            hidden_prev = {k: torch.tanh(self.h0_proj(v)) for k, v in tri_planes_t.items()}
            
        tri_planes_next = {}
        hidden_new = {}
        
        for key in ['xy', 'xz', 'yz']:
            coupled_input = torch.cat([tri_planes_t[key], act_emb], dim=1)
            
            # Use dropout to encourage reliance on the action tensor
            h_prev_dropped = self.memory_dropout(hidden_prev[key])
            
            h_new = self.dynamics_rnn(coupled_input, h_prev_dropped)
            
            # Predict the change, and add it to the current state.
            delta_plane = self.feature_proj(h_new)
            tri_planes_next[key] = tri_planes_t[key] + delta_plane
            hidden_new[key] = h_new
            
        return tri_planes_next, hidden_new

    def predict_inverse_action_sequence(self, planes_t, planes_next):
        """Forces the network to deduce the action history from the physical change."""
        preds = []
        for key in ['xy', 'xz', 'yz']:
            coupled_state = torch.cat([planes_t[key], planes_next[key]], dim=1)
            preds.append(self.inverse_head(coupled_state))
            
        combined_pred = sum(preds) / 3.0 # Consensus of the 3 planes
        return combined_pred.view(planes_t['xy'].shape[0], self.history_len, -1) # [Batch, History_Len, Action_Dim]