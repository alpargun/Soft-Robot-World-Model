import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        
        combined_can = torch.cat([x, reset_gate * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_can(combined_can))
        
        h_new = (1 - update_gate) * h_prev + update_gate * h_candidate
        return h_new

class Dynamics2D(nn.Module):
    # THE FIX: Increased action_embed_dim to 64. 
    # This doubles the ConvGRU hidden state capacity, preventing the fading memory leak.
    def __init__(self, feature_dim=64, action_dim=3, action_embed_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_embed_dim = action_embed_dim
        
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.Tanh()
        )
        
        self.dynamics_rnn = ConvGRUCell(
            input_dim=feature_dim + action_embed_dim, 
            hidden_dim=action_embed_dim
        )

        self.h0_proj = nn.Conv2d(feature_dim, action_embed_dim, kernel_size=1)
        self.feature_proj = nn.Conv2d(action_embed_dim, feature_dim, kernel_size=1)
        self.memory_dropout = nn.Dropout2d(p=0.20)
        
        self.history_len = 5
        self.inverse_head = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 32, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, action_dim * self.history_len),
            nn.Sigmoid()
        )

    def forward(self, features_t, action_t, hidden_prev=None):
        B, C, H, W = features_t.shape
        
        act_emb = self.action_mlp(action_t).view(B, self.action_embed_dim, 1, 1).expand(B, self.action_embed_dim, H, W)
        
        if hidden_prev is None:
            hidden_prev = torch.tanh(self.h0_proj(features_t))
            
        coupled_input = torch.cat([features_t, act_emb], dim=1)
        h_prev_dropped = self.memory_dropout(hidden_prev)
        
        h_new = self.dynamics_rnn(coupled_input, h_prev_dropped)
        
        # PURE SINGLEVIEW5 LOGIC: Direct decoding from the newly expanded GRU memory.
        features_next = self.feature_proj(h_new)
        
        return features_next, h_new

    def predict_inverse_action_sequence(self, features_t, features_next):
        coupled_state = torch.cat([features_t, features_next], dim=1)
        pred_seq = self.inverse_head(coupled_state)
        return pred_seq.view(coupled_state.shape[0], self.history_len, -1)