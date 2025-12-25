"""
Neural steering controller - uses learned model to predict steer_command.

Maintains a sliding window of:
  [v_ego, a_ego, roll_lataccel, target_lataccel, steer_command, measured_lataccel]

At each step, predicts the next steer_command based on context.
"""

from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CONTEXT_LENGTH = 10
ACC_G = 9.81


class SteerMLP(nn.Module):
    def __init__(self, context_len=CONTEXT_LENGTH, context_dim=6, current_dim=5, hidden=128):
        super().__init__()
        input_dim = context_len * context_dim + current_dim
        
        self.register_buffer('ctx_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('ctx_std', torch.tensor([15., 1., 0.5, 1., 0.5, 1.]))
        self.register_buffer('cur_mean', torch.tensor([20., 0., 0., 0., 0.]))
        self.register_buffer('cur_std', torch.tensor([15., 1., 0.5, 1., 1.]))
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        x = torch.cat([context.flatten(1), current], dim=1)
        return self.net(x).squeeze(-1)


class SteerTransformer(nn.Module):
    def __init__(self, context_len=CONTEXT_LENGTH, context_dim=6, current_dim=5, 
                 d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.register_buffer('ctx_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('ctx_std', torch.tensor([15., 1., 0.5, 1., 0.5, 1.]))
        self.register_buffer('cur_mean', torch.tensor([20., 0., 0., 0., 0.]))
        self.register_buffer('cur_std', torch.tensor([15., 1., 0.5, 1., 1.]))
        
        self.context_proj = nn.Linear(context_dim, d_model)
        self.current_proj = nn.Linear(current_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(context_len + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        
        ctx_emb = self.context_proj(context)
        cur_emb = self.current_proj(current).unsqueeze(1)
        seq = torch.cat([ctx_emb, cur_emb], dim=1)
        seq = seq + self.pos_emb
        out = self.transformer(seq)
        return self.head(out[:, -1]).squeeze(-1)


class Controller(BaseController):
    def __init__(self):
        # Try models in order of preference
        model_dir = Path(__file__).parent.parent / "models"
        for name in ["steer_model_smooth_2.pt", "steer_model.pt", "steer_model_1.pt"]:
            model_path = model_dir / name
            if model_path.exists():
                break
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Neural model not found. "
                "Run: python train_steer_model.py --data_path data --epochs 20"
            )
        
        print(f"Loading neural model: {model_path.name}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model_type = checkpoint.get('model_type', 'transformer')
        hidden = checkpoint.get('hidden', 128)
        
        if model_type == 'mlp':
            self.model = SteerMLP(hidden=hidden)
        else:
            self.model = SteerTransformer(d_model=hidden)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Context buffer: (v_ego, a_ego, roll_lataccel, target_lataccel, steer_command, measured_lataccel)
        self.context = []
        self.prev_steer = 0.0
    
    @torch.no_grad()
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build current observation
        roll_lataccel = state.roll_lataccel
        v_ego = state.v_ego
        a_ego = state.a_ego
        
        # Add to context (using previous steer command and current measured lataccel)
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, self.prev_steer, current_lataccel]
        self.context.append(obs)
        
        # Keep only last CONTEXT_LENGTH
        if len(self.context) > CONTEXT_LENGTH:
            self.context = self.context[-CONTEXT_LENGTH:]
        
        # If not enough context yet, use simple proportional control
        if len(self.context) < CONTEXT_LENGTH:
            steer = 0.3 * (target_lataccel - current_lataccel)
            self.prev_steer = np.clip(steer, -2, 2)
            return self.prev_steer
        
        # Prepare inputs
        ctx = torch.tensor(self.context, dtype=torch.float32).unsqueeze(0)  # (1, 10, 6)
        cur = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel], 
                          dtype=torch.float32).unsqueeze(0)  # (1, 5)
        
        # Predict
        steer = self.model(ctx, cur).item()
        steer = np.clip(steer, -2, 2)
        
        self.prev_steer = steer
        return steer

