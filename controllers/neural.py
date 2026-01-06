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
    def __init__(self, model_path=None):
        # Find model
        model_dir = Path(__file__).parent.parent / "models"
        
        if model_path:
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = model_dir / model_path
        else:
            # Try models in order of preference
            for name in ["steer_model_smooth_2.pt", "steer_model.pt", "steer_model_1.pt"]:
                model_path = model_dir / name
                if model_path.exists():
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Neural model not found at {model_path}. "
                "Run: python train_steer_model.py --data_path data --epochs 20"
            )
        
        print(f"Loading neural model: {model_path.name}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model_type = checkpoint.get('model_type', 'transformer')
        hidden = checkpoint.get('hidden', 128)
        
        # Get state dict - handle different checkpoint formats
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise ValueError(f"Unknown checkpoint format, keys: {list(checkpoint.keys())}")
        
        # Auto-detect model type from state_dict keys
        state_keys = set(state_dict.keys())
        if 'past_proj.weight' in state_keys or 'future_proj.weight' in state_keys:
            model_type = 'rl'  # SteerModelRL signature
        
        if model_type == 'mlp':
            self.model = SteerMLP(hidden=hidden)
        elif model_type == 'rl':
            # RL model from train_steer_model_rl.py
            from train_steer_model_rl import SteerModelRL
            self.model = SteerModelRL(d_model=hidden)
            self.is_rl_model = True
        else:
            self.model = SteerTransformer(d_model=hidden)
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.is_rl_model = model_type == 'rl'
        
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
        # For RL model: [v_ego, a_ego, roll, target, measured, steer]
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, self.prev_steer]
        self.context.append(obs)
        
        # Keep only last CONTEXT_LENGTH
        if len(self.context) > CONTEXT_LENGTH:
            self.context = self.context[-CONTEXT_LENGTH:]
        
        # If not enough context yet, use simple proportional control
        if len(self.context) < CONTEXT_LENGTH:
            steer = 0.3 * (target_lataccel - current_lataccel)
            self.prev_steer = np.clip(steer, -2, 2)
            return self.prev_steer
        
        if self.is_rl_model:
            # RL model with separate past/current/future
            # past_ctx: (10, 6) - [vEgo, aEgo, roll, targetLat, measuredLat, steerCmd]
            past_ctx = torch.tensor(self.context, dtype=torch.float32).unsqueeze(0)  # (1, 10, 6)
            
            # current: (4,) - [vEgo, aEgo, roll, measuredLat]
            current = torch.tensor([v_ego, a_ego, roll_lataccel, current_lataccel], 
                                  dtype=torch.float32).unsqueeze(0)  # (1, 4)
            
            # future_ctx: (10, 4) - [vEgo, aEgo, roll, targetLat]
            if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) >= CONTEXT_LENGTH:
                future_ctx = torch.tensor(np.stack([
                    np.array(future_plan.v_ego[:CONTEXT_LENGTH]),
                    np.array(future_plan.a_ego[:CONTEXT_LENGTH]),
                    np.array(future_plan.roll_lataccel[:CONTEXT_LENGTH]),
                    np.array(future_plan.lataccel[:CONTEXT_LENGTH]),
                ], axis=1), dtype=torch.float32).unsqueeze(0)  # (1, 10, 4)
            else:
                # No future plan, use zeros
                future_ctx = torch.zeros(1, CONTEXT_LENGTH, 4)
            
            # Predict (deterministic for inference)
            output = self.model(past_ctx, current, future_ctx, deterministic=True)
            if isinstance(output, tuple):
                steer = output[0].item()  # mean
            else:
                steer = output.item()
        else:
            # Original model interface
            ctx = torch.tensor(self.context, dtype=torch.float32).unsqueeze(0)  # (1, 10, 6)
            cur = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel], 
                              dtype=torch.float32).unsqueeze(0)  # (1, 5)
            steer = self.model(ctx, cur).item()
        
        steer = np.clip(steer, -2, 2)
        self.prev_steer = steer
        return steer

