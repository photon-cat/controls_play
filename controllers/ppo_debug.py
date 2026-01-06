"""Controller using the simple MLP policy from ppo_debug training."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Use simple base class instead of importing from controllers
class BaseController:
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        raise NotImplementedError

CONTEXT_LENGTH = 10
LOOKAHEAD_LENGTH = 10
STEER_RANGE = 2.0


class SteerPolicy(nn.Module):
    """Simple MLP policy - same architecture as train_ppo_debug.py"""
    def __init__(self, hidden=256):
        super().__init__()
        
        input_dim = CONTEXT_LENGTH * 6 + 4 + LOOKAHEAD_LENGTH * 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor(-1.0))
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
    def forward(self, x, deterministic=True):
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        mean = torch.tanh(mean) * STEER_RANGE
        return mean, None, None


class Controller(BaseController):
    def __init__(self, model_path=None):
        model_dir = Path(__file__).parent.parent / "models"
        
        if model_path:
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = model_dir / model_path
        else:
            # Find latest ppo_debug checkpoint
            ppo_dirs = list(model_dir.glob('ppo_debug_*'))
            if ppo_dirs:
                latest = sorted(ppo_dirs)[-1]
                model_path = latest / 'best_model.pt'
            else:
                raise FileNotFoundError("No ppo_debug model found")
        
        print(f"Loading PPO debug model: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.model = SteerPolicy()
        self.model.load_state_dict(checkpoint['policy_state'])
        
        # Load input normalization if available
        if 'input_mean' in checkpoint:
            self.model.input_mean = checkpoint['input_mean']
            self.model.input_std = checkpoint['input_std']
        
        self.model.eval()
        
        self.history = []
        self.prev_steer = 0.0
    
    @torch.no_grad()
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        roll = state.roll_lataccel
        obs = [state.v_ego, state.a_ego, roll, target_lataccel, current_lataccel, self.prev_steer]
        self.history.append(obs)
        
        if len(self.history) < CONTEXT_LENGTH:
            return 0.0
        
        if not future_plan or len(future_plan.lataccel) < LOOKAHEAD_LENGTH:
            return self.prev_steer
        
        # Build flattened input
        past_ctx = np.array(self.history[-CONTEXT_LENGTH:], dtype=np.float32)
        current = np.array([state.v_ego, state.a_ego, roll, current_lataccel], dtype=np.float32)
        future_ctx = np.stack([
            np.array(future_plan.v_ego[:LOOKAHEAD_LENGTH]),
            np.array(future_plan.a_ego[:LOOKAHEAD_LENGTH]),
            np.array(future_plan.roll_lataccel[:LOOKAHEAD_LENGTH]),
            np.array(future_plan.lataccel[:LOOKAHEAD_LENGTH]),
        ], axis=1).astype(np.float32)
        
        flat_input = np.concatenate([past_ctx.flatten(), current, future_ctx.flatten()])
        x = torch.tensor(flat_input, dtype=torch.float32).unsqueeze(0)
        
        action, _, _ = self.model(x, deterministic=True)
        steer = action.item()
        steer = np.clip(steer, -STEER_RANGE, STEER_RANGE)
        
        self.prev_steer = steer
        return steer

