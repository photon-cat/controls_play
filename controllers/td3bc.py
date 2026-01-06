"""
TD3+BC controller wrapper for tinyphysics simulator.
Loads trained offline RL model and uses it for steering control.
"""

from . import BaseController
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class Actor(nn.Module):
    """Deterministic policy network"""
    def __init__(self, state_dim=104, action_dim=1, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state) * 2.0  # Scale to [-2, 2]


class Controller(BaseController):
    """TD3+BC learned controller"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            # Use latest model by default
            models_dir = Path(__file__).parent.parent / "models"
            # Find all td3bc directories that have final.pt
            td3bc_dirs = []
            for d in sorted(models_dir.glob("td3bc_*")):
                if (d / "final.pt").exists():
                    td3bc_dirs.append(d)
            
            if td3bc_dirs:
                model_path = td3bc_dirs[-1] / "final.pt"
            else:
                raise ValueError("No TD3+BC model with final.pt found!")
        
        self.model_path = model_path
        self.device = torch.device('cpu')
        
        # Load model
        self.actor = Actor(state_dim=104, action_dim=1, hidden=256).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
        
        # Load normalization stats
        dataset_path = Path(__file__).parent.parent / "datasets" / "pid_ff_1k.npz"
        if dataset_path.exists():
            data = np.load(dataset_path)
            self.state_mean = data['state_mean']
            self.state_std = data['state_std']
        else:
            # No normalization
            self.state_mean = np.zeros(104)
            self.state_std = np.ones(104)
        
        # History for state construction
        self.context_len = 10
        self.lookahead_len = 10
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []
    
    def reset(self):
        """Reset history"""
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute steering command using TD3+BC policy.
        
        Args:
            target_lataccel: desired lateral acceleration
            current_lataccel: current lateral acceleration
            state: current vehicle state (v_ego, a_ego, roll_lataccel)
            future_plan: future trajectory plan
        
        Returns:
            steering command in [-2, 2]
        """
        # Update history
        self.state_history.append(state)
        self.target_history.append(target_lataccel)
        self.lataccel_history.append(current_lataccel)
        
        # Construct state vector (104 dims)
        obs = self._get_obs(state, current_lataccel, target_lataccel, future_plan)
        
        # Normalize and convert to tensor
        obs_norm = (obs - self.state_mean) / (self.state_std + 1e-8)
        
        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().numpy()[0, 0]
        
        # Store action
        self.action_history.append(action)
        
        return float(action)
    
    def _get_obs(self, state, current_lataccel, target_lataccel, future_plan):
        """
        Construct 104-dim observation vector.
        Format: [past_ctx (10x6), current (4), future_ctx (10x4)]
        """
        # Past context: [v_ego, a_ego, roll, target_lat, measured_lat, steer_cmd]
        past_ctx = []
        for i in range(max(0, len(self.state_history) - self.context_len), len(self.state_history)):
            if i < len(self.state_history):
                s = self.state_history[i]
                t = self.target_history[i] if i < len(self.target_history) else 0.0
                m = self.lataccel_history[i] if i < len(self.lataccel_history) else 0.0
                a = self.action_history[i] if i < len(self.action_history) else 0.0
                past_ctx.append([s.v_ego, s.a_ego, s.roll_lataccel, t, m, a])
        
        # Pad if needed
        while len(past_ctx) < self.context_len:
            past_ctx.insert(0, [0, 0, 0, 0, 0, 0])
        past_ctx = np.array(past_ctx[-self.context_len:]).flatten()
        
        # Current state: [v_ego, a_ego, roll, measured_lat]
        current = np.array([
            state.v_ego,
            state.a_ego,
            state.roll_lataccel,
            current_lataccel
        ])
        
        # Future context: [v_ego, a_ego, roll, target_lat]
        future_ctx = []
        if future_plan is not None and hasattr(future_plan, 'lataccel'):
            for i in range(min(self.lookahead_len, len(future_plan.lataccel))):
                future_ctx.append([
                    future_plan.v_ego[i] if i < len(future_plan.v_ego) else state.v_ego,
                    future_plan.a_ego[i] if i < len(future_plan.a_ego) else state.a_ego,
                    future_plan.roll_lataccel[i] if i < len(future_plan.roll_lataccel) else state.roll_lataccel,
                    future_plan.lataccel[i]
                ])
        
        # Pad if needed
        while len(future_ctx) < self.lookahead_len:
            future_ctx.append([state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel])
        future_ctx = np.array(future_ctx[:self.lookahead_len]).flatten()
        
        # Concatenate
        obs = np.concatenate([past_ctx, current, future_ctx]).astype(np.float32)
        
        return obs

