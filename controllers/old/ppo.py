"""
PPO-trained controller for inference.

Usage:
    python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller ppo
"""

from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CONTEXT_LENGTH = 20


class TemporalPolicy(nn.Module):
    """Transformer-based policy (must match training architecture)."""

    def __init__(
        self,
        hist_features=6,
        fut_features=4,
        current_features=6,
        seq_len=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.0,  # No dropout at inference
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        self.hist_embed = nn.Linear(hist_features, d_model)
        self.fut_embed = nn.Linear(fut_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.hist_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fut_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
            ),
            num_layers
        )

        self.current_embed = nn.Sequential(
            nn.Linear(current_features, d_model),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.tensor(-0.5))
        self.critic = nn.Linear(128, 1)
        self.action_scale = 2.0

    def forward(self, hist, current, future):
        h = self.hist_embed(hist) + self.pos_embed
        h = self.hist_transformer(h)
        h_summary = h[:, -1]

        f = self.fut_embed(future) + self.pos_embed
        f = self.fut_transformer(f)
        f_summary = f[:, 0]

        c = self.current_embed(current)

        fused = torch.cat([h_summary, f_summary, c], dim=-1)
        features = self.fusion(fused)
        return features

    def get_action(self, hist, current, future, deterministic=True):
        features = self.forward(hist, current, future)
        mean = torch.tanh(self.actor_mean(features)) * self.action_scale

        if deterministic:
            return mean
        else:
            std = self.actor_log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            return dist.sample()


class Controller(BaseController):
    def __init__(self):
        self.policy = None
        self.device = 'cpu'

        # History buffers
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []

        self.prev_action = 0.0
        self.step_count = 0

        # Load model
        self._load_model()

    def _load_model(self):
        model_paths = [
            Path('models/ppo_best.pt'),
            Path('models/ppo_final.pt'),
        ]

        for path in model_paths:
            if path.exists():
                self.policy = TemporalPolicy()
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.policy.eval()
                print(f"Loaded PPO model from {path}")
                return

        print("WARNING: No PPO model found. Using zero controller.")

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1

        # Update histories
        self.state_history.append(state)
        self.lataccel_history.append(current_lataccel)
        self.target_history.append(target_lataccel)

        # Trim to context length
        if len(self.state_history) > CONTEXT_LENGTH:
            self.state_history = self.state_history[-CONTEXT_LENGTH:]
        if len(self.lataccel_history) > CONTEXT_LENGTH:
            self.lataccel_history = self.lataccel_history[-CONTEXT_LENGTH:]
        if len(self.target_history) > CONTEXT_LENGTH:
            self.target_history = self.target_history[-CONTEXT_LENGTH:]
        if len(self.action_history) > CONTEXT_LENGTH:
            self.action_history = self.action_history[-CONTEXT_LENGTH:]

        # Need model and enough history
        if self.policy is None or len(self.state_history) < CONTEXT_LENGTH:
            self.action_history.append(self.prev_action)
            return self.prev_action

        # Build observation
        obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan)

        # Get action
        with torch.no_grad():
            hist_t = torch.tensor(obs['hist'], dtype=torch.float32).unsqueeze(0)
            curr_t = torch.tensor(obs['current'], dtype=torch.float32).unsqueeze(0)
            fut_t = torch.tensor(obs['future'], dtype=torch.float32).unsqueeze(0)

            action = self.policy.get_action(hist_t, curr_t, fut_t, deterministic=True)
            action = action.squeeze().item()

        action = np.clip(action, -2.0, 2.0)
        self.action_history.append(action)
        self.prev_action = action

        return action

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan):
        # History
        hist = np.zeros((CONTEXT_LENGTH, 6), dtype=np.float32)
        for i in range(CONTEXT_LENGTH):
            idx = i
            if idx < len(self.state_history):
                s = self.state_history[idx]
                hist[i] = [
                    s.v_ego / 30.0,
                    s.a_ego / 4.0,
                    s.roll_lataccel / 2.0,
                    self.lataccel_history[idx] / 5.0 if idx < len(self.lataccel_history) else 0,
                    self.target_history[idx] / 5.0 if idx < len(self.target_history) else 0,
                    self.action_history[idx] / 2.0 if idx < len(self.action_history) else 0,
                ]

        # Current
        error = target_lataccel - current_lataccel
        current = np.array([
            state.v_ego / 30.0,
            state.a_ego / 4.0,
            state.roll_lataccel / 2.0,
            current_lataccel / 5.0,
            target_lataccel / 5.0,
            error / 5.0,
        ], dtype=np.float32)

        # Future
        future = np.zeros((CONTEXT_LENGTH, 4), dtype=np.float32)
        if future_plan and hasattr(future_plan, 'v_ego') and future_plan.v_ego:
            future_len = min(CONTEXT_LENGTH, len(future_plan.v_ego))
            for i in range(future_len):
                future[i] = [
                    future_plan.v_ego[i] / 30.0 if i < len(future_plan.v_ego) else 0,
                    future_plan.a_ego[i] / 4.0 if i < len(future_plan.a_ego) else 0,
                    future_plan.roll_lataccel[i] / 2.0 if i < len(future_plan.roll_lataccel) else 0,
                    future_plan.lataccel[i] / 5.0 if i < len(future_plan.lataccel) else 0,
                ]

        return {
            'hist': hist,
            'current': current,
            'future': future,
        }

    def observe_action(self, action, step_idx):
        """Called during warmup to observe actual action used."""
        if self.action_history:
            self.action_history[-1] = action
        self.prev_action = action
