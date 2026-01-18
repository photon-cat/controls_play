"""
Residual PPO controller - PID + learned corrections.

Usage:
    python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller ppo_residual
"""

from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CONTEXT_LENGTH = 20


class PIDController:
    def __init__(self, p=0.195, i=0.100, d=-0.053):
        self.p = p
        self.i = i
        self.d = d
        self.error_integral = 0
        self.prev_error = 0

    def reset(self):
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        return action, error, self.error_integral, error_diff


class ResidualPolicy(nn.Module):
    def __init__(
        self,
        hist_features=6,
        fut_features=4,
        current_features=10,
        seq_len=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.0,
        max_residual=0.5,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.max_residual = max_residual

        self.hist_embed = nn.Linear(hist_features, d_model)
        self.fut_embed = nn.Linear(fut_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
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
        self.actor_log_std = nn.Parameter(torch.tensor(-1.0))
        self.critic = nn.Linear(128, 1)

    def forward(self, hist, current, future):
        h = self.hist_embed(hist) + self.pos_embed
        h = self.hist_transformer(h)
        h_summary = h[:, -1]

        f = self.fut_embed(future) + self.pos_embed
        f = self.fut_transformer(f)
        f_summary = f[:, 0]

        c = self.current_embed(current)

        fused = torch.cat([h_summary, f_summary, c], dim=-1)
        return self.fusion(fused)

    def get_residual(self, hist, current, future):
        features = self.forward(hist, current, future)
        mean = torch.tanh(self.actor_mean(features)) * self.max_residual
        return mean


class Controller(BaseController):
    def __init__(self):
        self.policy = None
        self.device = 'cpu'
        self.pid = PIDController()

        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []

        self.prev_action = 0.0
        self.current_lataccel = 0.0
        self.step_count = 0

        self._load_model()

    def _load_model(self):
        model_paths = [
            Path('models/ppo_residual_best.pt'),
            Path('models/ppo_residual_final.pt'),
        ]

        for path in model_paths:
            if path.exists():
                self.policy = ResidualPolicy()
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.policy.eval()
                print(f"Loaded residual PPO model from {path}")
                return

        print("WARNING: No residual PPO model found. Using pure PID.")

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        self.current_lataccel = current_lataccel

        # Update histories
        self.state_history.append(state)
        self.lataccel_history.append(current_lataccel)
        self.target_history.append(target_lataccel)

        if len(self.state_history) > CONTEXT_LENGTH:
            self.state_history = self.state_history[-CONTEXT_LENGTH:]
        if len(self.lataccel_history) > CONTEXT_LENGTH:
            self.lataccel_history = self.lataccel_history[-CONTEXT_LENGTH:]
        if len(self.target_history) > CONTEXT_LENGTH:
            self.target_history = self.target_history[-CONTEXT_LENGTH:]
        if len(self.action_history) > CONTEXT_LENGTH:
            self.action_history = self.action_history[-CONTEXT_LENGTH:]

        # Get PID action
        pid_action, pid_error, pid_integral, pid_deriv = self.pid.update(
            target_lataccel, current_lataccel
        )

        # Get residual from neural network
        residual = 0.0
        if self.policy is not None and len(self.state_history) >= CONTEXT_LENGTH:
            obs = self._build_obs(target_lataccel, current_lataccel, state, future_plan,
                                  pid_action, pid_error, pid_integral, pid_deriv)

            with torch.no_grad():
                hist_t = torch.tensor(obs['hist'], dtype=torch.float32).unsqueeze(0)
                curr_t = torch.tensor(obs['current'], dtype=torch.float32).unsqueeze(0)
                fut_t = torch.tensor(obs['future'], dtype=torch.float32).unsqueeze(0)

                residual = self.policy.get_residual(hist_t, curr_t, fut_t)
                residual = residual.squeeze().item()

        # Combine PID + residual
        action = pid_action + residual
        action = np.clip(action, -2.0, 2.0)

        self.action_history.append(action)
        self.prev_action = action

        return action

    def _build_obs(self, target_lataccel, current_lataccel, state, future_plan,
                   pid_action, pid_error, pid_integral, pid_deriv):
        # History
        hist = np.zeros((CONTEXT_LENGTH, 6), dtype=np.float32)
        for i in range(CONTEXT_LENGTH):
            if i < len(self.state_history):
                s = self.state_history[i]
                hist[i] = [
                    s.v_ego / 30.0,
                    s.a_ego / 4.0,
                    s.roll_lataccel / 2.0,
                    self.lataccel_history[i] / 5.0 if i < len(self.lataccel_history) else 0,
                    self.target_history[i] / 5.0 if i < len(self.target_history) else 0,
                    self.action_history[i] / 2.0 if i < len(self.action_history) else 0,
                ]

        # Current + PID state
        error = target_lataccel - current_lataccel
        current = np.array([
            state.v_ego / 30.0,
            state.a_ego / 4.0,
            state.roll_lataccel / 2.0,
            current_lataccel / 5.0,
            target_lataccel / 5.0,
            error / 5.0,
            pid_action / 2.0,
            pid_error / 5.0,
            pid_integral / 50.0,
            pid_deriv / 2.0,
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

        return {'hist': hist, 'current': current, 'future': future}

    def observe_action(self, action, step_idx):
        if self.action_history:
            self.action_history[-1] = action
        self.prev_action = action
        # Reset PID during warmup
        if step_idx < 100:
            self.pid.reset()
