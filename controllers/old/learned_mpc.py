"""
MPC controller using learned single-step dynamics model.
Rolls out model autoregressively to evaluate action sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from . import BaseController

# Constants
HISTORY_LEN = 10
FUTURE_LEN = 10
NUM_FEATURES = 5  # vEgo, aEgo, rollLateralAcceleration, currentLateralAcceleration, steerCommand


class DynamicsMLP(nn.Module):
    """Must match training architecture."""

    def __init__(self, input_dim=51, hidden_dims=[128, 64, 32], output_dim=1):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Controller(BaseController):
    def __init__(self, model_path='models/dynamics.pt'):
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        self.model = DynamicsMLP()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Normalization stats
        self.history_mean = checkpoint['history_mean']
        self.history_std = checkpoint['history_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        self.delta_mean = checkpoint['delta_mean']
        self.delta_std = checkpoint['delta_std']

        # History buffer - stores full state info
        self.state_history = deque(maxlen=HISTORY_LEN)

        # MPC params
        self.n_candidates = 64
        self.n_refinements = 3
        self.elite_frac = 0.1
        self.action_range = (-2.0, 2.0)
        self.lat_accel_cost_weight = 50.0
        self.jerk_cost_weight = 1.0

        # Warm start
        self.prev_action = 0.0
        self.prev_plan = None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Store current state
        self.state_history.append({
            'vEgo': state.v_ego,
            'aEgo': state.a_ego,
            'rollLateralAcceleration': state.roll_lataccel,
            'currentLateralAcceleration': current_lataccel,
            'steerCommand': self.prev_action,
        })

        # Need full history
        if len(self.state_history) < HISTORY_LEN:
            self.prev_action = 0.0
            return 0.0

        # Get target trajectory
        target_lataccels = [target_lataccel] + list(future_plan.lataccel[:FUTURE_LEN-1])
        while len(target_lataccels) < FUTURE_LEN:
            target_lataccels.append(target_lataccels[-1])
        target_lataccels = np.array(target_lataccels[:FUTURE_LEN])

        # Get future state info for rollout (vEgo, aEgo, roll)
        future_vego = [state.v_ego] + list(future_plan.v_ego[:FUTURE_LEN-1])
        future_aego = [state.a_ego] + list(future_plan.a_ego[:FUTURE_LEN-1])
        future_roll = [state.roll_lataccel] + list(future_plan.roll_lataccel[:FUTURE_LEN-1])
        while len(future_vego) < FUTURE_LEN:
            future_vego.append(future_vego[-1])
            future_aego.append(future_aego[-1])
            future_roll.append(future_roll[-1])

        # Initialize CEM
        if self.prev_plan is not None:
            init_mean = np.concatenate([self.prev_plan[1:], [self.prev_plan[-1]]])
        else:
            init_mean = target_lataccels * 0.1

        init_mean = np.clip(init_mean, self.action_range[0], self.action_range[1])
        mean = init_mean.copy()
        std = np.full(FUTURE_LEN, 0.3)

        best_action_seq = mean.copy()
        best_cost = float('inf')

        # CEM optimization
        for _ in range(self.n_refinements):
            # Generate candidates
            candidates = []
            for _ in range(self.n_candidates - 1):
                seq = mean + np.random.randn(FUTURE_LEN) * std
                seq = np.clip(seq, self.action_range[0], self.action_range[1])
                candidates.append(seq)
            candidates.append(mean.copy())

            # Evaluate candidates via rollout
            costs = []
            for action_seq in candidates:
                predicted = self._rollout(action_seq, future_vego, future_aego, future_roll)
                cost = self._compute_cost(predicted, target_lataccels, action_seq)
                costs.append((cost, action_seq))

            # Select elites
            costs.sort(key=lambda x: x[0])
            n_elite = max(1, int(self.n_candidates * self.elite_frac))
            elites = np.array([c[1] for c in costs[:n_elite]])

            if costs[0][0] < best_cost:
                best_cost = costs[0][0]
                best_action_seq = costs[0][1].copy()

            # Update distribution
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0) + 0.05

        # Take first action
        action = float(best_action_seq[0])
        action = np.clip(action, self.action_range[0], self.action_range[1])

        self.prev_action = action
        self.prev_plan = best_action_seq

        return action

    def _rollout(self, action_seq, future_vego, future_aego, future_roll):
        """Roll out dynamics model to predict trajectory (delta prediction)."""
        # Build initial history from buffer
        history = []
        for s in self.state_history:
            history.append([
                s['vEgo'], s['aEgo'], s['rollLateralAcceleration'],
                s['currentLateralAcceleration'], s['steerCommand']
            ])
        history = np.array(history)  # Shape: (HISTORY_LEN, 5)

        predictions = []
        current_lataccel = history[-1, 3]  # Last currentLateralAcceleration

        for t in range(FUTURE_LEN):
            # Flatten history
            hist_flat = history.flatten()

            # Normalize
            hist_norm = (hist_flat - self.history_mean) / self.history_std
            action_norm = (action_seq[t] - self.action_mean) / self.action_std

            # Predict delta
            x = np.concatenate([hist_norm, [action_norm]])
            x_tensor = torch.FloatTensor(x).unsqueeze(0)

            with torch.no_grad():
                delta_norm = self.model(x_tensor).item()

            delta = delta_norm * self.delta_std + self.delta_mean
            pred = current_lataccel + delta  # Next = current + delta
            predictions.append(pred)

            # Update for next step
            current_lataccel = pred
            new_state = np.array([
                future_vego[t],
                future_aego[t],
                future_roll[t],
                pred,
                action_seq[t]
            ])
            history = np.vstack([history[1:], new_state])

        return np.array(predictions)

    def _compute_cost(self, predicted, target, action_seq):
        """Compute cost for predicted trajectory."""
        # Tracking cost
        lat_cost = np.mean((predicted - target) ** 2) * self.lat_accel_cost_weight

        # Jerk cost
        jerk = np.diff(predicted) / 0.1
        jerk_cost = np.mean(jerk ** 2) * self.jerk_cost_weight

        # Action smoothness cost
        action_diff = np.diff(action_seq)
        action_cost = np.mean(action_diff ** 2) * 0.1

        return lat_cost + jerk_cost + action_cost
