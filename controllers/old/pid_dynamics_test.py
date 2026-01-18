"""
PID controller that also runs dynamics model predictions for testing.
Logs both predicted and actual lataccel to compare model accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from . import BaseController

HISTORY_LEN = 10


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
        # PID gains
        self.kp = 0.3
        self.ki = 0.05
        self.kd = -0.1

        # PID state
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Load dynamics model
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model = DynamicsMLP()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.history_mean = checkpoint['history_mean']
            self.history_std = checkpoint['history_std']
            self.action_mean = checkpoint['action_mean']
            self.action_std = checkpoint['action_std']
            self.delta_mean = checkpoint['delta_mean']
            self.delta_std = checkpoint['delta_std']
            self.model_loaded = True
            print("Dynamics model loaded for testing (delta prediction)")
        except Exception as e:
            print(f"Could not load dynamics model: {e}")
            self.model_loaded = False

        # History buffer
        self.state_history = deque(maxlen=HISTORY_LEN)
        self.prev_action = 0.0

        # Prediction tracking
        self.last_prediction = None
        self.prediction_errors = []

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Check prediction from last step
        if self.last_prediction is not None:
            error = abs(self.last_prediction - current_lataccel)
            self.prediction_errors.append(error)

        # Store current state
        self.state_history.append({
            'vEgo': state.v_ego,
            'aEgo': state.a_ego,
            'rollLateralAcceleration': state.roll_lataccel,
            'currentLateralAcceleration': current_lataccel,
            'steerCommand': self.prev_action,
        })

        # PID control
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_derivative = error - self.prev_error
        self.prev_error = error

        action = (self.kp * error +
                  self.ki * self.error_integral +
                  self.kd * error_derivative)
        action = np.clip(action, -2.0, 2.0)

        # Make prediction for next step (if model loaded and have enough history)
        self.last_prediction = None
        if self.model_loaded and len(self.state_history) >= HISTORY_LEN:
            self.last_prediction = self._predict(action, current_lataccel)

        self.prev_action = action
        return action

    def _predict(self, action, current_lataccel):
        """Predict next lataccel using dynamics model (delta prediction)."""
        # Build history
        history = []
        for s in self.state_history:
            history.append([
                s['vEgo'], s['aEgo'], s['rollLateralAcceleration'],
                s['currentLateralAcceleration'], s['steerCommand']
            ])
        hist_flat = np.array(history).flatten()

        # Normalize
        hist_norm = (hist_flat - self.history_mean) / self.history_std
        action_norm = (action - self.action_mean) / self.action_std

        # Predict delta
        x = np.concatenate([hist_norm, [action_norm]])
        x_tensor = torch.FloatTensor(x).unsqueeze(0)

        with torch.no_grad():
            delta_norm = self.model(x_tensor).item()

        delta = delta_norm * self.delta_std + self.delta_mean
        # Next lataccel = current + delta
        return current_lataccel + delta

    def get_log(self):
        """Return extra log data."""
        log = {}
        if self.last_prediction is not None:
            log['predictedLataccel'] = self.last_prediction
        if len(self.prediction_errors) > 0:
            log['predictionError'] = self.prediction_errors[-1]
            log['meanPredError'] = np.mean(self.prediction_errors)
        return log
