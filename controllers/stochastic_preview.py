from . import BaseController
import numpy as np


class Controller(BaseController):
  def __init__(self):
    self.response_rate = 0.38
    self.horizon = 6
    self.num_samples = 64
    self.action_noise = 0.12
    self.jerk_weight = 0.8
    self.action_weight = 0.02
    self.smoothing = 0.5
    self.prev_action = 0.0
    self.base_gain = 1.0
    self.gain_scale = 1.0
    self.gain_variance = 10.0
    self.gain_forgetting = 0.98
    self.min_gain = 0.3
    self.max_gain = 4.0
    self.gain = 1.0
    self._log = {}

  def _update_gain(self, current_lataccel, roll_lataccel, base_gain):
    action = self.prev_action
    if abs(action) < 1e-3:
      self.base_gain = base_gain
      self.gain = float(np.clip(self.base_gain * self.gain_scale, self.min_gain, self.max_gain))
      return
    output = current_lataccel - roll_lataccel
    covariance = self.gain_variance
    denom = self.gain_forgetting + action * covariance * action
    gain_factor = covariance * action / denom
    scaled_output = output / base_gain
    self.gain_scale = self.gain_scale + gain_factor * (scaled_output - action * self.gain_scale)
    self.gain_scale = float(np.clip(self.gain_scale, 0.3, 3.0))
    self.base_gain = base_gain
    self.gain = float(np.clip(self.base_gain * self.gain_scale, self.min_gain, self.max_gain))
    self.gain_variance = (covariance - gain_factor * action * covariance) / self.gain_forgetting

  def _base_gain_from_speed(self, v_ego):
    velocities = np.array([0.0, 5.0, 15.0, 25.0, 35.0, 45.0], dtype=float)
    gains = np.array([0.9, 1.0, 1.2, 1.5, 1.9, 2.3], dtype=float)
    v_clamped = float(np.clip(v_ego, velocities[0], velocities[-1]))
    return float(np.interp(v_clamped, velocities, gains))

  def get_log(self):
    return self._log

  def _build_sequence(self, current_value, future_values):
    sequence = [current_value]
    if future_values:
      sequence.extend(list(future_values[:self.horizon - 1]))
    while len(sequence) < self.horizon:
      sequence.append(sequence[-1])
    return np.array(sequence, dtype=float)

  def _simulate_cost(self, action_sequence, target_sequence, roll_sequence, current_lataccel):
    predicted_lataccel = current_lataccel
    cost = 0.0
    for step in range(self.horizon):
      predicted_lataccel = predicted_lataccel + self.response_rate * (
        (self.gain * action_sequence[step] + roll_sequence[step]) - predicted_lataccel
      )
      error = predicted_lataccel - target_sequence[step]
      cost += error * error
      if step > 0:
        delta_action = action_sequence[step] - action_sequence[step - 1]
        cost += self.jerk_weight * delta_action * delta_action
      cost += self.action_weight * action_sequence[step] * action_sequence[step]
    return cost

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    base_gain = self._base_gain_from_speed(state.v_ego)
    self._update_gain(current_lataccel, state.roll_lataccel, base_gain)
    target_sequence = self._build_sequence(
      target_lataccel,
      future_plan.lataccel if future_plan else None,
    )
    roll_sequence = self._build_sequence(
      state.roll_lataccel,
      future_plan.roll_lataccel if future_plan else None,
    )
    baseline_action = (target_sequence - roll_sequence) / self.gain
    baseline_action = np.clip(baseline_action, -2.0, 2.0)
    best_sequence = baseline_action
    best_cost = self._simulate_cost(baseline_action, target_sequence, roll_sequence, current_lataccel)
    for _ in range(self.num_samples):
      noise = np.random.normal(scale=self.action_noise, size=self.horizon)
      candidate = np.clip(baseline_action + noise, -2.0, 2.0)
      cost = self._simulate_cost(candidate, target_sequence, roll_sequence, current_lataccel)
      if cost < best_cost:
        best_cost = cost
        best_sequence = candidate
    action = (1.0 - self.smoothing) * best_sequence[0] + self.smoothing * self.prev_action
    action = float(np.clip(action, -2.0, 2.0))
    self.prev_action = action
    self._log = {
      'sp_gain': self.gain,
      'sp_base_gain': self.base_gain,
      'sp_gain_scale': self.gain_scale,
      'sp_cost': best_cost,
      'sp_action': action,
    }
    return action
