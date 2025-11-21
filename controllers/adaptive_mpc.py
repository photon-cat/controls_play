from collections import deque
import numpy as np

from controllers import BaseController


class Controller(BaseController):
  """
  PID baseline with gentle autotuning, gain scheduling, and anti-windup.

  The authority is intentionally limited so we stay near the well-behaved region,
  while the P/D terms decay with speed to avoid high-speed oscillations.
  """

  def __init__(self):
    # Base PID gains taken from the hand-tuned controller
    self.kp_base = 0.195
    self.ki_base = 0.100
    self.kd_base = -0.053

    # Autotune state
    self.kp_scale = 1.0
    self.ki_scale = 1.0
    self.kp_scale_min = 0.6
    self.kp_scale_max = 1.6
    self.ki_scale_min = 0.4
    self.ki_scale_max = 1.2
    self.autotune_alpha = 0.02
    self.error_window = deque(maxlen=40)
    self.error_rms = 0.0
    self.high_band = 0.35  # rms error threshold for adding authority
    self.low_band = 0.12   # rms error threshold for reducing authority

    # PID internal state
    self.error_integral = 0.0
    self.integral_limit = 5.0
    self.integral_decay = 0.998
    self.prev_error = 0.0
    self.derivative_state = 0.0
    self.derivative_alpha = 0.25

    # Output limits (/Users/delta/comma/controls_play/tinyphysics.py clamps steer to +-2)
    self.steer_min = -2.0
    self.steer_max = 2.0
    self.authority_limit = 0.9  # additional software authority limit
    self.max_delta = 0.15       # slew-rate limit per cycle
    self.prev_command = 0.0

  def observe_applied_action(self, applied_action: float):
    # Track what was actually applied so the slew limiter stays in sync
    applied = float(np.clip(applied_action, -self.authority_limit, self.authority_limit))
    self.prev_command = applied

  def _autotune(self):
    if not self.error_window:
      return
    abs_error = np.abs(np.array(self.error_window))
    rms = float(np.sqrt(np.mean(abs_error ** 2)))
    self.error_rms = (1.0 - self.autotune_alpha) * self.error_rms + self.autotune_alpha * rms

    if self.error_rms > self.high_band:
      self.kp_scale = min(self.kp_scale + 0.02, self.kp_scale_max)
      self.ki_scale = min(self.ki_scale + 0.01, self.ki_scale_max)
    elif self.error_rms < self.low_band:
      self.kp_scale = max(self.kp_scale - 0.01, self.kp_scale_min)
      self.ki_scale = max(self.ki_scale - 0.005, self.ki_scale_min)

  def _get_feedforward(self, target_lataccel, future_plan):
    if future_plan and getattr(future_plan, "lataccel", None):
      lookahead = min(5, len(future_plan.lataccel))
      if lookahead > 0:
        future_delta = float(future_plan.lataccel[lookahead - 1]) - float(target_lataccel)
        return 0.05 * future_delta
    return 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = float(target_lataccel) - float(current_lataccel)
    self.error_window.append(error)
    self._autotune()

    # Integral term with decay to avoid runaway bias accumulation
    self.error_integral *= self.integral_decay
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)

    # Derivative smoothing keeps noise out of the D channel
    raw_derivative = error - self.prev_error
    self.derivative_state = (
      self.derivative_alpha * raw_derivative +
      (1.0 - self.derivative_alpha) * self.derivative_state
    )
    self.prev_error = error

    # Gain scheduling vs speed (never blows up below 5 m/s)
    speed = max(float(getattr(state, "v_ego", 0.0)), 0.0)
    speed_scale = 1.0 / max(speed, 5.0)
    kp = self.kp_base * self.kp_scale * speed_scale
    kd = self.kd_base * speed_scale
    ki = self.ki_base * self.ki_scale

    feedforward = self._get_feedforward(target_lataccel, future_plan)
    command = kp * error + ki * self.error_integral + kd * self.derivative_state + feedforward

    # Physical clamp with classical clamping anti-windup
    command_clamped = np.clip(command, self.steer_min, self.steer_max)
    saturated_high = command > self.steer_max
    saturated_low = command < self.steer_min
    if saturated_high and error > 0.0:
      self.error_integral -= error
    elif saturated_low and error < 0.0:
      self.error_integral -= error

    if saturated_high or saturated_low:
      # Rebuild command using the updated integral so future steps stay aligned
      command = kp * error + ki * self.error_integral + kd * self.derivative_state + feedforward
      command_clamped = np.clip(command, self.steer_min, self.steer_max)

    # Limited authority + slew-rate limit
    desired = np.clip(command_clamped, -self.authority_limit, self.authority_limit)
    delta = np.clip(desired - self.prev_command, -self.max_delta, self.max_delta)
    command_limited = self.prev_command + delta
    command_limited = np.clip(command_limited, -self.authority_limit, self.authority_limit)
    self.prev_command = command_limited

    return command_limited
