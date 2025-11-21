from collections import deque
import numpy as np

from controllers import BaseController


class Controller(BaseController):
  """
  Limited-authority PID controller with gentle autotuning.

  Starts from the known-good PID gains and slowly scales P/I terms
  based on the recent tracking error magnitude. Output authority
  and slew-rate are capped to avoid oscillations.
  """

  def __init__(self):
    # Base PID gains (same as controllers/pid)
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
    self.high_band = 0.35  # m/s^2 error threshold for adding authority
    self.low_band = 0.12   # m/s^2 error threshold for reducing authority

    # PID integrator / derivative state
    self.error_integral = 0.0
    self.integral_limit = 5.0
    self.integral_decay = 0.998
    self.prev_error = 0.0
    self.derivative_state = 0.0
    self.derivative_alpha = 0.25

    # Output shaping / limits
    self.max_output = 0.9           # limited steering authority
    self.max_delta = 0.15           # slew-rate limit per update
    self.prev_command = 0.0

  def observe_applied_action(self, applied_action: float):
    # Keep slew-rate limiter aligned with what the simulator applies
    self.prev_command = float(np.clip(applied_action, -self.max_output, self.max_output))

  def _autotune(self):
    if not self.error_window:
      return
    abs_error = np.abs(np.array(self.error_window))
    rms = float(np.sqrt(np.mean(abs_error**2)))
    self.error_rms = (1.0 - self.autotune_alpha) * self.error_rms + self.autotune_alpha * rms

    if self.error_rms > self.high_band:
      self.kp_scale = min(self.kp_scale + 0.02, self.kp_scale_max)
      self.ki_scale = min(self.ki_scale + 0.01, self.ki_scale_max)
    elif self.error_rms < self.low_band:
      self.kp_scale = max(self.kp_scale - 0.01, self.kp_scale_min)
      self.ki_scale = max(self.ki_scale - 0.005, self.ki_scale_min)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = float(target_lataccel) - float(current_lataccel)
    self.error_window.append(error)
    self._autotune()

    # Integral with decay + clamp
    self.error_integral *= self.integral_decay
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)

    # Filtered derivative
    raw_derivative = error - self.prev_error
    self.derivative_state = (
      self.derivative_alpha * raw_derivative +
      (1.0 - self.derivative_alpha) * self.derivative_state
    )
    self.prev_error = error

    kp = self.kp_base * self.kp_scale
    ki = self.ki_base * self.ki_scale
    kd = self.kd_base

    command = kp * error + ki * self.error_integral + kd * self.derivative_state

    # Optional gentle feedforward: anticipate nearest target change
    if future_plan and getattr(future_plan, "lataccel", None):
      lookahead = min(5, len(future_plan.lataccel))
      if lookahead > 0:
        future_delta = float(future_plan.lataccel[lookahead - 1]) - float(target_lataccel)
        command += 0.05 * future_delta

    # Slew-rate limit and clamp authority
    desired = np.clip(command, -self.max_output, self.max_output)
    delta = np.clip(desired - self.prev_command, -self.max_delta, self.max_delta)
    command = self.prev_command + delta
    command = np.clip(command, -self.max_output, self.max_output)
    self.prev_command = command

    return command
