"""
PID controller with modest tuning tweaks for real data:
- Integral clamp to avoid windup
- Light derivative smoothing to reduce jerk
- Tunable gains via constructor for quick iteration
"""

from . import BaseController
import numpy as np


class Controller(BaseController):
  def __init__(self, p=0.22, i=0.10, d=-0.03, integral_limit=2.0, deriv_alpha=0.4):
    """
    Args:
      p, i, d: PID gains
      integral_limit: clamp for the accumulated error
      deriv_alpha: smoothing factor (0-1]; 1 = no smoothing
    """
    self.p = p
    self.i = i
    self.d = d
    self.integral_limit = integral_limit
    self.deriv_alpha = deriv_alpha
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.smoothed_error_diff = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel

    # Integral with clamping to prevent windup on long biases
    self.error_integral = np.clip(self.error_integral + error, -self.integral_limit, self.integral_limit)

    # Smoothed derivative to reduce noise-induced jerk
    raw_diff = error - self.prev_error
    self.smoothed_error_diff = (self.deriv_alpha * raw_diff) + (1 - self.deriv_alpha) * self.smoothed_error_diff
    self.prev_error = error

    return (self.p * error) + (self.i * self.error_integral) + (self.d * self.smoothed_error_diff)
