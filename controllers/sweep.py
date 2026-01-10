"""
Sweep controller for linearization.
Applies sinusoidal steering commands to map steer_command -> lat_accel.
"""
from . import BaseController
import numpy as np

class Controller(BaseController):
  def __init__(self):
    self.step = 0
    self.amplitude = 1.0  # Max steering amplitude
    self.period = 100     # Steps per full sine wave
    self._log = {}

  def get_log(self):
    return self._log

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Apply sinusoidal steering command
    phase = (self.step / self.period) * 2 * np.pi
    u = self.amplitude * np.sin(phase)

    self._log = {
      'sweep_step': self.step,
      'sweep_phase': phase,
      'sweep_u': u,
    }

    self.step += 1
    return u
