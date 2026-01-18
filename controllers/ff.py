from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Feedforward controller with impulse response horizon tracking
  """
  def __init__(self):
    self.k_lat = 1.0
    self.k_roll = 1.0
    self.H = 10
    # Impulse response: effect of unit input at t+0, t+1, t+2, ...
    self.impulse = np.array([0.2, 0.2, 0.3, 0.1, 0.1, 0.1, 0, 0, 0, 0])
    # Horizon: predicted accel at future steps from past inputs
    self.horizon = np.zeros(self.H)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # predicted command
    cmd_des = (self.k_lat * target_lataccel) - (self.k_roll * state.roll_lataccel)

    # predicted accel from past inputs
    a_predicted = self.horizon[0]

    # increment horizon based on cmd
    self.horizon += (self.impulse * cmd_des)

    # Compute u to hit target: a_predicted + u * impulse[0] = cmd_des
    u = (cmd_des - a_predicted) / (self.impulse[0] + 1e-6)

    # Shift horizon (time advances)
    self.horizon = np.roll(self.horizon, -1)
    self.horizon[-1] = 0

    # Add effect of current u to future timesteps
    # u applied now has impulse[k+1] effect at t+1+k
    for k in range(self.H - 1):
      self.horizon[k] += u * self.impulse[k + 1]

    return u
