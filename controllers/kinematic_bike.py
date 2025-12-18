from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  PURE FEEDFORWARD bicycle model
  --------------------------------
  Inputs:
    - target_lataccel (m/s^2)
    - state.v_ego (m/s)

  Output:
    - steer command

  No feedback, no memory, no integration.
  """

  def __init__(self):
    # --- vehicle geometry ---
    self.L = 2.8        # wheelbase (m) — tune once

    # --- understeer model ---
    self.Ku = 0.0025    # understeer gradient (s^2/m^2)

    # --- steering unit conversion ---
    self.k_steer = 1.0  # global gain (this is the key scalar)

    # --- limits (purely safety) ---
    self.steer_max = 1.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    v = max(state.v_ego, 0.5)
    ay = target_lataccel
    delta_rad = self.L * ay / (v*v)          # kinematic bicycle
    steer = self.k_steer * delta_rad
    steer = np.clip(steer, -2.0, 2.0)
    return steer
