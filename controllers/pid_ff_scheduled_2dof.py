"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll (not used)

added feedforwaerd prediction  
  

"""
from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  Gain-scheduled 2-DOF PID + preview FF in lateral-accel domain
  """
  def __init__(self):
    self.v_points = np.arange(0, 41, 4)

    self.p_points = np.array([0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.23])
    self.i_points = np.array([0.100, 0.10, 0.11, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.03])
    self.d_points = np.array([-0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.001])

    self.k_ff_points = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5])

    # NEW: 2-DOF reference weight for P (beta)
    # start conservative at high speed to reduce reference-chasing oscillations
    self.beta_points = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.5])

    self.error_integral = 0.0
    self.prev_y = 0.0  # previous measurement for D-on-measurement
    self.preview_steps = 3

    self.u_prev = 0.0 
    self.u_alpha = 0.5   # smoothing factor (0.05–0.25 typical)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # preview reference
    if future_plan is not None and hasattr(future_plan, "lataccel") and len(future_plan.lataccel) > self.preview_steps:
      r = float(future_plan.lataccel[self.preview_steps])
    else:
      r = float(target_lataccel)

    y = float(current_lataccel)
    v = float(state.v_ego)

    # scheduled gains
    kp = np.interp(v, self.v_points, self.p_points)
    ki = np.interp(v, self.v_points, self.i_points)
    kd = np.interp(v, self.v_points, self.d_points)
    kf = np.interp(v, self.v_points, self.k_ff_points)
    beta = np.interp(v, self.v_points, self.beta_points)

    # feed-forward (unchanged)
    u_ff = kf * r

    # 2-DOF PID
    e_i = (r - y)                 # integral tracks true error
    e_p = (beta * r - y)          # proportional sees weighted reference
    dy  = (y - self.prev_y)       # derivative on measurement only
    self.prev_y = y

    self.error_integral += e_i

    u_fb = kp * e_p + ki * self.error_integral + kd * dy

    u_raw = u_ff + u_fb

    u = (1.0 - self.u_alpha) * self.u_prev + self.u_alpha * u_raw
    self.u_prev = u
    return u
