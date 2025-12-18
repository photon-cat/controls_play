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
  PID controller with preview feed-forward for lateral acceleration tracking
  """
  def __init__(self):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053

    self.k_ff = 0.2    # feed-forward gain (tune this first)

    self.error_integral = 0.0
    self.prev_error = 0.0

    self.preview_steps = 3

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # --- preview target ---
    if future_plan is not None and hasattr(future_plan, "lataccel") and len(future_plan.lataccel) > self.preview_steps:
      a_ref = future_plan.lataccel[self.preview_steps]
    else:
      a_ref = target_lataccel

    # --- feed-forward ---
    u_ff = self.k_ff * a_ref

    # --- feedback ---
    error = a_ref - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    u_fb = (
      self.p * error +
      self.i * self.error_integral +
      self.d * error_diff
    )

    return u_ff + u_fb
