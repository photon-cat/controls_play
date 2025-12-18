"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll (not used)

added feedforwaerd prediction  
added finite horizon integral  

"""

from . import BaseController
from collections import deque

class Controller(BaseController):
  """
  PID controller with preview feed-forward for lateral acceleration tracking
  """
  def __init__(self):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053

    # Feedforward (keep this!)
    self.k_ff = 0.2
    self.preview_steps = 3

    # Sliding-window integral
    self.i_window = 15              # try 10–15
    self.err_hist = deque(maxlen=self.i_window)

    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # --- preview reference ---
    if future_plan is not None and hasattr(future_plan, "lataccel") and len(future_plan.lataccel) > self.preview_steps:
      a_ref = future_plan.lataccel[self.preview_steps]
    else:
      a_ref = target_lataccel

    # --- feed-forward ---
    u_ff = self.k_ff * a_ref

    # --- feedback error ---
    error = a_ref - current_lataccel

    # windowed integral
    self.err_hist.append(error)
    error_integral = sum(self.err_hist)

    # derivative (per-step)
    error_diff = error - self.prev_error
    self.prev_error = error

    u_fb = (
      self.p * error +
      self.i * error_integral +
      self.d * error_diff
    )

    return u_ff + u_fb
