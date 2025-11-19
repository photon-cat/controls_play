from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  PID controller with roll compensation.
  
  Compensates for lateral acceleration induced by road banking/roll
  before computing control error.
  """
  
  def __init__(self):
    # PID gains (tuned for the system)
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    
    # State tracking
    self.error_integral = 0
    self.prev_error = 0
    
    # Roll compensation gain
    # 1.0 = full compensation, 0.0 = no compensation
    self.roll_compensation_gain = 0.8
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute PID control with roll compensation.
    
    The idea: road roll causes lateral acceleration that we don't need to
    counteract with steering. By subtracting it from both target and current,
    we focus control effort on the actual tracking error.
    """
    # Extract roll lateral acceleration
    roll_lataccel = state.roll_lataccel
    
    # Apply roll compensation
    compensated_target = target_lataccel - self.roll_compensation_gain * roll_lataccel
    compensated_current = current_lataccel - self.roll_compensation_gain * roll_lataccel
    
    # Compute error in compensated frame
    error = compensated_target - compensated_current
    
    # PID terms
    self.error_integral += error
    # Anti-windup: limit integral
    self.error_integral = np.clip(self.error_integral, -100, 100)
    
    error_diff = error - self.prev_error
    self.prev_error = error
    
    # PID control law
    control = self.p * error + self.i * self.error_integral + self.d * error_diff
    
    return control

