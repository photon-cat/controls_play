from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Best Feedforward PID Controller from optimization run.
  
  Parameters from lowest cost result (137.60):
  kp=0.2885, ki=0.0296, kd=-0.1636, kff=0.0476, 
  kff_accel=0.0701, roll_comp_gain=0.6529, derivative_alpha=0.6790
  """
  
  def __init__(self):
    # PID gains (from optimization)
    self.kp = 0.2885
    self.ki = 0.0296
    self.kd = -0.1636
    
    # Feedforward gains (from optimization)
    self.kff = 0.0476
    self.kff_accel = 0.0701
    
    # Roll compensation (from optimization)
    self.roll_comp_gain = 0.6529
    
    # Derivative filtering (from optimization)
    self.derivative_alpha = 0.6790
    
    # State tracking
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_target = 0.0
    self.filtered_derivative = 0.0
    
    # Look-ahead for anticipation
    self.lookahead_steps = 3
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """PID control with feedforward anticipation from future plan."""
    # Roll compensation
    roll_lataccel = state.roll_lataccel
    compensated_target = target_lataccel - self.roll_comp_gain * roll_lataccel
    compensated_current = current_lataccel - self.roll_comp_gain * roll_lataccel
    
    # Error signal
    error = compensated_target - compensated_current
    
    # Integral with anti-windup
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -100, 100)
    
    # Filtered derivative
    raw_derivative = error - self.prev_error
    self.filtered_derivative = (self.derivative_alpha * raw_derivative + 
                                (1 - self.derivative_alpha) * self.filtered_derivative)
    
    # PID control
    u_pid = self.kp * error + self.ki * self.error_integral + self.kd * self.filtered_derivative
    
    # Feedforward from future plan
    u_feedforward = 0.0
    if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > 0:
      if self.lookahead_steps < len(future_plan.lataccel):
        future_target = future_plan.lataccel[self.lookahead_steps]
        target_rate = (future_target - target_lataccel) / (self.lookahead_steps * 0.1)
        u_feedforward += self.kff * target_rate
      
      if len(future_plan.a_ego) > self.lookahead_steps:
        future_accel = future_plan.a_ego[self.lookahead_steps]
        current_accel = state.a_ego
        accel_change = future_accel - current_accel
        u_feedforward += self.kff_accel * accel_change
    
    # Total control
    control = u_pid + u_feedforward
    
    # Update state
    self.prev_error = error
    self.prev_target = target_lataccel
    
    return control

