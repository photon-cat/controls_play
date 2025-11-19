from . import BaseController
import numpy as np
from scipy.optimize import minimize


class Controller(BaseController):
  """
  Model Predictive Controller - uses future plan for optimal lookahead control.
  
  Strategy:
  1. Use 50-step future plan to anticipate trajectory
  2. Optimize control sequence over prediction horizon
  3. Account for roll disturbance compensation
  4. Minimize weighted sum of tracking error and jerk
  5. Respect actuator constraints
  
  This is closer to "perfect" control as it uses all available information.
  """
  
  def __init__(self):
    self.dt = 0.1
    
    # MPC parameters
    self.horizon = 10  # Optimize over 10 steps (1 second)
    self.control_horizon = 5  # Only change control for first 5 steps
    
    # System identification (learned from data/model)
    self.steer_to_lataccel_gain = 2.8  # rough estimate
    self.lataccel_time_constant = 0.15  # how fast lataccel responds
    
    # State tracking
    self.prev_lataccel = 0.0
    self.prev_steer = 0.0
    self.lataccel_rate = 0.0
    
    # Roll compensation (adaptive)
    self.roll_compensation_gain = 0.85
    
    # Cost weights (tuned for the objective function)
    self.tracking_weight = 50.0  # matches LAT_ACCEL_COST_MULTIPLIER
    self.jerk_weight = 1.0
    self.control_effort_weight = 0.1
    self.control_rate_weight = 2.0  # penalize control changes
    
    # Constraints
    self.steer_limit = 2.0
    self.max_acc_delta = 0.5
    
  def predict_lataccel(self, current_lataccel, steer_action, roll_lataccel):
    """
    Simple first-order model of lateral acceleration response to steering.
    
    lataccel_next = lataccel + (steer * gain - lataccel) / time_constant
    """
    # Target from steer command (compensate for roll)
    target_from_steer = steer_action * self.steer_to_lataccel_gain
    
    # First-order response
    delta_lataccel = (target_from_steer - current_lataccel) * self.dt / self.lataccel_time_constant
    
    # Clip to physical constraint
    delta_lataccel = np.clip(delta_lataccel, -self.max_acc_delta, self.max_acc_delta)
    
    next_lataccel = current_lataccel + delta_lataccel
    
    return next_lataccel
  
  def cost_function(self, control_sequence, current_lataccel, target_sequence, roll_sequence):
    """
    MPC cost function to minimize over prediction horizon.
    
    Args:
      control_sequence: Array of steer commands [u_0, u_1, ..., u_H]
      current_lataccel: Current lateral acceleration
      target_sequence: Target lataccel for next H steps
      roll_sequence: Roll lataccel for next H steps
    
    Returns:
      Total cost (tracking error + jerk + control effort)
    """
    cost = 0.0
    lataccel = current_lataccel
    prev_control = self.prev_steer
    
    lataccel_history = [lataccel]
    
    for i in range(len(control_sequence)):
      # Clip control to limits
      steer = np.clip(control_sequence[i], -self.steer_limit, self.steer_limit)
      
      # Predict next lataccel
      roll_comp = self.roll_compensation_gain * roll_sequence[i] if i < len(roll_sequence) else 0
      lataccel = self.predict_lataccel(lataccel - roll_comp, steer, roll_sequence[i] if i < len(roll_sequence) else 0)
      lataccel_history.append(lataccel)
      
      # Tracking error cost
      if i < len(target_sequence):
        tracking_error = target_sequence[i] - lataccel
        cost += self.tracking_weight * tracking_error**2
      
      # Control effort cost
      cost += self.control_effort_weight * steer**2
      
      # Control rate cost (penalize big steering changes)
      control_rate = (steer - prev_control) / self.dt
      cost += self.control_rate_weight * control_rate**2
      prev_control = steer
    
    # Jerk cost (rate of change of lataccel)
    lataccel_array = np.array(lataccel_history)
    jerk = np.diff(lataccel_array) / self.dt
    cost += self.jerk_weight * np.sum(jerk**2)
    
    return cost
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    MPC control update - optimize control sequence using future plan.
    
    Args:
      target_lataccel: Current target lateral acceleration
      current_lataccel: Current actual lateral acceleration
      state: Vehicle state (roll_lataccel, v_ego, a_ego)
      future_plan: Future trajectory plan (50 steps ahead)
    
    Returns:
      Optimal steering command
    """
    # Extract roll compensation
    roll_lataccel = state.roll_lataccel
    
    # Build target and roll sequences from future plan
    if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > 0:
      # Use future plan for lookahead
      target_sequence = np.array([target_lataccel] + future_plan.lataccel[:self.horizon-1])
      roll_sequence = np.array([roll_lataccel] + future_plan.roll_lataccel[:self.horizon-1])
    else:
      # Fallback if no future plan
      target_sequence = np.ones(self.horizon) * target_lataccel
      roll_sequence = np.ones(self.horizon) * roll_lataccel
    
    # Ensure we have enough elements
    if len(target_sequence) < self.horizon:
      target_sequence = np.pad(target_sequence, (0, self.horizon - len(target_sequence)), 
                                mode='edge')
    if len(roll_sequence) < self.horizon:
      roll_sequence = np.pad(roll_sequence, (0, self.horizon - len(roll_sequence)), 
                             mode='edge')
    
    # Initial guess for control sequence (warm start from previous)
    initial_guess = np.ones(self.control_horizon) * self.prev_steer
    
    # Optimize control sequence
    bounds = [(-self.steer_limit, self.steer_limit)] * self.control_horizon
    
    result = minimize(
      self.cost_function,
      initial_guess,
      args=(current_lataccel, target_sequence[:self.control_horizon], 
            roll_sequence[:self.control_horizon]),
      method='SLSQP',
      bounds=bounds,
      options={'maxiter': 20, 'ftol': 1e-4}  # Quick optimization
    )
    
    # Extract first control action (receding horizon)
    optimal_steer = result.x[0] if result.success else initial_guess[0]
    
    # Store for next iteration
    self.prev_steer = optimal_steer
    self.prev_lataccel = current_lataccel
    
    return optimal_steer

