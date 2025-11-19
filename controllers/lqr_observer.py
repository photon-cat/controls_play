from . import BaseController
import numpy as np
from scipy import linalg
from collections import defaultdict


class RollDisturbanceObserver:
  """
  Adaptive disturbance observer that learns roll compensation.
  
  Maintains a lookup table mapping roll_lataccel -> observed disturbance effect.
  Updates online based on tracking error correlation with roll.
  """
  
  def __init__(self, num_bins=20, learning_rate=0.01):
    self.num_bins = num_bins
    self.learning_rate = learning_rate
    
    # Compensation table: maps roll_lataccel bins to compensation gains
    self.roll_range = [-5.0, 5.0]  # m/s^2
    self.bin_edges = np.linspace(self.roll_range[0], self.roll_range[1], num_bins + 1)
    self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
    
    # Compensation gains per bin (learned online)
    self.compensation_table = np.ones(num_bins) * 0.8  # start with 0.8 gain
    
    # Statistics for adaptation
    self.bin_counts = np.zeros(num_bins)
    self.bin_errors = np.zeros(num_bins)
    
    # History for observer
    self.error_history = []
    self.roll_history = []
    self.max_history = 50
    
  def get_bin_index(self, roll_lataccel):
    """Get bin index for given roll lateral acceleration."""
    idx = np.digitize(roll_lataccel, self.bin_edges) - 1
    return np.clip(idx, 0, self.num_bins - 1)
  
  def get_compensation_gain(self, roll_lataccel):
    """Lookup compensation gain from table."""
    idx = self.get_bin_index(roll_lataccel)
    return self.compensation_table[idx]
  
  def update_observer(self, roll_lataccel, tracking_error, control_effort):
    """
    Update disturbance observer with new observation.
    
    The idea: if we have tracking error correlated with roll, 
    adjust the compensation gain to reduce this correlation.
    """
    # Store history
    self.error_history.append(tracking_error)
    self.roll_history.append(roll_lataccel)
    
    if len(self.error_history) > self.max_history:
      self.error_history.pop(0)
      self.roll_history.pop(0)
    
    # Update bin statistics
    idx = self.get_bin_index(roll_lataccel)
    self.bin_counts[idx] += 1
    self.bin_errors[idx] += abs(tracking_error)
    
    # Adaptive learning: adjust compensation if error is correlated with roll
    if len(self.error_history) >= 10 and self.bin_counts[idx] > 5:
      # If tracking error is positive when roll is present, increase compensation
      # If error is negative, decrease compensation
      mean_error = self.bin_errors[idx] / self.bin_counts[idx]
      
      # Gradient descent on compensation gain
      # More error -> increase compensation
      gain_adjustment = self.learning_rate * np.sign(tracking_error * roll_lataccel)
      self.compensation_table[idx] += gain_adjustment
      
      # Bound compensation gains to reasonable range
      self.compensation_table[idx] = np.clip(self.compensation_table[idx], 0.0, 1.5)
  
  def get_table_summary(self):
    """Return compensation table for debugging."""
    return {
      'bin_centers': self.bin_centers.tolist(),
      'compensation_gains': self.compensation_table.tolist(),
      'bin_counts': self.bin_counts.tolist(),
    }


class Controller(BaseController):
  """
  LQR controller with adaptive roll disturbance observer and compensation table.
  
  Features:
  - Optimal LQR control for lateral tracking
  - Disturbance observer that learns roll compensation online
  - Lookup table for adaptive feedforward compensation
  """
  
  def __init__(self):
    # System dynamics
    self.dt = 0.1  # 10Hz
    
    # State: [lateral_accel_error, lateral_accel_rate]
    self.A = np.array([
      [1.0, self.dt],
      [0.0, 0.85]   # more damping for stability
    ])
    
    self.B = np.array([
      [0.0],
      [1.0]  # much lower gain for smooth control
    ])
    
    # LQR cost matrices - heavily tuned for low jerk
    Q = np.array([
      [50.0, 0.0],   # position error weight (reduced)
      [0.0, 0.1]     # rate error weight (low = smooth)
    ])
    
    R = np.array([[5.0]])  # high control penalty = very smooth control
    
    # Compute LQR gain
    self.K = self._compute_lqr_gain(self.A, self.B, Q, R)
    
    # State estimation
    self.lataccel_rate = 0.0
    self.prev_lataccel = 0.0
    
    # Roll disturbance observer with compensation table
    self.observer = RollDisturbanceObserver(num_bins=25, learning_rate=0.005)
    
    # Integrator for steady-state error
    self.error_integral = 0.0
    self.ki = 0.03  # reduced for stability
    
    # Debugging/logging
    self.step_count = 0
    self.log_interval = 100
    
  def _compute_lqr_gain(self, A, B, Q, R):
    """Solve discrete-time LQR problem."""
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute optimal control with adaptive roll compensation.
    
    Args:
      target_lataccel: Desired lateral acceleration
      current_lataccel: Current lateral acceleration  
      state: Vehicle state (roll_lataccel, v_ego, a_ego)
      future_plan: Future trajectory plan
    
    Returns:
      Optimal steering command with disturbance compensation
    """
    self.step_count += 1
    
    # Extract roll lateral acceleration
    roll_lataccel = state.roll_lataccel
    
    # Get adaptive compensation gain from observer's lookup table
    roll_compensation_gain = self.observer.get_compensation_gain(roll_lataccel)
    
    # Apply adaptive roll compensation
    compensated_target = target_lataccel - roll_compensation_gain * roll_lataccel
    compensated_current = current_lataccel - roll_compensation_gain * roll_lataccel
    
    # Estimate lateral acceleration rate
    self.lataccel_rate = (compensated_current - self.prev_lataccel) / self.dt
    self.prev_lataccel = compensated_current
    
    # Compute error state
    error_lataccel = compensated_target - compensated_current
    error_rate = -self.lataccel_rate
    
    # State vector
    x_error = np.array([[error_lataccel], [error_rate]])
    
    # LQR control law
    u_lqr = (self.K @ x_error)[0, 0]
    
    # Integral term with anti-windup
    self.error_integral += error_lataccel
    self.error_integral = np.clip(self.error_integral, -30, 30)
    u_integral = self.ki * self.error_integral
    
    # Total control
    control = u_lqr + u_integral
    
    # Update disturbance observer (learn from this experience)
    # Track uncompensated error to improve roll compensation
    raw_error = target_lataccel - current_lataccel
    self.observer.update_observer(roll_lataccel, raw_error, control)
    
    # Periodic logging of compensation table
    if self.step_count % self.log_interval == 0:
      table = self.observer.get_table_summary()
      # Print sample of compensation gains
      print(f"Step {self.step_count}: Roll compensation gains (sampled): "
            f"{[f'{g:.3f}' for g in table['compensation_gains'][::5]]}")
    
    return control

