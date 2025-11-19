from . import BaseController
import numpy as np
from scipy import linalg


class Controller(BaseController):
  """
  Linear Quadratic Regulator with roll compensation.
  
  State space model for lateral dynamics:
  x = [lataccel, lataccel_rate]
  u = steer_command
  
  The controller compensates for road roll and minimizes:
  J = integral(x'Qx + u'Ru)dt
  """
  
  def __init__(self):
    # System dynamics (simple double integrator approximation)
    self.dt = 0.1  # 10Hz
    
    # State: [lateral_accel, lateral_accel_rate]
    # Discretized system: x[k+1] = A*x[k] + B*u[k]
    self.A = np.array([
      [1.0, self.dt],
      [0.0, 0.95]   # slight damping on rate
    ])
    
    self.B = np.array([
      [0.0],
      [3.5]  # gain from steer to lataccel rate
    ])
    
    # LQR cost matrices
    # Q penalizes state error, R penalizes control effort
    Q = np.array([
      [100.0, 0.0],   # heavily penalize position error
      [0.0, 1.0]      # lightly penalize rate error
    ])
    
    R = np.array([[0.5]])  # control effort penalty
    
    # Solve discrete-time algebraic Riccati equation
    self.K = self._compute_lqr_gain(self.A, self.B, Q, R)
    
    # State estimation
    self.lataccel_rate = 0.0
    self.prev_lataccel = 0.0
    
    # Roll compensation parameters
    self.roll_compensation_gain = 1.0  # can tune this
    
    # Integrator for steady-state error
    self.error_integral = 0.0
    self.ki = 0.05  # integral gain
    
  def _compute_lqr_gain(self, A, B, Q, R):
    """
    Solve the discrete-time LQR problem.
    Returns optimal gain matrix K where u = -K*x
    """
    # Solve DARE (Discrete-time Algebraic Riccati Equation)
    P = linalg.solve_discrete_are(A, B, Q, R)
    
    # Compute LQR gain
    K = linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    return K
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute optimal control with roll compensation.
    
    Args:
      target_lataccel: Desired lateral acceleration
      current_lataccel: Current lateral acceleration
      state: Vehicle state (roll_lataccel, v_ego, a_ego)
      future_plan: Future trajectory plan
    
    Returns:
      Optimal steering command
    """
    # Extract roll lateral acceleration from state
    roll_lataccel = state.roll_lataccel
    
    # Compensate target for road roll
    # The car naturally experiences roll_lataccel from road banking,
    # so we adjust our target to account for this
    compensated_target = target_lataccel - self.roll_compensation_gain * roll_lataccel
    compensated_current = current_lataccel - self.roll_compensation_gain * roll_lataccel
    
    # Estimate lateral acceleration rate
    self.lataccel_rate = (compensated_current - self.prev_lataccel) / self.dt
    self.prev_lataccel = compensated_current
    
    # Compute error state
    error_lataccel = compensated_target - compensated_current
    error_rate = -self.lataccel_rate  # we want rate to be zero at steady state
    
    # State vector [error_lataccel, error_rate]
    x_error = np.array([[error_lataccel], [error_rate]])
    
    # LQR control law: u = K * x_error
    u_lqr = (self.K @ x_error)[0, 0]
    
    # Add integral term for steady-state error elimination
    self.error_integral += error_lataccel
    # Anti-windup: limit integral
    self.error_integral = np.clip(self.error_integral, -50, 50)
    u_integral = self.ki * self.error_integral
    
    # Total control
    control = u_lqr + u_integral
    
    return control

