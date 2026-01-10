from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  Hybrid Feedforward + Extremum Seeking Controller
  Uses direct error for fast response, ESC for optimization
  """
  def __init__(self):
    # Feedforward gain (direct error response)
    self.kp = 0.15
    
    # Dither signal parameters
    self.omega = 12.0
    self.a = 0.003
    
    # ESC adaptation gain
    self.k = 0.5
    
    # Filter cutoffs
    self.omega_l = 1.5
    self.omega_h = 1.5
    
    # Internal states
    self.theta_hat = 0.0     # ESC correction term
    self.xi = 0.0
    self.eta = 0.0
    
    # For jerk calculation
    self.prev_lataccel = 0.0
    
    # Time tracking
    self.t = 0.0
    self.dt = 0.1
    
    # Cost weights
    self.lat_accel_weight = 50.0
    self.jerk_weight = 1.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = target_lataccel - current_lataccel
    
    # Feedforward: direct proportional response to error
    feedforward = self.kp * error
    
    # Jerk calculation
    jerk = (current_lataccel - self.prev_lataccel) / self.dt
    self.prev_lataccel = current_lataccel
    
    # Cost function
    lat_accel_error = error ** 2
    jerk_penalty = jerk ** 2
    cost = (self.lat_accel_weight * lat_accel_error) + (self.jerk_weight * jerk_penalty)
    
    # High-pass filter
    y_hp = cost - self.eta
    self.eta += self.omega_h * self.dt * (cost - self.eta)
    
    # Dither
    dither = self.a * np.sin(self.omega * self.t)
    
    # Demodulation and low-pass filter
    demod = y_hp * dither
    self.xi += self.omega_l * self.dt * (demod - self.xi)
    
    # ESC update (adaptive trim)
    self.theta_hat -= self.k * self.dt * self.xi
    self.theta_hat = np.clip(self.theta_hat, -0.5, 0.5)
    
    # Combined output: feedforward + ESC trim + dither
    steer_command = feedforward + self.theta_hat + dither
    
    self.t += self.dt
    
    return steer_command