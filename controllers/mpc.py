from . import BaseController
import numpy as np
from collections import deque

CONTEXT_LENGTH = 20

class Controller(BaseController):
  def __init__(self):
    # PID baseline (keep your tuning)
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053

    self.error_integral = 0.0
    self.prev_error = 0.0

    # MPC params
    self.H = 5          # Horizon (shorter for speed)
    self.K = 20         # Number of samples

    self.w_e  = 1.0     # Tracking error weight
    self.w_u  = 0.02    # Control effort weight
    self.w_du = 0.3     # Control rate weight
    self.w_j  = 1.5     # Jerk weight

    self.prev_u = 0.0

    # Physics model (will be set externally)
    self.physics_model = None

    # History for physics simulation
    self.state_history = deque(maxlen=CONTEXT_LENGTH)
    self.action_history = deque(maxlen=CONTEXT_LENGTH)
    self.lataccel_history = deque(maxlen=CONTEXT_LENGTH)

    self.initialized = False

    # Logging
    self._log = {}

  def set_physics_model(self, model):
    """Set the physics model for MPC simulation"""
    self.physics_model = model

  def get_log(self):
    """Return log data for current timestep"""
    return self._log

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Initialize history on first call
    if not self.initialized:
      for _ in range(CONTEXT_LENGTH):
        self.state_history.append(state)
        self.action_history.append(0.0)
        self.lataccel_history.append(current_lataccel)
      self.initialized = True

    # ----- baseline PID -----
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    u_base = (
      self.p * error +
      self.i * self.error_integral +
      self.d * error_diff
    )

    # If physics model not available, just use PID
    if self.physics_model is None:
      self.prev_u = u_base
      self.state_history.append(state)
      self.action_history.append(u_base)
      self.lataccel_history.append(current_lataccel)
      self._log = {
        'mpc_u_base': u_base,
        'mpc_best_cost': 0.0,
        'mpc_best_u': u_base,
        'mpc_error': error,
        'mpc_mode': 'pid_only',
      }
      return u_base

    # ----- MPC optimization -----
    # Extract future plan lataccel if available
    if hasattr(future_plan, 'lataccel') and future_plan.lataccel:
      future_lataccel = future_plan.lataccel
    else:
      future_lataccel = []

    # Sample control perturbations
    best_cost = np.inf
    best_u = u_base

    for _ in range(self.K):
      # Sample a smooth control sequence
      du_seq = np.random.uniform(-0.1, 0.1, self.H)
      # Apply smoothing
      du_seq = np.cumsum(du_seq) / np.arange(1, self.H + 1)

      # Simulate forward
      u = u_base
      sim_lataccel = current_lataccel
      cost = 0.0
      prev_ay = current_lataccel

      # Build simulation history (copy current history)
      sim_states = list(self.state_history)
      sim_actions = list(self.action_history)
      sim_preds = list(self.lataccel_history)

      for i in range(self.H):
        # Apply control perturbation
        u = u_base + du_seq[i]
        u = np.clip(u, -2.0, 2.0)

        # Update histories
        sim_states.append(state)  # Use current state (simplified)
        sim_actions.append(u)
        sim_preds.append(sim_lataccel)

        # Keep only last CONTEXT_LENGTH
        if len(sim_states) > CONTEXT_LENGTH:
          sim_states.pop(0)
          sim_actions.pop(0)
          sim_preds.pop(0)

        # Predict next lataccel using physics model
        try:
          sim_lataccel = self.physics_model.get_current_lataccel(
            sim_states=sim_states,
            actions=sim_actions,
            past_preds=sim_preds
          )
          # Clip acceleration change
          sim_lataccel = np.clip(sim_lataccel,
                                 prev_ay - 0.5,
                                 prev_ay + 0.5)
        except:
          # If simulation fails, fall back to simple model
          sim_lataccel = prev_ay + (u * 2.0 - prev_ay) * 0.3

        # Get reference
        ay_ref = future_lataccel[i] if i < len(future_lataccel) else target_lataccel

        # Compute cost
        cost += self.w_e * (sim_lataccel - ay_ref)**2
        cost += self.w_u * u**2
        cost += self.w_du * du_seq[i]**2
        cost += self.w_j * (sim_lataccel - prev_ay)**2

        prev_ay = sim_lataccel

        # Early termination if cost too high
        if cost > best_cost * 1.5:
          break

      # Update best
      if cost < best_cost:
        best_cost = cost
        best_u = u_base + du_seq[0]

    # Apply control with smoothing
    u_cmd = 0.8 * self.prev_u + 0.2 * best_u
    u_cmd = np.clip(u_cmd, -2.0, 2.0)

    # Update histories
    self.state_history.append(state)
    self.action_history.append(u_cmd)
    self.lataccel_history.append(current_lataccel)

    # Log MPC data
    self._log = {
      'mpc_u_base': u_base,
      'mpc_best_cost': best_cost,
      'mpc_best_u': best_u,
      'mpc_error': error,
      'mpc_mode': 'mpc',
    }

    self.prev_u = u_cmd
    return u_cmd
