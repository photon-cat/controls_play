from . import BaseController
import numpy as np
from scipy.optimize import minimize

# Constants from tinyphysics
DEL_T = 0.1
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5

class Controller(BaseController):
    """
    Linear MPC controller using SLSQP optimizer.

    First-order lag model: applying steer "force" ripples over multiple timesteps.
    lataccel[t+1] = lataccel[t] + alpha * (gain * steer[t] + roll[t] - lataccel[t])
    """
    def __init__(self):
        # Model parameters (fitted from simulation data)
        self.gain = 1.1        # steer to steady-state lateral accel gain
        self.alpha = 0.19      # response rate (smaller = more lag, more timesteps to settle)

        # MPC parameters
        self.horizon = 15      # longer horizon to see through the lag

        # Cost weights (match scoring: 50:1 ratio)
        self.tracking_weight = 50.0
        self.jerk_weight = 1.0
        self.steer_rate_weight = 5.0  # penalize steer changes for smoothness

        # State
        self.prev_steer = 0.0
        self.prev_solution = None

    def predict(self, current_lataccel, steer_seq, roll_seq):
        """
        Predict lateral acceleration trajectory with first-order lag dynamics.

        The steer command acts as a "force" that takes multiple timesteps to
        fully realize its effect on lataccel.

        Args:
            current_lataccel: Current lateral acceleration
            steer_seq: Sequence of steer commands (length = horizon)
            roll_seq: Sequence of roll lateral accelerations (length = horizon)

        Returns:
            Predicted lataccel trajectory (length = horizon)
        """
        pred = np.zeros(len(steer_seq))
        lataccel = current_lataccel

        for t in range(len(steer_seq)):
            # Steady-state response if we held this steer forever
            steady_state = self.gain * steer_seq[t] + roll_seq[t]

            # First-order lag: move toward steady state by alpha fraction
            delta = self.alpha * (steady_state - lataccel)

            # Rate limit (from physics model)
            delta = np.clip(delta, -MAX_ACC_DELTA, MAX_ACC_DELTA)
            lataccel = lataccel + delta

            pred[t] = lataccel

        return pred

    def cost_function(self, steer_seq, current_lataccel, target_seq, roll_seq, prev_steer):
        """
        Compute MPC cost matching the scoring function.
        """
        pred = self.predict(current_lataccel, steer_seq, roll_seq)

        # Tracking cost: (target - pred)^2
        tracking_cost = np.sum((target_seq - pred) ** 2)

        # Jerk cost: (pred[t] - pred[t-1])^2 / DEL_T^2
        jerk = np.diff(pred) / DEL_T
        jerk_cost = np.sum(jerk ** 2)

        # Also penalize first jerk from current state
        first_jerk = (pred[0] - current_lataccel) / DEL_T
        jerk_cost += first_jerk ** 2

        # Steer rate cost: penalize changes in steer command
        steer_rate = np.diff(np.concatenate([[prev_steer], steer_seq]))
        steer_rate_cost = np.sum(steer_rate ** 2)

        total_cost = (self.tracking_weight * tracking_cost +
                      self.jerk_weight * jerk_cost +
                      self.steer_rate_weight * steer_rate_cost)
        return total_cost

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Build target and roll sequences
        horizon = min(self.horizon, len(future_plan.lataccel) + 1)
        target_seq = np.array([target_lataccel] + future_plan.lataccel[:horizon-1])
        roll_seq = np.array([state.roll_lataccel] + future_plan.roll_lataccel[:horizon-1])

        # Ensure sequences are same length
        min_len = min(len(target_seq), len(roll_seq))
        target_seq = target_seq[:min_len]
        roll_seq = roll_seq[:min_len]
        horizon = min_len

        # Initial guess: warm start or zeros
        if self.prev_solution is not None and len(self.prev_solution) >= horizon:
            x0 = np.concatenate([self.prev_solution[1:horizon], [self.prev_solution[-1]]])
        else:
            x0 = np.zeros(horizon)

        # Bounds for steer commands
        bounds = [(STEER_RANGE[0], STEER_RANGE[1]) for _ in range(horizon)]

        # Optimize
        result = minimize(
            self.cost_function,
            x0,
            args=(current_lataccel, target_seq, roll_seq, self.prev_steer),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )

        # Save for warm start
        self.prev_solution = result.x

        # Update state
        steer_cmd = result.x[0]
        self.prev_steer = steer_cmd

        return steer_cmd
