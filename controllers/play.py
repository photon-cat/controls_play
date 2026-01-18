"""
Adaptive MPC controller with online system identification.
- Estimates dynamics from recent observations using recursive least squares
- Uses identified model for MPC prediction
- Adapts quickly to different segment dynamics
"""
from . import BaseController
from collections import deque
import numpy as np

DEL_T = 0.1
HORIZON = 10
HISTORY_LEN = 20  # Steps for system ID

# Constraints
MAX_U = 2.0
MAX_RATE = 0.5

# System ID parameters
FORGETTING_FACTOR = 0.95  # Exponential forgetting for RLS


class Controller(BaseController):
    def __init__(self):
        self.prev_u = 0.0
        self._log = {}

        # History for system identification
        # Store: (steer_cmd, lataccel, roll, v_ego)
        self.history = deque(maxlen=HISTORY_LEN)

        # Recursive Least Squares state
        # Model: next_lataccel = a * current_lataccel + b * steer + c * roll + d
        # Theta = [a, b, c, d]
        self.theta = np.array([0.9, 1.5, 1.0, 0.0])  # Initial guess
        self.P = np.eye(4) * 10.0  # Covariance matrix

        # For debugging
        self.step_count = 0

    def get_log(self):
        return self._log

    def _update_sysid(self, prev_lataccel, prev_steer, prev_roll, current_lataccel):
        """Update system ID using recursive least squares."""
        # Feature vector: [prev_lataccel, prev_steer, prev_roll, 1]
        phi = np.array([prev_lataccel, prev_steer, prev_roll, 1.0])

        # RLS update with forgetting factor
        y = current_lataccel
        y_pred = phi @ self.theta

        # Kalman-like gain
        P_phi = self.P @ phi
        denom = FORGETTING_FACTOR + phi @ P_phi
        K = P_phi / denom

        # Update estimate
        error = y - y_pred
        self.theta = self.theta + K * error

        # Update covariance
        self.P = (self.P - np.outer(K, phi @ self.P)) / FORGETTING_FACTOR

        # Keep P bounded to avoid numerical issues
        self.P = np.clip(self.P, -100, 100)

        return error

    def _predict_lataccel(self, current_lataccel, steer, roll):
        """Predict next lataccel using identified model."""
        phi = np.array([current_lataccel, steer, roll, 1.0])
        return phi @ self.theta

    def _rollout(self, current_lataccel, u_seq, roll_seq):
        """Rollout predictions for MPC."""
        lataccels = [current_lataccel]
        for i, u in enumerate(u_seq):
            roll = roll_seq[i] if i < len(roll_seq) else roll_seq[-1]
            next_lat = self._predict_lataccel(lataccels[-1], u, roll)
            lataccels.append(next_lat)
        return np.array(lataccels[1:])

    def _mpc_cost(self, u_seq, current_lataccel, target_seq, roll_seq, prev_u):
        """Compute MPC cost."""
        n = len(u_seq)
        pred = self._rollout(current_lataccel, u_seq, roll_seq)

        # Lataccel tracking cost
        lataccel_cost = np.sum((target_seq[:n] - pred[:n]) ** 2) * 5000

        # Jerk cost
        u_full = np.concatenate([[prev_u], u_seq])
        jerk = np.diff(u_full) / DEL_T
        jerk_cost = np.sum(jerk ** 2) * 100

        return lataccel_cost + jerk_cost

    def _compute_perfect_loss(self, target_seq, prev_u):
        """Compute the minimum possible jerk cost (perfect tracking, smooth control)."""
        n = len(target_seq)
        # Perfect tracking means we'd ideally have smooth u transitions
        # Minimum jerk is achieved by staying at prev_u (jerk = 0)
        # But we need some jerk to track, so estimate based on target changes
        target_changes = np.diff(target_seq) if n > 1 else np.array([0])
        # Rough estimate: u needs to change proportionally to target changes
        estimated_u_changes = target_changes / max(self.theta[1], 0.5)  # divide by steer gain
        min_jerk = np.sum((estimated_u_changes / DEL_T) ** 2) * 100
        return min_jerk  # Perfect lataccel cost is 0

    def _optimize_mpc(self, current_lataccel, target_seq, roll_seq, prev_u):
        """Optimize control sequence until cost is within 1.1x of perfect."""
        n = min(HORIZON, len(target_seq), len(roll_seq))
        if n < 1:
            return 0.0

        # Initialize u sequence with feedforward estimate
        steer_gain = max(self.theta[1], 0.5)
        u_seq = np.zeros(n)
        for i in range(n):
            # Feedforward: u = (target - roll) / gain
            target_turning = target_seq[i] - roll_seq[i]
            u_seq[i] = target_turning / steer_gain
        u_seq = np.clip(u_seq, -MAX_U, MAX_U)

        # Apply rate limits to initial sequence
        u_seq[0] = np.clip(u_seq[0], prev_u - MAX_RATE, prev_u + MAX_RATE)
        for i in range(1, n):
            u_seq[i] = np.clip(u_seq[i], u_seq[i-1] - MAX_RATE, u_seq[i-1] + MAX_RATE)

        # Compute target cost threshold
        perfect_loss = self._compute_perfect_loss(target_seq[:n], prev_u)
        target_cost = perfect_loss * 1.1 + 10.0  # Add small buffer

        # Iterative coordinate descent optimization
        best_cost = self._mpc_cost(u_seq, current_lataccel, target_seq, roll_seq, prev_u)

        for iteration in range(100):  # Max iterations
            improved = False

            # Optimize each timestep
            for i in range(n):
                # Determine valid range for u[i]
                if i == 0:
                    u_min = max(-MAX_U, prev_u - MAX_RATE)
                    u_max = min(MAX_U, prev_u + MAX_RATE)
                else:
                    u_min = max(-MAX_U, u_seq[i-1] - MAX_RATE)
                    u_max = min(MAX_U, u_seq[i-1] + MAX_RATE)

                if i < n - 1:
                    # Also constrain based on next step
                    u_min = max(u_min, u_seq[i+1] - MAX_RATE)
                    u_max = min(u_max, u_seq[i+1] + MAX_RATE)

                # Grid search for this timestep
                best_u_i = u_seq[i]
                for u_candidate in np.linspace(u_min, u_max, 11):
                    u_seq[i] = u_candidate
                    cost = self._mpc_cost(u_seq, current_lataccel, target_seq, roll_seq, prev_u)
                    if cost < best_cost:
                        best_cost = cost
                        best_u_i = u_candidate
                        improved = True

                u_seq[i] = best_u_i

            # Check if we've reached target cost
            if best_cost <= target_cost:
                break

            # Early stop if no improvement
            if not improved:
                break

        return u_seq[0]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Update system ID from previous step
        if len(self.history) > 0:
            prev = self.history[-1]
            prev_lataccel, prev_steer, prev_roll = prev[0], prev[1], prev[2]
            sysid_error = self._update_sysid(prev_lataccel, prev_steer, prev_roll, current_lataccel)
        else:
            sysid_error = 0.0

        # Store current observation
        self.history.append((current_lataccel, self.prev_u, roll_lataccel, v_ego))

        # Build target and roll sequences
        target_seq = [target_lataccel]
        roll_seq = [roll_lataccel]
        if future_plan.lataccel:
            target_seq.extend(future_plan.lataccel[:HORIZON-1])
        if future_plan.roll_lataccel:
            roll_seq.extend(future_plan.roll_lataccel[:HORIZON-1])
        target_seq = np.array(target_seq)
        roll_seq = np.array(roll_seq)

        # MPC optimization
        u_cmd = self._optimize_mpc(current_lataccel, target_seq, roll_seq, self.prev_u)

        # Rate limit
        u_cmd = np.clip(u_cmd, self.prev_u - MAX_RATE, self.prev_u + MAX_RATE)
        u_cmd = np.clip(u_cmd, -MAX_U, MAX_U)

        # Debug output
        a, b, c, d = self.theta
        error = target_lataccel - current_lataccel
        print(f"u={u_cmd:+.2f} | tgt={target_lataccel:+.2f} | cur={current_lataccel:+.2f} | err={error:+.2f} | gain={b:.2f}")

        self.prev_u = u_cmd
        self._log = {'theta_a': a, 'theta_b': b, 'theta_c': c, 'theta_d': d}
        return u_cmd
