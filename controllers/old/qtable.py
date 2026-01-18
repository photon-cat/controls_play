"""
Q-Table controller with discretized states.
- Discretizes state into bins for fast lookup
- Uses FF+PID as base, Q-table learns corrections
- Much faster learning than neural network
"""
from . import BaseController
import numpy as np

STEER_RANGE = [-2.0, 2.0]

# Gain LUT
GAIN_LUT = {
    0.5: 1.32, 1.5: 0.96, 2.5: 1.09, 3.5: 1.27, 4.5: 1.53,
    5.5: 1.67, 6.5: 1.78, 7.5: 1.88, 8.5: 1.72, 9.5: 1.33,
    10.5: 1.29, 11.5: 1.48, 12.5: 1.67, 13.5: 1.79, 14.5: 1.40,
    15.5: 1.06, 16.5: 1.16, 17.5: 1.60, 18.5: 1.69, 19.5: 1.48,
    20.5: 1.00, 21.5: 1.18, 22.5: 1.35, 23.5: 1.30, 24.5: 1.12,
    25.5: 1.59, 26.5: 1.60, 27.5: 1.70, 28.5: 1.61, 29.5: 1.57,
    30.5: 1.59, 31.5: 1.15, 32.5: 1.65, 33.5: 1.90, 34.5: 1.73,
    35.5: 1.95, 36.5: 1.35, 37.5: 2.45, 38.5: 1.69, 39.5: 2.43,
    40.5: 2.57,
}

GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])

# Discretization
ERROR_BINS = np.linspace(-1.0, 1.0, 11)  # 10 bins for error
DERROR_BINS = np.linspace(-0.5, 0.5, 7)  # 6 bins for error rate
STEER_BINS = np.linspace(-1.0, 1.0, 7)  # 6 bins for prev steer

# Correction actions: small adjustments to base policy
CORRECTION_VALUES = np.array([-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])


class Controller(BaseController):
    def __init__(self):
        # Q-table: state (error_bin, derror_bin, steer_bin) -> action corrections
        n_error = len(ERROR_BINS) - 1
        n_derror = len(DERROR_BINS) - 1
        n_steer = len(STEER_BINS) - 1
        n_actions = len(CORRECTION_VALUES)

        # Initialize Q-table with zeros (no correction by default)
        self.q_table = np.zeros((n_error, n_derror, n_steer, n_actions))

        # Q-learning params
        self.alpha = 0.3  # Learning rate
        self.gamma = 0.9  # Discount
        self.epsilon = 0.2  # Exploration

        # Lag model
        self.lag_base = 3.0
        self.lag_scale = 2.0

        # PID gains
        self.kp = 0.195
        self.ki = 0.100
        self.kd = -0.053

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.prev_lataccel = None

        # For Q-learning updates
        self.prev_state_idx = None
        self.prev_action_idx = None

        self._log = {}

    def _get_gain(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def _compute_lag(self, steer_mag):
        lag = int(round(self.lag_base + self.lag_scale * steer_mag))
        return min(lag, 8)

    def _discretize_state(self, error, derror, prev_steer):
        """Convert continuous state to discrete indices."""
        error_idx = np.clip(np.digitize(error, ERROR_BINS) - 1, 0, len(ERROR_BINS) - 2)
        derror_idx = np.clip(np.digitize(derror, DERROR_BINS) - 1, 0, len(DERROR_BINS) - 2)
        steer_idx = np.clip(np.digitize(prev_steer, STEER_BINS) - 1, 0, len(STEER_BINS) - 2)
        return (error_idx, derror_idx, steer_idx)

    def _compute_reward(self, error, derror):
        """Reward: negative cost."""
        lataccel_cost = error ** 2 * 50
        jerk_cost = (derror / 0.1) ** 2
        return -(lataccel_cost + jerk_cost) / 10.0  # Scale down

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        roll_la = state.roll_lataccel
        v_ego = state.v_ego

        gain = self._get_gain(v_ego)

        # Error and derivative
        error = target_lataccel - current_lataccel
        derror = (current_lataccel - self.prev_lataccel) if self.prev_lataccel is not None else 0.0

        # PID
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = self.kp * error + self.ki * self.error_integral + self.kd * error_diff

        # Feedforward with lag lookahead
        lag = self._compute_lag(abs(self.prev_steer))
        if lag > 0 and future_plan.lataccel and len(future_plan.lataccel) >= lag:
            lookahead_target = future_plan.lataccel[lag - 1]
            lookahead_roll = future_plan.roll_lataccel[lag - 1] if future_plan.roll_lataccel and len(future_plan.roll_lataccel) >= lag else roll_la
        else:
            lookahead_target = target_lataccel
            lookahead_roll = roll_la

        turning = lookahead_target - lookahead_roll
        u_ff = turning / max(gain, 0.5)

        # Base policy
        u_base = u_ff + u_pid

        # Get current state indices
        state_idx = self._discretize_state(error, derror, self.prev_steer)

        # Q-learning update from previous step
        if self.prev_state_idx is not None and self.prev_action_idx is not None:
            reward = self._compute_reward(error, derror)

            # Get max Q for current state
            max_next_q = np.max(self.q_table[state_idx])

            # TD update
            old_q = self.q_table[self.prev_state_idx][self.prev_action_idx]
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
            self.q_table[self.prev_state_idx][self.prev_action_idx] = new_q

        # Select action (correction)
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(len(CORRECTION_VALUES))
        else:
            action_idx = np.argmax(self.q_table[state_idx])

        correction = CORRECTION_VALUES[action_idx]

        # Final command
        u_cmd = u_base + correction
        u_cmd = np.clip(u_cmd, STEER_RANGE[0], STEER_RANGE[1])

        # Store for next update
        self.prev_state_idx = state_idx
        self.prev_action_idx = action_idx
        self.prev_steer = u_cmd
        self.prev_lataccel = current_lataccel

        self._log = {
            'gain': gain,
            'correction': correction,
            'q_max': np.max(self.q_table[state_idx]),
            'error': error,
        }

        return u_cmd
