"""
Feedforward controller with ADAPTIVE lookahead based on measured ML model lag.

Key findings from analysis:
1. ML model lag scales with command magnitude:
   - lag_50 ≈ 3 + 2.5 * |steer|  (steps)
   - Small steer (0.2): ~3 steps (0.3s)
   - Large steer (1.5): ~6.5 steps (0.65s)
2. Larger commands cause more tire slip → longer response time
3. steer = (target - roll) / gain
"""
from . import BaseController
import numpy as np

# Gain LUT from labeled data fitting
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

# Adaptive lag model: lag = BASE_LAG + LAG_SLOPE * |steer|
BASE_LAG_STEPS = 3      # Minimum lag for small commands
LAG_SLOPE = 2.5         # Additional lag per unit steer magnitude
MIN_LOOKAHEAD = 3       # Minimum lookahead steps
MAX_LOOKAHEAD = 10      # Maximum lookahead steps


class Controller(BaseController):
    def __init__(self):
        # Adaptive lookahead parameters
        self.base_lag = BASE_LAG_STEPS
        self.lag_slope = LAG_SLOPE
        self.min_lookahead = MIN_LOOKAHEAD
        self.max_lookahead = MAX_LOOKAHEAD

        # Track previous steer for smoothing
        self.prev_steer = 0.0

        # Small PID for error correction
        self.kp = 0.0
        self.ki = 0.0
        self.kd = -0.00
        self.pid_weight = 0.3

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Logging
        self._log = {}

    def get_gain(self, v_ego):
        """Interpolate gain from LUT."""
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def compute_lookahead(self, expected_steer):
        """Compute adaptive lookahead based on expected steer magnitude.

        Larger steer commands have more tire slip -> longer lag.
        lag = 3 + 2.5 * |steer| steps
        """
        lag = self.base_lag + self.lag_slope * abs(expected_steer)
        return int(np.clip(lag, self.min_lookahead, self.max_lookahead))

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel
        gain = self.get_gain(v_ego)

        # ---- Estimate expected steer to compute adaptive lookahead ----
        # First estimate: what steer would we need for current target?
        current_turning = target_lataccel - roll_lataccel
        estimated_steer = current_turning / gain

        # Use smoothed estimate (blend with previous)
        estimated_steer = 0.7 * estimated_steer + 0.3 * self.prev_steer

        # Compute adaptive lookahead based on expected maneuver intensity
        lookahead = self.compute_lookahead(estimated_steer)

        # ---- Lookahead feedforward ----
        # Use future target/roll to compensate for model lag
        if (future_plan.lataccel and future_plan.roll_lataccel and
            len(future_plan.lataccel) > lookahead and
            len(future_plan.roll_lataccel) > lookahead):
            # Look ahead by adaptive lag amount
            future_target = future_plan.lataccel[lookahead - 1]
            future_roll = future_plan.roll_lataccel[lookahead - 1]
            future_v = future_plan.v_ego[lookahead - 1] if len(future_plan.v_ego) > lookahead else v_ego
            future_gain = self.get_gain(future_v)
        else:
            # Fallback to current values
            future_target = target_lataccel
            future_roll = roll_lataccel
            future_gain = gain

        # Feedforward: steer for what we want to happen after lag
        turning_lataccel = future_target - future_roll
        u_ff = turning_lataccel / future_gain

        # ---- PID for current error correction ----
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10, 10)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = (self.kp * error + self.ki * self.error_integral + self.kd * error_diff)
        u_pid = u_pid / gain  # Scale to steer units

        # ---- Combine ----
        u_cmd = u_ff + self.pid_weight * u_pid
        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        # Update previous steer for next iteration
        self.prev_steer = u_cmd

        # Log
        self._log = {
            'la_gain': gain,
            'la_lookahead': lookahead,
            'la_estimated_steer': estimated_steer,
            'la_future_target': future_target,
            'la_future_roll': future_roll,
            'la_u_ff': u_ff,
            'la_u_pid': u_pid,
        }

        return u_cmd
