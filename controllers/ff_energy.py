"""
Energy-budget feedforward controller.

Strategy (as user described):
1. Compute minimum steer needed to hit target at t+10
2. Budget = 1.1 * that minimum (10% headroom)
3. Within budget, track t+1, t+2, t+3... as closely as possible
4. This reduces jerk by not overreacting to short-term deviations
"""
from . import BaseController
import numpy as np

# Gain LUT
GAIN_LUT = {
    0.5: 1.32, 1.5: 0.96, 2.5: 1.09, 3.5: 1.27, 4.5: 1.53,
    5.5: 1.67, 6.5: 1.78, 7.5: 1.88, 8.5: 1.72, 9.5: 1.33,
    10.5: 1.29, 11.5: 1.48, 12.5: 1.67, 13.5: 1.79, 14.5: 1.40,
    15.5: 1.06, 16.5: 1.16, 17.5: 1.60, 18.5: 1.69, 19.5: 1.48,
    20.5: 1.00, 21.5: 1.18, 22.5: 1.35, 23.5: 1.30, 24.5: 1.12,
    25.5: 1.59, 26.5: 1.60, 27.5: 1.70, 28.5: 1.61, 29.5: 1.57,
    30.5: 1.59, 31.5: 1.15, 32.5: 1.65, 33.5: 2.08, 34.5: 1.73,
    35.5: 1.95, 36.5: 1.35, 37.5: 2.45, 38.5: 1.69, 39.5: 2.43,
    40.5: 2.57,
}

GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])


class Controller(BaseController):
    def __init__(self):
        # Energy budget
        self.horizon = 5            # Look ahead for budget
        self.budget_mult = 1.1      # Allow 10% more than minimum needed
        self.min_budget = 0.15      # Minimum budget for small targets

        # Adaptive lag: lag = base + slope * |steer|
        self.lag_base = 4           # Minimum lag
        self.lag_slope = 3.0        # Additional lag per unit steer
        self.min_lookahead = 4
        self.max_lookahead = 10

        # State tracking
        self.prev_steer = 0.0
        self.steer_history = []     # For trajectory smoothing

        # Logging
        self._log = {}

    def compute_lookahead(self, estimated_steer):
        """Adaptive lookahead based on expected steer magnitude."""
        lag = self.lag_base + self.lag_slope * abs(estimated_steer)
        return int(np.clip(lag, self.min_lookahead, self.max_lookahead))

    def get_gain(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll = state.roll_lataccel
        gain = self.get_gain(v_ego)

        # ---- Step 1: What steer do we need for t+horizon (budget reference)? ----
        far_idx = self.horizon - 1
        if (future_plan.lataccel and len(future_plan.lataccel) > far_idx and
            future_plan.roll_lataccel and len(future_plan.roll_lataccel) > far_idx):
            far_target = future_plan.lataccel[far_idx]
            far_roll = future_plan.roll_lataccel[far_idx]
            far_v = future_plan.v_ego[far_idx] if len(future_plan.v_ego) > far_idx else v_ego
            u_far = (far_target - far_roll) / self.get_gain(far_v)
        else:
            u_far = (target_lataccel - roll) / gain

        # ---- Step 2: Set energy budget based on far target ----
        budget = max(self.min_budget, abs(u_far) * self.budget_mult)

        # ---- Step 3: Estimate steer magnitude to compute adaptive lookahead ----
        # Use current target as estimate of what we'll need
        u_estimate = (target_lataccel - roll) / gain
        lookahead = self.compute_lookahead(u_estimate)

        # ---- Step 4: Get target at adaptive lookahead point ----
        la_idx = lookahead - 1
        if (future_plan.lataccel and len(future_plan.lataccel) > la_idx and
            future_plan.roll_lataccel and len(future_plan.roll_lataccel) > la_idx):
            la_target = future_plan.lataccel[la_idx]
            la_roll = future_plan.roll_lataccel[la_idx]
            la_v = future_plan.v_ego[la_idx] if len(future_plan.v_ego) > la_idx else v_ego
            u_desired = (la_target - la_roll) / self.get_gain(la_v)
        else:
            u_desired = u_estimate

        # ---- Step 5: Constrain to energy budget ----
        if abs(u_desired) > budget:
            u_budget = np.sign(u_desired) * budget
        else:
            u_budget = u_desired

        # No smoothing - proper design should produce smooth output
        u_cmd = u_budget

        # Clip to valid range
        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        # Update state
        self.prev_steer = u_cmd

        # Log
        self._log = {
            'en_u_far': u_far,
            'en_budget': budget,
            'en_lookahead': lookahead,
            'en_u_desired': u_desired,
            'en_u_cmd': u_cmd,
        }

        return u_cmd
