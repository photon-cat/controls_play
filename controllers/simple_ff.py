"""
Simple feedforward controller with lag lookahead.
- Uses velocity-based gain LUT
- Looks ahead by lag steps for target
- Lag = base + scale * |steer|
"""
from . import BaseController
import numpy as np
import pandas as pd

ACC_G = 9.81
CONTROL_START_IDX = 100
STEER_RANGE = [-2.0, 2.0]

# Gain LUT from data analysis
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


class Controller(BaseController):
    def __init__(self):
        # Lag model: lag steps = base + scale * |steer|
        self.lag_base = 3.0
        self.lag_scale = 2.0

        # State tracking
        self.prev_steer = 0.0
        self.step_count = 0

        self._log = {}

    def _get_gain(self, v_ego):
        """Get gain from velocity LUT."""
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def _compute_lag(self, steer_mag):
        """Compute lag in steps based on steer magnitude."""
        lag = int(round(self.lag_base + self.lag_scale * steer_mag))
        return min(lag, 8)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        roll_la = state.roll_lataccel
        v_ego = state.v_ego

        gain = self._get_gain(v_ego)

        # Compute lag based on previous steer magnitude
        lag = self._compute_lag(abs(self.prev_steer))

        # Look ahead by lag steps for target
        if lag > 0 and future_plan.lataccel and len(future_plan.lataccel) >= lag:
            lookahead_target = future_plan.lataccel[lag - 1]
            lookahead_roll = future_plan.roll_lataccel[lag - 1] if future_plan.roll_lataccel and len(future_plan.roll_lataccel) >= lag else roll_la
        else:
            lookahead_target = target_lataccel
            lookahead_roll = roll_la

        # Feedforward: steer = (target - roll) / gain
        turning = lookahead_target - lookahead_roll
        u_cmd = turning / max(gain, 0.5)

        u_cmd = np.clip(u_cmd, STEER_RANGE[0], STEER_RANGE[1])

        self.prev_steer = u_cmd

        error = target_lataccel - current_lataccel
        self._log = {
            'gain': gain,
            'lag': lag,
            'error': error,
            'lookahead_target': lookahead_target,
        }

        return u_cmd
