"""
Hybrid Feedforward + PID controller.
Uses LUT for feedforward gain and roll offset compensation.
"""
from . import BaseController
import numpy as np

# Gain LUT from labeled data fitting (v_mid in m/s)
# Outliers (R² < 0.5) interpolated from neighbors
GAIN_LUT = {
    0.5: 1.32,
    1.5: 0.96,
    2.5: 1.09,
    3.5: 1.27,
    4.5: 1.53,
    5.5: 1.67,
    6.5: 1.78,   # interpolated (was 3.4, R²=0.04)
    7.5: 1.88,
    8.5: 1.72,
    9.5: 1.33,
    10.5: 1.29,
    11.5: 1.48,  # interpolated (was 5.2, outlier)
    12.5: 1.67,
    13.5: 1.79,
    14.5: 1.40,
    15.5: 1.06,
    16.5: 1.16,
    17.5: 1.60,
    18.5: 1.69,
    19.5: 1.48,
    20.5: 1.00,
    21.5: 1.18,
    22.5: 1.35,
    23.5: 1.30,
    24.5: 1.12,
    25.5: 1.59,
    26.5: 1.60,
    27.5: 1.70,  # interpolated (R²=-0.01)
    28.5: 1.61,
    29.5: 1.57,
    30.5: 1.59,
    31.5: 1.15,
    32.5: 1.65,
    33.5: 1.90,
    34.5: 1.73,
    35.5: 1.95,
    36.5: 1.35,
    37.5: 2.45,
    38.5: 1.69,
    39.5: 2.43,
    40.5: 2.57,
}

# Convert to sorted arrays for interpolation
GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])


class Controller(BaseController):
    def __init__(self):
        # PID gains (for error correction)
        self.kp = 0.18
        self.ki = 0.1
        self.kd = -0.053

        # Feedforward weight (0 = pure PID, 1 = pure FF)
        self.ff_weight = 1.0
        self.pid_weight = 0.0

        # Scale PID by gain LUT (converts lataccel error to steer units)
        self.scale_pid_by_gain = True

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Logging
        self._log = {}

    def get_gain(self, v_ego):
        """Interpolate gain from LUT."""
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # ---- Feedforward ----
        # Remove roll contribution - we only steer for turning
        turning_lataccel = target_lataccel - roll_lataccel

        # Get gain from LUT
        gain = self.get_gain(v_ego)

        # Feedforward steering command
        u_ff = turning_lataccel / gain

        # ---- PID (error correction) ----
        error = target_lataccel - current_lataccel
        self.error_integral += error
        # Anti-windup
        self.error_integral = np.clip(self.error_integral, -10, 10)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = (
            self.kp * error +
            self.ki * self.error_integral +
            self.kd * error_diff
        )

        # Scale PID by gain to convert from lataccel to steer units
        if self.scale_pid_by_gain:
            u_pid = u_pid / gain

        # ---- Combine FF + PID ----
        # Scale FF contribution, always add full PID
        u_cmd = (u_ff * self.ff_weight) + (u_pid * self.pid_weight)

        # Clip to valid range
        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        # Log
        self._log = {
            'ff_u_ff': u_ff,
            'ff_u_pid': u_pid,
            'ff_gain': gain,
            'ff_turning_lataccel': turning_lataccel,
            'ff_roll_lataccel': roll_lataccel,
            'ff_error': error,
        }

        return u_cmd
