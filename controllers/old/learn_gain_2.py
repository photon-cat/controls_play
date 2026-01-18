"""
Adaptive gain controller using online gradient descent.
- Starts from a velocity-based gain LUT
- Updates gain online to reduce lateral acceleration error
- Adds a PID correction on top of feedforward
- Uses a simple rate-dependent lag lookahead to account for command delay
"""
from . import BaseController
import numpy as np

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

CONTROL_START_IDX = 100


class Controller(BaseController):
    def __init__(self):
        self.gain = None
        self.min_gain = 0.4
        self.max_gain = 4.0

        self.base_lr = 0.02
        self.command_rate_scale = 0.8
        self.turning_threshold = 0.05

        self.ff_weight = 1.0

        self.kp = 0.1
        self.ki = 0.1
        self.kd = -0.05
        self.error_integral = 0.0
        self.prev_error = 0.0

        self.prev_u_cmd = 0.0
        self.prev_u_rate = 0.0
        self.step_count = 0

        self._log = {}

    def get_gain_lut(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def _compute_lag_steps(self, v_ego):
        rate_mag = abs(self.prev_u_rate)
        lag = int(np.clip(round(rate_mag * self.command_rate_scale * (1.0 + 0.02 * v_ego)), 0, 3))
        return lag

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_hopw count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        if self.gain is None:
            self.gain = self.get_gain_lut(v_ego)

        lag_steps = self._compute_lag_steps(v_ego)
        effective_target = target_lataccel
        effective_roll = roll_lataccel
        if lag_steps > 0 and future_plan.lataccel and len(future_plan.lataccel) > lag_steps:
            effective_target = future_plan.lataccel[lag_steps - 1]
            if future_plan.roll_lataccel and len(future_plan.roll_lataccel) > lag_steps:
                effective_roll = future_plan.roll_lataccel[lag_steps - 1]

        turning_lataccel = effective_target - effective_roll
        u_ff = turning_lataccel / max(self.gain, 1e-3)

        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = (self.kp * error + self.ki * self.error_integral + self.kd * error_diff)
        u_cmd = np.clip(u_pid + (self.ff_weight * u_ff), -2.0, 2.0)

        turning_now = target_lataccel - roll_lataccel
        if self.step_count >= CONTROL_START_IDX and abs(turning_now) > self.turning_threshold:
            lr = self.base_lr / (1.0 + abs(self.prev_u_rate))
            grad = error * turning_now / max(self.gain, 1e-3)
            self.gain -= lr * grad
            self.gain = float(np.clip(self.gain, self.min_gain, self.max_gain))

        self.prev_u_rate = u_cmd - self.prev_u_cmd
        self.prev_u_cmd = u_cmd

        self._log = {
            'learn_gain': self.gain,
            'learn_lag_steps': lag_steps,
            'learn_lr': self.base_lr,
            'learn_error': error,
        }

        return u_cmd