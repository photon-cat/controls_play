"""
Adaptive controller: FF+PID base with learned correction.
- Base policy: feedforward + PID (known good)
- Online learning: adjusts gain and correction based on observed errors
- Learns what correction to apply given the state
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


class Controller(BaseController):
    def __init__(self):
        # Lag model
        self.lag_base = 3.0
        self.lag_scale = 2.0

        # PID gains
        self.kp = 0.195
        self.ki = 0.100
        self.kd = -0.053

        # Adaptive gain multiplier (learned online)
        self.gain_mult = 1.0
        self.gain_lr = 0.02

        # Adaptive bias correction
        self.bias = 0.0
        self.bias_lr = 0.01

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.prev_lataccel = None

        # History for learning
        self.steer_history = []
        self.error_history = []

        self._log = {}

    def _get_gain(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def _compute_lag(self, steer_mag):
        lag = int(round(self.lag_base + self.lag_scale * steer_mag))
        return min(lag, 8)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        roll_la = state.roll_lataccel
        v_ego = state.v_ego

        base_gain = self._get_gain(v_ego)
        gain = base_gain * self.gain_mult

        # PID on error
        error = target_lataccel - current_lataccel
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

        # Combine with bias correction
        u_cmd = u_ff + u_pid + self.bias

        u_cmd = np.clip(u_cmd, STEER_RANGE[0], STEER_RANGE[1])

        # Online learning from delayed feedback
        self.steer_history.append(u_cmd)
        self.error_history.append(error)
        if len(self.steer_history) > 20:
            self.steer_history.pop(0)
            self.error_history.pop(0)

        # Learn gain multiplier: if we consistently overshoot, reduce gain
        if len(self.steer_history) >= lag + 1 and lag > 0:
            delayed_steer = self.steer_history[-(lag + 1)]
            delayed_error = self.error_history[-(lag + 1)]

            # If we steered positive and ended up with positive error (undershoot), increase gain
            # If we steered positive and ended up with negative error (overshoot), decrease gain
            if abs(delayed_steer) > 0.1:
                correction = -delayed_error * delayed_steer
                self.gain_mult += self.gain_lr * correction
                self.gain_mult = np.clip(self.gain_mult, 0.5, 2.0)

        # Learn bias: if there's systematic error, add bias correction
        if len(self.error_history) >= 5:
            recent_error_mean = np.mean(self.error_history[-5:])
            self.bias += self.bias_lr * recent_error_mean
            self.bias = np.clip(self.bias, -0.5, 0.5)

        self.prev_steer = u_cmd
        self.prev_lataccel = current_lataccel

        self._log = {
            'gain': gain,
            'gain_mult': self.gain_mult,
            'bias': self.bias,
            'u_ff': u_ff,
            'u_pid': u_pid,
            'error': error,
        }

        return u_cmd
