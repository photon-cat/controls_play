"""
Adaptive gain controller using online gradient descent.
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
        self.min_gain = 0.5
        self.max_gain = 4.0
        self.base_lr = 0.02
        self.step_count = 0
        self.last_action = None
        self.prev_action = None
        self.kp = 0.195
        self.ki = 0.1
        self.kd = -0.053
        self.ff_weight = 0.2
        self.error_integral = 0.0
        self.prev_error = 0.0
        self._log = {}

    def get_gain(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def observe_action(self, action, step_idx):
        self.prev_action = self.last_action
        self.last_action = action

    def _update_gain(self, turning_lataccel, steer_cmd, v_ego):
        if steer_cmd is None:
            return
        if abs(steer_cmd) < 0.05:
            return

        error = turning_lataccel - self.gain * steer_cmd

        action_rate = 0.0
        if self.prev_action is not None:
            action_rate = abs(steer_cmd - self.prev_action)

        speed_scale = np.clip(v_ego / 15.0, 0.5, 2.0)
        lr = self.base_lr * speed_scale / (1.0 + 0.01 * self.step_count)
        lr = lr / (1.0 + action_rate)

        self.gain += lr * error * steer_cmd
        self.gain = float(np.clip(self.gain, self.min_gain, self.max_gain))

        self._log['learn_gain_error'] = error
        self._log['learn_gain_lr'] = lr

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        if self.gain is None:
            self.gain = self.get_gain(v_ego)

        if self.step_count >= CONTROL_START_IDX:
            turning_lataccel = current_lataccel - roll_lataccel
            self._update_gain(turning_lataccel, self.last_action, v_ego)

        turning_target = target_lataccel - roll_lataccel
        u_ff = turning_target / self.gain

        error = target_lataccel - current_lataccel
        self.error_integral = np.clip(self.error_integral + error, -10.0, 10.0)
        error_diff = error - self.prev_error
        self.prev_error = error
        u_pid = self.kp * error + self.ki * self.error_integral + self.kd * error_diff

        u_cmd = u_pid + (self.ff_weight * u_ff)
        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        self._log.update({
            'learn_gain': self.gain,
            'learn_gain_turning_target': turning_target,
            'learn_gain_u_ff': u_ff,
            'learn_gain_u_pid': u_pid,
        })

        return u_cmd