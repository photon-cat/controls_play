"""
FF roll controller with autotune.
- Runs PID for steps 100-300, learning roll correction
- Switches to FF at step 300
"""
from .. import BaseController
import numpy as np
from collections import defaultdict

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

CONTROL_START_IDX = 100
FF_START_IDX = 300  # Switch from PID to FF at this step


class Controller(BaseController):
    def __init__(self):
        # PID gains
        self.kp = 0.2
        self.ki = 0.1
        self.kd = -0.05

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step_count = 0

        # Learning storage: (roll_lataccel, steer_used) pairs
        self.roll_steer_data = []

        # Learned roll correction gain (roll_lataccel -> steer)
        # Default: steer = -roll_lataccel / gain (theoretical)
        self.roll_gain = None  # Will be learned
        self.learned = False

        # Mode tracking
        self.mode = 'pid'  # 'pid' or 'ff'

        # Logging
        self._log = {}

    def get_gain(self, v_ego):
        """Interpolate gain from LUT."""
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def learn_roll_correction(self):
        """Fit roll_gain from observed (roll_lataccel, steer) pairs."""
        if len(self.roll_steer_data) < 20:
            return

        roll_arr = np.array([d[0] for d in self.roll_steer_data])
        steer_arr = np.array([d[1] for d in self.roll_steer_data])

        # Filter for significant roll
        mask = np.abs(roll_arr) > 0.1
        if mask.sum() < 10:
            return

        roll_fit = roll_arr[mask]
        steer_fit = steer_arr[mask]

        # Fit: steer = roll_gain * roll_lataccel
        # (We expect negative gain since we steer against roll)
        denom = np.sum(roll_fit ** 2)
        if denom > 0.01:
            self.roll_gain = np.sum(roll_fit * steer_fit) / denom

            # R² check
            steer_pred = self.roll_gain * roll_fit
            ss_res = np.sum((steer_fit - steer_pred) ** 2)
            ss_tot = np.sum((steer_fit - np.mean(steer_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            self.learned = True
            print(f"Learned roll_gain: {self.roll_gain:.4f} (R²={r2:.3f})")
            print(f"  Theoretical: {-1/self.get_gain(20):.4f} at 20 m/s")

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel
        gain = self.get_gain(v_ego)

        # Determine mode
        if self.step_count >= FF_START_IDX and not self.learned:
            self.learn_roll_correction()

        if self.step_count >= FF_START_IDX and self.learned:
            self.mode = 'ff'
        else:
            self.mode = 'pid'

        # ---- PID ----
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10, 10)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = self.kp * error + self.ki * self.error_integral + self.kd * error_diff

        # ---- FF roll correction ----
        if self.roll_gain is not None:
            u_ff_roll = self.roll_gain * roll_lataccel
        else:
            # Theoretical default
            u_ff_roll = -roll_lataccel / gain

        # ---- Output based on mode ----
        if self.mode == 'pid':
            u_cmd = u_pid
            # Store data for learning (only during PID control phase)
            if self.step_count >= CONTROL_START_IDX:
                self.roll_steer_data.append((roll_lataccel, u_pid))
        else:
            # FF mode - use learned roll correction
            u_cmd = u_ff_roll

        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        # Log
        self._log = {
            'roll_mode': self.mode,
            'roll_u_pid': u_pid,
            'roll_u_ff': u_ff_roll,
            'roll_lataccel': roll_lataccel,
            'roll_gain_learned': self.roll_gain if self.roll_gain else 0,
            'roll_n_samples': len(self.roll_steer_data),
        }

        return u_cmd
