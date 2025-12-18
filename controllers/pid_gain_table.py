from . import BaseController
import numpy as np


class Controller(BaseController):
    """
    Minimal PID controller with gain scheduling via tabulated curves.
    Gains are scaled as a function of lateral-accel magnitude using
    10-point curves and linear interpolation.
    """

    def __init__(self):
        # Base gains (copied from simple pid)
        self.kp_base = 0.195
        self.ki_base = 0.100
        self.kd_base = -0.053  # sign matches original controller

        # Reference speed where scheduling reaches the last table entry
        self.speed_ref = 35.0  # m/s (~78 mph)

        # 10-point gain curves over normalized speed in [0, 1]
        xs = np.linspace(0.0, 1.0, 10)
        self.xs = xs
        # Decrease P as speed rises to reduce aggression at high speed
        self.kp_curve = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])
        # Keep I and D unchanged (all ones)
        self.ki_curve = np.ones_like(self.kp_curve)
        self.kd_curve = np.ones_like(self.kp_curve)

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Limits
        self.integral_limit = 8.0
        self.dt = 0.1  # simulator step

    def _schedule_scales(self, state):
        """Interpolate gain scales from the 10-point curves."""
        speed_norm = np.clip(abs(state.v_ego) / self.speed_ref, 0.0, 1.0)
        kp_scale = np.interp(speed_norm, self.xs, self.kp_curve)
        ki_scale = np.interp(speed_norm, self.xs, self.ki_curve)
        kd_scale = np.interp(speed_norm, self.xs, self.kd_curve)
        return kp_scale, ki_scale, kd_scale

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        kp_s, ki_s, kd_s = self._schedule_scales(state)

        error = target_lataccel - current_lataccel

        # Integral with clamp
        self.error_integral = np.clip(
            self.error_integral + error,
            -self.integral_limit,
            self.integral_limit,
        )

        # Derivative (scaled by dt to represent rate)
        d_error = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Scheduled PID
        u = (
            (self.kp_base * kp_s) * error
            + (self.ki_base * ki_s) * self.error_integral
            + (self.kd_base * kd_s) * d_error
        )

        # Output clamp
        u = float(np.clip(u, -2.0, 2.0))

        # Anti-windup: soften integral when saturated
        if abs(u) >= 2.0:
            self.error_integral *= 0.9

        return u

