from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    Auto-tuned PID controller using Ziegler-Nichols relay method.

    Gains derived from system identification on challenging scenario:
    - Ultimate Period (Pu): 1.160 s
    - Ultimate Gain (Ku): 0.213
    - Conservative detuning applied (0.7x)
    """

    def __init__(self):
        # Auto-tuned gains (conservative Ziegler-Nichols)
        self.kp = 0.0893
        self.ki = 0.1540
        self.kd = 0.0130

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0

        # Anti-windup
        self.integral_limit = 10.0

        # Feedforward (increased to better anticipate)
        self.kff = 0.18

        # Adaptive rate limiting
        self.base_steer_rate = 0.5
        self.max_steer_rate = 0.5

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Auto-tuned PID control update.
        """
        # Tracking error
        error = target_lataccel - current_lataccel

        # Integral with anti-windup
        self.error_integral += error
        self.error_integral = np.clip(
            self.error_integral,
            -self.integral_limit,
            self.integral_limit
        )

        # Derivative
        error_derivative = error - self.prev_error
        self.prev_error = error

        # PID control
        pid_output = (
            self.kp * error +
            self.ki * self.error_integral +
            self.kd * error_derivative
        )

        # Feedforward from future plan
        feedforward = 0.0
        if len(future_plan.lataccel) >= 4:
            weights = np.array([0.5, 0.3, 0.15, 0.05])
            future_targets = np.array(future_plan.lataccel[:4])
            future_avg = np.sum(weights * future_targets)
            feedforward = self.kff * future_avg

        # Combined control
        steer_command = pid_output + feedforward

        # Adaptive rate limiting: relax when error is large, tighten when close
        error_magnitude = abs(error)
        if error_magnitude > 1.0:
            # Large error: allow faster response
            adaptive_rate = self.base_steer_rate * 1.5
        elif error_magnitude > 0.5:
            # Medium error: normal rate
            adaptive_rate = self.base_steer_rate
        else:
            # Small error: reduce rate to prevent overshoot
            adaptive_rate = self.base_steer_rate * 0.6

        steer_command = np.clip(
            steer_command,
            self.prev_steer - adaptive_rate,
            self.prev_steer + adaptive_rate
        )

        # Saturation
        steer_command = np.clip(steer_command, -2.0, 2.0)

        # Anti-windup: reduce integral if saturated
        if abs(steer_command) >= 2.0:
            self.error_integral *= 0.9

        # Update state
        self.prev_steer = steer_command

        return steer_command
