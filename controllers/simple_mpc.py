from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    Simplified predictive controller using PID + feedforward.

    Since accurately modeling TinyPhysicsModel is difficult, we use:
    1. PID for feedback control
    2. Feedforward based on future target trajectory
    3. Rate limiting for smoothness
    """

    def __init__(self):
        # PID gains (similar to baseline but tuned)
        self.kp = 0.3
        self.ki = 0.08
        self.kd = 0.0

        # Feedforward gain (use future information)
        self.kff = 0.15

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0

        # Smoothing
        self.max_steer_rate = 0.5  # Max change per timestep

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Predictive control using PID + feedforward from future plan.
        """
        # Current error
        error = target_lataccel - current_lataccel

        # PID terms
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10, 10)  # Anti-windup

        error_derivative = error - self.prev_error
        self.prev_error = error

        # Feedback control
        feedback = self.kp * error + self.ki * self.error_integral + self.kd * error_derivative

        # Feedforward: use near-term future targets
        # Look ahead 3-5 steps to anticipate needed steering
        lookahead = min(5, len(future_plan.lataccel))
        if lookahead > 0:
            # Average of next few targets
            future_avg = np.mean(future_plan.lataccel[:lookahead])
            # Feedforward based on where we're going
            feedforward = self.kff * future_avg
        else:
            feedforward = 0.0

        # Combined control
        steer_command = feedback + feedforward

        # Rate limiting for smoothness (reduces jerk cost)
        max_change = self.max_steer_rate
        steer_command = np.clip(
            steer_command,
            self.prev_steer - max_change,
            self.prev_steer + max_change
        )

        # Saturation limits
        steer_command = np.clip(steer_command, -2.0, 2.0)

        self.prev_steer = steer_command

        return steer_command
