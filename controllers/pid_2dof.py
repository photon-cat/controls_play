from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    2-DOF PID Controller for lateral acceleration tracking.

    Two Degrees of Freedom PID separates:
    - Setpoint tracking (feedforward path with setpoint weighting)
    - Disturbance rejection (feedback path)

    This provides better tracking performance and reduces overshoot
    compared to classical 1-DOF PID.
    """

    def __init__(self):
        # PID gains (tuned for tracking + smoothness)
        self.kp = 0.30      # Proportional gain
        self.ki = 0.08      # Integral gain
        self.kd = 0.10      # Derivative gain

        # Setpoint weighting (0 to 1)
        # b = 0: pure feedback, no setpoint in proportional term (reduces overshoot)
        # b = 1: classical PID
        self.b = 0.7        # Setpoint weight for proportional term
        self.c = 0.6        # Setpoint weight for derivative term

        # State variables
        self.error_integral = 0.0
        self.prev_output = 0.0
        self.prev_target = 0.0

        # Anti-windup
        self.integral_limit = 15.0

        # Filtering and smoothing
        self.alpha_filter = 0.3  # Low-pass filter for derivative
        self.filtered_derivative = 0.0

        # Feedforward from future trajectory
        self.kff = 0.15  # Feedforward gain

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        2-DOF PID control update.

        Control law:
        u(t) = Kp*(b*r(t) - y(t)) + Ki*∫e(t)dt + Kd*(c*dr(t)/dt - dy(t)/dt)

        where:
        - r(t) = setpoint (target_lataccel)
        - y(t) = process variable (current_lataccel)
        - e(t) = r(t) - y(t) = tracking error
        """

        # Tracking error (for integral term)
        error = target_lataccel - current_lataccel

        # Integral term (uses full error)
        self.error_integral += error

        # Anti-windup: clamp integral
        self.error_integral = np.clip(
            self.error_integral,
            -self.integral_limit,
            self.integral_limit
        )

        # Proportional term (with setpoint weighting)
        # P = Kp * (b*setpoint - measurement)
        proportional_term = self.kp * (self.b * target_lataccel - current_lataccel)

        # Integral term
        integral_term = self.ki * self.error_integral

        # Derivative term (with setpoint weighting on rate of change)
        # D = Kd * (c*d(setpoint)/dt - d(measurement)/dt)
        target_derivative = target_lataccel - self.prev_target
        output_derivative = current_lataccel - self.prev_output

        # Derivative with setpoint weighting
        raw_derivative = self.c * target_derivative - output_derivative

        # Low-pass filter on derivative to reduce noise
        self.filtered_derivative = (
            self.alpha_filter * raw_derivative +
            (1 - self.alpha_filter) * self.filtered_derivative
        )

        derivative_term = self.kd * self.filtered_derivative

        # 2-DOF PID output
        pid_output = proportional_term + integral_term + derivative_term

        # Feedforward: use near-term future to anticipate trajectory
        feedforward = 0.0
        if len(future_plan.lataccel) >= 3:
            # Look ahead 2-3 steps
            future_avg = np.mean(future_plan.lataccel[:3])
            feedforward = self.kff * future_avg

        # Combined control
        control_output = pid_output + feedforward

        # Saturation limits
        control_output = np.clip(control_output, -2.0, 2.0)

        # Anti-windup: back-calculation
        # If saturated, reduce integral windup
        if control_output >= 2.0 or control_output <= -2.0:
            # Unwind integral slightly
            self.error_integral *= 0.95

        # Update state for next iteration
        self.prev_output = current_lataccel
        self.prev_target = target_lataccel

        return control_output
