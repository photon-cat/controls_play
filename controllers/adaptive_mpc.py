"""Preview-enhanced PID-style controller.

The controller keeps the simplicity of a PID loop while adding a short preview
window and a light feedforward term so it reacts earlier on curved segments and
reduces total control cost.
"""

import numpy as np

from controllers import BaseController


class Controller(BaseController):
    def __init__(self):
        # Steering/lataccel relationship
        self.steer_to_latacc = 2.6

        # Feedback gains (PID-style)
        self.kp = 0.34
        self.ki = 0.1
        self.kd = -0.04
        self.integral = 0.0
        self.integral_limit = 50.0
        self.integral_decay = 0.995

        # Limits
        self.steer_limit = 2.0
        self.rate_limit = 0.3
        self.roll_comp_gain = 0.85

        # Preview/target shaping
        self.preview_steps = 8
        self.preview_decay = 0.92

        # State/history
        self.prev_command = 0.0
        self.initialized = False
        self.prev_error = 0.0
        self.prev_lataccel = 0.0
        self.prev_applied_action = 0.0
        self.last_applied_action = 0.0
        self.alpha_command = 0.45

    def observe_applied_action(self, applied_action: float):
        self.prev_applied_action = self.last_applied_action
        self.last_applied_action = float(applied_action)

    def _compute_preview_target(self, target_lataccel: float, future_plan):
        if future_plan is None or len(future_plan.lataccel) == 0:
            return target_lataccel

        lookahead = min(self.preview_steps, len(future_plan.lataccel))
        future_lat = np.array(future_plan.lataccel[:lookahead])
        future_roll = np.array(future_plan.roll_lataccel[:lookahead]) if len(future_plan.roll_lataccel) >= lookahead else np.zeros(lookahead)
        compensated_future = future_lat - self.roll_comp_gain * future_roll
        weights = self.preview_decay ** np.arange(lookahead)
        blended = float((weights * compensated_future).sum() / weights.sum())
        return 0.5 * target_lataccel + 0.5 * blended

    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        roll_comp = self.roll_comp_gain * float(state.roll_lataccel)
        measured = float(current_lataccel) - roll_comp
        target = float(target_lataccel) - roll_comp

        if not self.initialized:
            self.prev_lataccel = measured
            self.initialized = True

        # Preview the target to smooth commands.
        shaped_target = self._compute_preview_target(target, future_plan)

        # Feedback terms.
        error = shaped_target - measured
        d_error = error - self.prev_error

        self.integral = np.clip(
            self.integral * self.integral_decay + error,
            -self.integral_limit,
            self.integral_limit,
        )
        correction = self.kp * error + self.ki * self.integral + self.kd * d_error

        # Feedforward based on current gain estimate.
        feedforward = np.clip(shaped_target / self.steer_to_latacc, -self.steer_limit, self.steer_limit)

        raw_command = feedforward + correction

        # Blend with the previous command for stability.
        blended_command = (1.0 - self.alpha_command) * self.prev_command + self.alpha_command * raw_command

        # Rate limiting relative to the last applied action to keep jerk low.
        ref = self.last_applied_action if self.initialized else self.prev_command
        max_delta = self.rate_limit
        limited_command = np.clip(blended_command, ref - max_delta, ref + max_delta)
        limited_command = np.clip(limited_command, -self.steer_limit, self.steer_limit)

        # Book-keeping for next step.
        self.prev_command = limited_command
        self.prev_lataccel = measured
        self.prev_error = error

        return float(limited_command)
