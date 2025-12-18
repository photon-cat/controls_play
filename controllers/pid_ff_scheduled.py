"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll (not used)

added feedforwaerd prediction
added gain scheduling
added output smoothing (u_alpha) with scheduling
"""

from . import BaseController
import numpy as np

class Controller(BaseController):
    """ PID controller with preview feed-forward for lateral acceleration tracking """
    def __init__(self):
        # Gains at [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40] m/s
        self.v_points = np.arange(0, 41, 4)
        self.p_points = np.array([0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195, 0.195])
        self.i_points = np.array([0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100])
        self.d_points = np.array([-0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053, -0.053])
        #feedforward gain
        self.k_ff_points = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.preview_steps = 3

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # --- preview target ---
        if future_plan is not None and hasattr(future_plan, "lataccel") and len(future_plan.lataccel) > self.preview_steps:
            a_ref = future_plan.lataccel[self.preview_steps]
        else:
            a_ref = target_lataccel
        # --- scheduled gains ---
        v_ego = state.v_ego
        kp = np.interp(v_ego, self.v_points, self.p_points)
        ki = np.interp(v_ego, self.v_points, self.i_points)
        kd = np.interp(v_ego, self.v_points, self.d_points)
        kf = np.interp(v_ego, self.v_points, self.k_ff_points)
        # --- feed-forward ---
        u_ff = kf * a_ref
        # --- feedback ---
        error = a_ref - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        u_fb = ( kp * error + ki * self.error_integral + kd * error_diff )
        return u_ff + u_fb
