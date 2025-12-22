"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll (not used)

added feedforwaerd prediction
added gain scheduling
added output smoothing (u_alpha) with scheduling

NEW BEST FOUND! Cost: 96.6186. Saving to controller...
Iter 069* | Cost:    96.62 | P:0.023 I:0.135 D:-0.016 FF:0.088 Prev:1.13

NEW BEST FOUND! Cost: 96.4610. Saving to controller...
Iter 098* | Cost:    96.46 | P:0.048 I:0.142 D:-0.040 FF:0.126 Prev:0.70
NEW BEST FOUND! Cost: 94.6039. Saving to controller...
Iter 103* | Cost:    94.60 | P:0.121 I:0.107 D:-0.018 FF:0.092 Prev:0.54
        self.v_points = np.arange(0, 41, 4)
        self.p_points = np.array([0.1968, 0.1970, 0.1933, 0.1885, 0.1770, 0.1591, 0.1525, 0.1576, 0.1691, 0.1657, 0.1676])
        self.i_points = np.array([0.1416, 0.1270, 0.1122, 0.1126, 0.1324, 0.1024, 0.1044, 0.1415, 0.1105, 0.1399, 0.0943])
        self.d_points = np.array([-0.0842, -0.0729, -0.0655, -0.0620, -0.0560, -0.0530, -0.0349, -0.0350, -0.0322, -0.0399, -0.0290])

        self.k_ff_points = np.array([0.1011, 0.1001, 0.1032, 0.0967, 0.0927, 0.1052, 0.0921, 0.1338, 0.1166, 0.1107, 0.1101])
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.preview_points = np.array([0.9406, 1.5230, 1.7908, 1.7987, 1.7609, 1.7248, 1.6955, 1.6479, 1.5753, 1.5829, 1.5737])
"""

from . import BaseController
import numpy as np

class Controller(BaseController):
    """ PID controller with preview feed-forward for lateral acceleration tracking """
    def __init__(self):
        # Gains at [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40] m/s
        self.v_points = np.arange(0, 41, 4)
        self.p_points = np.array([0.1968, 0.1970, 0.1933, 0.1885, 0.1770, 0.1591, 0.1525, 0.1576, 0.1691, 0.1657, 0.1676])
        self.i_points = np.array([0.1416, 0.1270, 0.1122, 0.1126, 0.1324, 0.1024, 0.1044, 0.1415, 0.1105, 0.1399, 0.0943])
        self.d_points = np.array([-0.0842, -0.0729, -0.0655, -0.0620, -0.0560, -0.0530, -0.0349, -0.0350, -0.0322, -0.0399, -0.0290])

        self.k_ff_points = np.array([0.1011, 0.1001, 0.1032, 0.0967, 0.0927, 0.1052, 0.0921, 0.1338, 0.1166, 0.1107, 0.1101])
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.preview_points = np.array([0.9406, 1.5230, 1.7908, 1.7987, 1.7609, 1.7248, 1.6955, 1.6479, 1.5753, 1.5829, 1.5737])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # --- scheduled gains ---
        v_ego = state.v_ego
        kp = np.interp(v_ego, self.v_points, self.p_points)
        ki = np.interp(v_ego, self.v_points, self.i_points)
        kd = np.interp(v_ego, self.v_points, self.d_points)
        kf = np.interp(v_ego, self.v_points, self.k_ff_points)
        preview_steps = np.interp(v_ego, self.v_points, self.preview_points)

        #fractional preview target
        if future_plan is not None and hasattr(future_plan, "lataccel"):
            idx_low = int(np.floor(preview_steps))
            idx_high = int(np.ceil(preview_steps))
            
            if len(future_plan.lataccel) > idx_high:
                frac = preview_steps - idx_low
                a_low = future_plan.lataccel[idx_low]
                a_high = future_plan.lataccel[idx_high]
                a_ref = a_low + frac * (a_high - a_low)
            elif len(future_plan.lataccel) > idx_low:
                a_ref = future_plan.lataccel[idx_low]
            else:
                a_ref = target_lataccel
        else:
            a_ref = target_lataccel

        # --- feed-forward ---
        u_ff = kf * a_ref
        # --- feedback ---
        error = a_ref - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        u_fb = ( kp * error + ki * self.error_integral + kd * error_diff )
        return u_ff + u_fb
