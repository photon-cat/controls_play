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
"""

from . import BaseController
import numpy as np

class Controller(BaseController):
    """ PID controller with preview feed-forward for lateral acceleration tracking """
    def __init__(self):
        # Gains at [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40] m/s
        self.v_points = np.arange(0, 41, 4)
        self.p_points = np.array([0.195, 0.2451, 0.1874, 0.1625, 0.0559, 0.2518, 0.1649, 0.0774, 0.2558, 0.0873, 0.1720])
        self.i_points = np.array([0.100, 0.1310, 0.0527, 0.1082, 0.1175, 0.1284, 0.1016, 0.1866, 0.1233, 0.1340, 0.0928])
        self.d_points = np.array([-0.053, -0.0373, -0.0361, -0.0716, -0.0056, -0.0814, -0.0265, -0.0046, -0.0724, -0.0161, -0.0286])
        #feedforward gain
        #self.k_ff_points = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        #self. ff with 0
        self.k_ff_points = np.array([0.0, 0.1290, 0.1415, 0.2035, 0.0943, 0.0569, 0.0930, 0.0491, 0.0781, 0.1442, 0.1631])
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.preview_points = np.array([1.0, 1.9310, 2.9486, 2.9057, 0.7484, 1.3000, 1.7293, 2.7336, 0.0931, 1.5230, 1.6097])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # --- scheduled gains ---
        v_ego = state.v_ego
        kp = np.interp(v_ego, self.v_points, self.p_points)
        ki = np.interp(v_ego, self.v_points, self.i_points)
        kd = np.interp(v_ego, self.v_points, self.d_points)
        kf = np.interp(v_ego, self.v_points, self.k_ff_points)
        preview_steps = np.interp(v_ego, self.v_points, self.preview_points)

        # --- preview target (fractional) ---
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
