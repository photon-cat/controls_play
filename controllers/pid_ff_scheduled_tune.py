from . import BaseController
import numpy as np

class Controller(BaseController):
    """ PID controller with preview feed-forward and gain scheduled (p i d terms and target control point) from 0-40m/s """
    def __init__(self):
        # gains at [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40] m/s
        self.v_points = np.arange(0, 41, 4)
        self.p_points = np.array([0.2207, 0.2040, 0.1855, 0.1696, 0.1601, 0.1583, 0.1590, 0.1596, 0.1610, 0.1651, 0.1720])
        self.i_points = np.array([0.1366, 0.1193, 0.1083, 0.1068, 0.1122, 0.1217, 0.1295, 0.1329, 0.1273, 0.1132, 0.0928])
        self.d_points = np.array([-0.0887, -0.0691, -0.0546, -0.0471, -0.0433, -0.0408, -0.0379, -0.0357, -0.0338, -0.0316, -0.0286])

        self.k_ff_points = np.array([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000])
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.preview_points = np.array([0.9396, 1.5628, 1.9278, 1.9776, 1.8278, 1.6836, 1.5812, 1.5230, 1.4707, 1.5023, 1.6097])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # linear interp for p i d ff & preview points 
        v_ego = state.v_ego
        kp = np.interp(v_ego, self.v_points, self.p_points)
        ki = np.interp(v_ego, self.v_points, self.i_points)
        kd = np.interp(v_ego, self.v_points, self.d_points)
        kf = np.interp(v_ego, self.v_points, self.k_ff_points)
        preview_steps = np.interp(v_ego, self.v_points, self.preview_points)

        #fractional preview target for lataccel 
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

        # feedforward 
        u_ff = kf * a_ref
        # pid feedback loop 
        error = a_ref - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        u_fb = ( kp * error + ki * self.error_integral + kd * error_diff )
        return u_ff + u_fb
