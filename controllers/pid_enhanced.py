"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll
"""

"""
first adjust to control yawratedeg vs target_lataccel
gain scheduling = adjust the gain of the controller based on the state of the vehicle
lateralaccelgain - adjust if controller overshoots target lataccel when closing on target lataccel if controller ever gets off
lateralTrackingGain- adjsut if controller is sluggish to close on small errors in lataccel
"""

from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    PD-style lateral controller for TinyPhysics
    (Garmin-like: no integral, rate damping via lataccel rate)
    """
    def __init__(self):
        # Servo gain (P)
        self.Kp = 0.2

        # Damping gain (acts like yaw-rate damper)
        self.Kd = 0.1

        # Normalized authority limit
        self.max_u = 1.0

        self.prev_lataccel = 0.0
        self.prev_step = None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Estimate dt (TinyPhysics steps are uniform, but be safe)
        if self.prev_step is None:
            lataccel_rate = 0.0
        else:
            lataccel_rate = current_lataccel - self.prev_lataccel

        self.prev_lataccel = current_lataccel
        self.prev_step = True

        # Primary error
        error = target_lataccel - current_lataccel

        # Raw control (P + damping)
        u_raw = self.Kp * error - self.Kd * lataccel_rate

        # Torque clamp
        u_norm = np.clip(u_raw, -self.max_u, self.max_u)

        # Map to actuator range [-2, +2]
        steer_cmd = 2.0 * u_norm

        return steer_cmd