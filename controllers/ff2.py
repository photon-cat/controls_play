from . import BaseController
import numpy as np

class Controller(BaseController):
    def __init__(self):
        # Feedback (PID) - reduced gains since feedforward does heavy lifting
        self.p = 0.2
        self.i = 0.1
        self.d = -0.05
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward
        self.ff_gain = 0.5          # how much to trust the plan
        self.lookahead_steps = 4     # how far ahead to look (tune to match lag)
        
        # Response lag model (first-order)
        self.lag_alpha = 1.0        # models how fast system actually responds
        self.predicted_lataccel = 0  # internal estimate of where we're heading

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get future target to compensate for response lag
        if hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > self.lookahead_steps:
            future_target = future_plan.lataccel[self.lookahead_steps]
        else:
            future_target = target_lataccel
        
        # Update internal prediction of where lataccel is heading
        # (simple lag model - actual response trails our commands)
        self.predicted_lataccel += self.lag_alpha * (current_lataccel - self.predicted_lataccel)
        
        # Feedforward: command based on where we need to BE, not where we ARE
        ff_command = self.ff_gain * future_target
        
        # Feedback: correct for errors using predicted response
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -5, 5)  # anti-windup
        error_diff = error - self.prev_error
        self.prev_error = error
        
        fb_command = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        return ff_command + fb_command