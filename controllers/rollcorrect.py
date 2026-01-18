from . import BaseController
import numpy as np 

class Controller(BaseController):
    #a simple pid controller
    def __init__(self,):
        self.p = 0.2 
        self.i = 0.1
        self.d = -0.05
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan): 
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        #print state
        print("State: ", state)
        #print future_plan
        print("Plan: ",future_plan)
        return (self.p * error + self.i * self.error_integral + self.d * error_diff)


