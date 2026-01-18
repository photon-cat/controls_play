from . import BaseController
import numpy as np 

class Controller(BaseController):
    #a simple pid controller
    def __init__(self,):
        self.gain = 1
        self.plan = np.array[0,0,0,0,0,0,0,0,0]



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