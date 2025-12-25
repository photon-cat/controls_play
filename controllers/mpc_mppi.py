# controllers/mpc.py
from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    Sampling-based MPC using Cross-Entropy Method (CEM)
    - Rolls out candidate steering sequences through a local linear model
    - No need to call the neural physics directly (too expensive)
    - Uses a learned local approximation for fast rollouts
    """
    def __init__(self):
        # MPC horizon & sampling params
        self.horizon = 10          # ~1 second lookahead
        self.n_samples = 64        # candidate trajectories
        self.n_elite = 8           # top trajectories to fit next distribution
        self.n_iterations = 3      # CEM refinement iterations
        
        # Local dynamics approximation: lataccel ≈ k_steer * steer + k_roll * roll
        # Tuned for typical highway speeds
        self.k_steer = 0.8         # steering effectiveness
        self.tau = 0.3             # first-order lag time constant
        
        # Cost weights (match eval)
        self.w_track = 50.0        # LAT_ACCEL_COST_MULTIPLIER
        self.w_jerk = 1.0
        self.w_effort = 0.01       # small penalty on control effort
        
        # Warm start: previous solution shifted
        self.prev_u = None
        self.dt = 0.1
        
    def predict_lataccel(self, current_lataccel, steer_seq, roll_seq, v_ego):
        """Simple first-order dynamics model for fast rollouts"""
        pred = np.zeros(len(steer_seq) + 1)
        pred[0] = current_lataccel
        
        # Steering effectiveness scales with velocity squared (like real physics)
        k = self.k_steer * (v_ego / 25.0) ** 0.5
        alpha = self.dt / self.tau
        
        for i, (u, roll) in enumerate(zip(steer_seq, roll_seq)):
            target = k * u + roll
            pred[i+1] = pred[i] + alpha * (target - pred[i])
            # Respect MAX_ACC_DELTA constraint
            pred[i+1] = np.clip(pred[i+1], pred[i] - 0.5, pred[i] + 0.5)
        
        return pred[1:]
    
    def compute_cost(self, pred_lataccel, target_seq, steer_seq):
        """Match the challenge cost function"""
        tracking_cost = np.sum((pred_lataccel - target_seq) ** 2)
        jerk_cost = np.sum(np.diff(pred_lataccel) ** 2) / (self.dt ** 2)
        effort_cost = np.sum(steer_seq ** 2)
        return self.w_track * tracking_cost + self.w_jerk * jerk_cost + self.w_effort * effort_cost
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel
        
        # Build target and roll sequences for horizon
        if future_plan and len(future_plan.lataccel) >= self.horizon:
            target_seq = np.array(future_plan.lataccel[:self.horizon])
            roll_seq = np.array(future_plan.roll_lataccel[:self.horizon])
        else:
            target_seq = np.full(self.horizon, target_lataccel)
            roll_seq = np.full(self.horizon, roll_lataccel)
        
        # Initialize mean/std for sampling (warm start from previous solution)
        if self.prev_u is not None and len(self.prev_u) == self.horizon:
            mean = np.roll(self.prev_u, -1)
            mean[-1] = mean[-2]  # extend last value
        else:
            mean = np.zeros(self.horizon)
        std = np.full(self.horizon, 0.3)
        
        # CEM optimization loop
        for _ in range(self.n_iterations):
            # Sample candidate sequences
            samples = np.random.normal(mean, std, (self.n_samples, self.horizon))
            samples = np.clip(samples, -2, 2)  # STEER_RANGE
            
            # Evaluate costs
            costs = np.zeros(self.n_samples)
            for j, u_seq in enumerate(samples):
                pred = self.predict_lataccel(current_lataccel, u_seq, roll_seq, v_ego)
                costs[j] = self.compute_cost(pred, target_seq, u_seq)
            
            # Select elite samples and refit distribution
            elite_idx = np.argsort(costs)[:self.n_elite]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0)
            std = np.std(elite, axis=0) + 1e-3  # prevent collapse
        
        # Store for warm start
        self.prev_u = mean.copy()
        
        return mean[0]  # apply first action