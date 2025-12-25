"""
MPC controller using the actual tinyphysics ONNX model as dynamics.

This is the RIGHT way to do MPC - use the real dynamics model for rollouts,
not a hand-tuned approximation.
"""

from . import BaseController
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import namedtuple

# Constants from tinyphysics
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
    """Same tokenizer as tinyphysics"""
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[token]


class Controller(BaseController):
    def __init__(self):
        # Load the ONNX dynamics model
        model_path = Path(__file__).parent.parent / "models" / "tinyphysics.onnx"
        
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        
        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])
        
        self.tokenizer = LataccelTokenizer()
        
        # MPC parameters
        self.horizon = 8            # rollout steps
        self.n_samples = 24         # candidate trajectories  
        self.n_elite = 6            # top trajectories for CEM
        self.n_iterations = 2       # CEM iterations
        
        # Cost weights (match challenge)
        self.w_track = 50.0
        self.w_jerk = 1.0
        
        # Context history (same as simulator maintains)
        self.state_history = []     # List of State tuples
        self.action_history = []    # List of steer commands
        self.lataccel_history = []  # List of lataccel predictions
        
        self.current_lataccel = 0.0
        self.prev_steer = 0.0
    
    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def predict_lataccel(self, sim_states, actions, past_preds, deterministic=True):
        """
        Call ONNX model to predict next lataccel.
        Uses deterministic mode (argmax) for MPC rollouts.
        """
        tokenized_preds = self.tokenizer.encode(past_preds)
        raw_states = [list(x) for x in sim_states]
        states = np.column_stack([actions, raw_states])
        
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_preds, axis=0).astype(np.int64)
        }
        
        res = self.ort_session.run(None, input_data)[0]
        
        if deterministic:
            # Use argmax for deterministic rollouts
            token = np.argmax(res[0, -1])
        else:
            # Sample from distribution
            probs = self.softmax(res / 0.8, axis=-1)
            token = np.random.choice(VOCAB_SIZE, p=probs[0, -1])
        
        return self.tokenizer.decode(token)
    
    def rollout(self, steer_sequence, state, future_plan):
        """
        Roll out a steering sequence using the ONNX dynamics model.
        Returns predicted lataccel trajectory.
        """
        # Copy current history - ensure all arrays have same length
        n = min(len(self.state_history), len(self.action_history), len(self.lataccel_history))
        n = min(n, CONTEXT_LENGTH)
        
        sim_states = list(self.state_history[-n:]) if n > 0 else []
        actions = list(self.action_history[-n:]) if n > 0 else []
        preds = list(self.lataccel_history[-n:]) if n > 0 else []
        
        lat = self.current_lataccel
        lataccel_traj = []
        
        for t, steer in enumerate(steer_sequence):
            # Get future state (assume it follows the plan)
            if future_plan and t < len(future_plan.v_ego):
                next_state = State(
                    roll_lataccel=future_plan.roll_lataccel[t] if t < len(future_plan.roll_lataccel) else state.roll_lataccel,
                    v_ego=future_plan.v_ego[t] if t < len(future_plan.v_ego) else state.v_ego,
                    a_ego=future_plan.a_ego[t] if t < len(future_plan.a_ego) else state.a_ego
                )
            else:
                next_state = state
            
            # Update context - add all three together to keep in sync
            sim_states.append(next_state)
            actions.append(steer)
            preds.append(lat)
            
            # Keep context length - trim all together
            if len(sim_states) > CONTEXT_LENGTH:
                sim_states = sim_states[-CONTEXT_LENGTH:]
                actions = actions[-CONTEXT_LENGTH:]
                preds = preds[-CONTEXT_LENGTH:]
            
            # Predict next lataccel using ONNX model
            if len(sim_states) >= CONTEXT_LENGTH and len(actions) == len(sim_states):
                pred = self.predict_lataccel(sim_states, actions, preds, deterministic=True)
                # Apply rate limit
                pred = np.clip(pred, lat - MAX_ACC_DELTA, lat + MAX_ACC_DELTA)
                lat = pred
            
            lataccel_traj.append(lat)
        
        return np.array(lataccel_traj)
    
    def compute_cost(self, lataccel_traj, target_seq, steer_seq):
        """Compute tracking + jerk cost"""
        n = min(len(lataccel_traj), len(target_seq))
        tracking = np.sum((lataccel_traj[:n] - target_seq[:n]) ** 2)
        jerk = np.sum(np.diff(lataccel_traj) ** 2) / (DEL_T ** 2) if len(lataccel_traj) > 1 else 0
        return self.w_track * tracking + self.w_jerk * jerk
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Update current lataccel first
        self.current_lataccel = current_lataccel
        
        # Build target sequence for horizon
        if future_plan and len(future_plan.lataccel) >= self.horizon:
            target_seq = np.array(future_plan.lataccel[:self.horizon])
        else:
            target_seq = np.full(self.horizon, target_lataccel)
        
        # CEM optimization
        mean = np.full(self.horizon, self.prev_steer)
        std = np.full(self.horizon, 0.3)
        
        for iteration in range(self.n_iterations):
            # Sample candidates
            candidates = np.random.normal(mean, std, (self.n_samples, self.horizon))
            candidates = np.clip(candidates, STEER_RANGE[0], STEER_RANGE[1])
            
            # Evaluate each candidate
            costs = np.zeros(self.n_samples)
            for i, steer_seq in enumerate(candidates):
                lataccel_traj = self.rollout(steer_seq, state, future_plan)
                costs[i] = self.compute_cost(lataccel_traj, target_seq, steer_seq)
            
            # Select elite samples
            elite_idx = np.argsort(costs)[:self.n_elite]
            elite = candidates[elite_idx]
            
            # Update distribution
            mean = np.mean(elite, axis=0)
            std = np.std(elite, axis=0) + 0.05  # prevent collapse
        
        # Use mean of final distribution
        steer = mean[0]
        steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
        
        # Update all histories together to keep in sync
        self.state_history.append(state)
        self.action_history.append(steer)
        self.lataccel_history.append(current_lataccel)
        
        # Trim histories - all together
        max_len = CONTEXT_LENGTH * 2
        if len(self.state_history) > max_len:
            self.state_history = self.state_history[-CONTEXT_LENGTH:]
            self.action_history = self.action_history[-CONTEXT_LENGTH:]
            self.lataccel_history = self.lataccel_history[-CONTEXT_LENGTH:]
        
        self.prev_steer = steer
        return steer

