"""
Hybrid MPC: Neural policy warm start + ONNX dynamics evaluation.

- Neural model proposes initial steering trajectory (learned prior)
- ONNX physics model evaluates candidates (accurate dynamics)
- CEM optimizes around neural proposal

This combines the best of both:
- Neural: learns patterns from data, good initial guess
- ONNX: accurate dynamics for evaluation
- MPC: optimizes actual cost, can beat imitation
"""

from . import BaseController
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from pathlib import Path
from collections import namedtuple

# Constants
NEURAL_CONTEXT = 10
ONNX_CONTEXT = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
ACC_G = 9.81

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
    def __init__(self):
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[token]


class SteerTransformer(nn.Module):
    def __init__(self, context_len=NEURAL_CONTEXT, context_dim=6, current_dim=5, 
                 d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.register_buffer('ctx_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('ctx_std', torch.tensor([15., 1., 0.5, 1., 0.5, 1.]))
        self.register_buffer('cur_mean', torch.tensor([20., 0., 0., 0., 0.]))
        self.register_buffer('cur_std', torch.tensor([15., 1., 0.5, 1., 1.]))
        
        self.context_proj = nn.Linear(context_dim, d_model)
        self.current_proj = nn.Linear(current_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(context_len + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        ctx_emb = self.context_proj(context)
        cur_emb = self.current_proj(current).unsqueeze(1)
        seq = torch.cat([ctx_emb, cur_emb], dim=1) + self.pos_emb
        out = self.transformer(seq)
        return self.head(out[:, -1]).squeeze(-1)


class Controller(BaseController):
    def __init__(self):
        # Load neural policy (for warm start) - prefer smooth model
        neural_path = Path(__file__).parent.parent / "models" / "steer_model_smooth.pt"
        if not neural_path.exists():
            neural_path = Path(__file__).parent.parent / "models" / "steer_model.pt"
        if neural_path.exists():
            checkpoint = torch.load(neural_path, map_location='cpu', weights_only=True)
            self.neural = SteerTransformer(d_model=checkpoint.get('hidden', 128))
            self.neural.load_state_dict(checkpoint['model_state'])
            self.neural.eval()
            self.use_neural = True
        else:
            self.use_neural = False
            print("Warning: No neural model found, using random initialization")
        
        # Load ONNX dynamics model
        onnx_path = Path(__file__).parent.parent / "models" / "tinyphysics.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with open(onnx_path, "rb") as f:
            self.onnx = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])
        self.tokenizer = LataccelTokenizer()
        
        # MPC parameters
        self.horizon = 10           # look ahead 10 steps (1 second)
        self.n_samples = 24         # more samples for longer horizon
        self.n_elite = 6
        self.n_iters = 2
        
        # Cost weights
        self.w_track = 50.0
        self.w_jerk = 10.0           # increased to reduce oscillation
        self.w_steer_rate = 5.0      # penalize steering changes
        
        # Smoothing
        self.max_steer_rate = 0.2    # max change per step
        self.prev_plan = None        # warm start from previous plan
        
        # History for ONNX model
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        
        # History for neural model  
        self.neural_context = []
        self.prev_steer = 0.0
        self.current_lataccel = 0.0
    
    def predict_lataccel_onnx(self, states, actions, preds):
        """Call ONNX model for dynamics prediction"""
        tokenized = self.tokenizer.encode(preds)
        raw = [list(s) for s in states]
        inp_states = np.column_stack([actions, raw]).astype(np.float32)
        
        res = self.onnx.run(None, {
            'states': np.expand_dims(inp_states, 0),
            'tokens': np.expand_dims(tokenized, 0).astype(np.int64)
        })[0]
        
        return self.tokenizer.decode(np.argmax(res[0, -1]))
    
    @torch.no_grad()
    def get_neural_trajectory(self, state, future_plan, horizon):
        """Use neural model to propose initial steering sequence"""
        if not self.use_neural or len(self.neural_context) < NEURAL_CONTEXT:
            return np.full(horizon, self.prev_steer)
        
        trajectory = []
        ctx = [list(row) for row in self.neural_context[-NEURAL_CONTEXT:]]
        lat = self.current_lataccel
        prev_s = self.prev_steer
        
        for t in range(horizon):
            # Get future state
            if future_plan and t < len(future_plan.v_ego):
                v = future_plan.v_ego[t]
                a = future_plan.a_ego[t]
                roll = future_plan.roll_lataccel[t] if t < len(future_plan.roll_lataccel) else state.roll_lataccel
                target = future_plan.lataccel[t] if t < len(future_plan.lataccel) else 0
            else:
                v, a, roll, target = state.v_ego, state.a_ego, state.roll_lataccel, 0
            
            # Neural prediction
            ctx_t = torch.tensor(ctx[-NEURAL_CONTEXT:], dtype=torch.float32).unsqueeze(0)
            cur_t = torch.tensor([v, a, roll, target, lat], dtype=torch.float32).unsqueeze(0)
            steer = self.neural(ctx_t, cur_t).item()
            steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
            trajectory.append(steer)
            
            # Update context for next step (simple dynamics approximation)
            ctx.append([v, a, roll, target, steer, lat])
            if len(ctx) > NEURAL_CONTEXT:
                ctx = ctx[-NEURAL_CONTEXT:]
            prev_s = steer
        
        return np.array(trajectory)
    
    def rollout_onnx(self, steer_seq, state, future_plan):
        """Rollout using ONNX dynamics - returns predicted lataccel trajectory"""
        n = min(len(self.state_history), len(self.action_history), len(self.lataccel_history))
        n = min(n, ONNX_CONTEXT)
        
        states = list(self.state_history[-n:]) if n > 0 else []
        actions = list(self.action_history[-n:]) if n > 0 else []
        preds = list(self.lataccel_history[-n:]) if n > 0 else []
        
        lat = self.current_lataccel
        traj = []
        
        for t, steer in enumerate(steer_seq):
            if future_plan and t < len(future_plan.v_ego):
                next_state = State(
                    roll_lataccel=future_plan.roll_lataccel[t] if t < len(future_plan.roll_lataccel) else state.roll_lataccel,
                    v_ego=future_plan.v_ego[t],
                    a_ego=future_plan.a_ego[t]
                )
            else:
                next_state = state
            
            states.append(next_state)
            actions.append(steer)
            preds.append(lat)
            
            if len(states) > ONNX_CONTEXT:
                states, actions, preds = states[-ONNX_CONTEXT:], actions[-ONNX_CONTEXT:], preds[-ONNX_CONTEXT:]
            
            if len(states) >= ONNX_CONTEXT:
                pred = self.predict_lataccel_onnx(states, actions, preds)
                pred = np.clip(pred, lat - MAX_ACC_DELTA, lat + MAX_ACC_DELTA)
                lat = pred
            
            traj.append(lat)
        
        return np.array(traj)
    
    def compute_cost(self, lat_traj, target_seq, steer_seq):
        n = min(len(lat_traj), len(target_seq))
        tracking = np.sum((lat_traj[:n] - target_seq[:n]) ** 2)
        jerk = np.sum(np.diff(lat_traj) ** 2) / DEL_T**2 if len(lat_traj) > 1 else 0
        steer_rate = np.sum(np.diff(steer_seq) ** 2) if len(steer_seq) > 1 else 0
        return self.w_track * tracking + self.w_jerk * jerk + self.w_steer_rate * steer_rate
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.current_lataccel = current_lataccel
        
        # Update neural context
        obs = [state.v_ego, state.a_ego, state.roll_lataccel, target_lataccel, self.prev_steer, current_lataccel]
        self.neural_context.append(obs)
        if len(self.neural_context) > NEURAL_CONTEXT:
            self.neural_context = self.neural_context[-NEURAL_CONTEXT:]
        
        # Target sequence
        if future_plan and len(future_plan.lataccel) >= self.horizon:
            target_seq = np.array(future_plan.lataccel[:self.horizon])
        else:
            target_seq = np.full(self.horizon, target_lataccel)
        
        # Get neural proposal - use 100% as starting point
        neural_proposal = self.get_neural_trajectory(state, future_plan, self.horizon)
        mean = neural_proposal.copy()
        
        std = np.full(self.horizon, 0.15)  # explore around neural prediction
        
        for _ in range(self.n_iters):
            # Sample around mean
            samples = np.random.normal(mean, std, (self.n_samples, self.horizon))
            samples = np.clip(samples, STEER_RANGE[0], STEER_RANGE[1])
            
            # Always include neural proposal
            samples[0] = neural_proposal
            
            # Evaluate with ONNX dynamics
            costs = np.zeros(self.n_samples)
            for i, seq in enumerate(samples):
                traj = self.rollout_onnx(seq, state, future_plan)
                costs[i] = self.compute_cost(traj, target_seq, seq)
            
            # Elite selection
            elite_idx = np.argsort(costs)[:self.n_elite]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0)
            std = np.std(elite, axis=0) + 0.02
        
        # Save plan for next step warm start
        self.prev_plan = mean.copy()
        
        # Output (no rate limiting - model trained to be smooth)
        steer = np.clip(mean[0], STEER_RANGE[0], STEER_RANGE[1])
        
        # Update ONNX history
        self.state_history.append(state)
        self.action_history.append(steer)
        self.lataccel_history.append(current_lataccel)
        
        if len(self.state_history) > ONNX_CONTEXT * 2:
            self.state_history = self.state_history[-ONNX_CONTEXT:]
            self.action_history = self.action_history[-ONNX_CONTEXT:]
            self.lataccel_history = self.lataccel_history[-ONNX_CONTEXT:]
        
        self.prev_steer = steer
        return steer

