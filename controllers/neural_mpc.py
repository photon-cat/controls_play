"""
Neural MPC controller - uses trained steer model for trajectory optimization.

Approach:
1. Use neural model to propose initial steering trajectory
2. Sample perturbations around that trajectory (MPPI-style)
3. Simulate dynamics to predict lataccel response
4. Score trajectories on tracking + jerk cost
5. Return weighted average of best actions
"""

from . import BaseController
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CONTEXT_LENGTH = 10
ACC_G = 9.81
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1


class SteerTransformer(nn.Module):
    """Same architecture as training"""
    def __init__(self, context_len=CONTEXT_LENGTH, context_dim=6, current_dim=5, 
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
            dropout=0.0, batch_first=True  # no dropout at inference
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        
        ctx_emb = self.context_proj(context)
        cur_emb = self.current_proj(current).unsqueeze(1)
        seq = torch.cat([ctx_emb, cur_emb], dim=1)
        seq = seq + self.pos_emb
        out = self.transformer(seq)
        return self.head(out[:, -1]).squeeze(-1)


class Controller(BaseController):
    def __init__(self):
        model_path = Path(__file__).parent.parent / "models" / "steer_model_noise_1.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        hidden = checkpoint.get('hidden', 128)
        self.model = SteerTransformer(d_model=hidden)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # MPC parameters
        self.horizon = 10           # lookahead steps (1 second)
        self.n_samples = 32         # candidate trajectories
        self.temperature = 0.05     # MPPI temperature (lower = more greedy)
        self.noise_std = 0.08       # perturbation noise (lower = smoother)
        
        # Dynamics model parameters (first-order lag)
        self.tau = 0.3              # time constant (slower response)
        self.k_steer = 0.7          # steering gain
        
        # Cost weights
        self.w_track = 50.0         # tracking cost (match challenge)
        self.w_jerk = 5.0           # jerk cost (increased to reduce oscillation)
        
        # Rate limiting
        self.max_steer_rate = 0.3   # max steer change per step
        
        # State
        self.context = []
        self.prev_steer = 0.0
        self.prev_lataccel = 0.0
    
    def simulate_dynamics(self, current_lataccel, steer_seq, roll_seq, v_ego):
        """First-order dynamics: lataccel responds to steer with lag"""
        pred = np.zeros(len(steer_seq))
        lat = current_lataccel
        
        k = self.k_steer * np.sqrt(v_ego / 25.0)  # velocity-dependent gain
        alpha = DEL_T / self.tau
        
        for i, (u, roll) in enumerate(zip(steer_seq, roll_seq)):
            target = k * u + roll
            lat = lat + alpha * (target - lat)
            lat = np.clip(lat, lat - MAX_ACC_DELTA, lat + MAX_ACC_DELTA)
            pred[i] = lat
        
        return pred
    
    def compute_cost(self, pred_lataccel, target_seq, steer_seq):
        """Tracking + jerk cost"""
        tracking = np.sum((pred_lataccel - target_seq) ** 2)
        jerk = np.sum(np.diff(pred_lataccel) ** 2) / (DEL_T ** 2)
        return self.w_track * tracking + self.w_jerk * jerk
    
    @torch.no_grad()
    def rollout_neural(self, context, v_ego, a_ego, roll_seq, target_seq):
        """
        Autoregressive rollout using neural model.
        Returns predicted steering sequence.
        """
        steer_seq = []
        ctx = [list(row) for row in context]  # copy
        lataccel = ctx[-1][5] if ctx else 0.0
        
        for t in range(self.horizon):
            if len(ctx) < CONTEXT_LENGTH:
                # Not enough context, use simple P control
                steer = 0.3 * (target_seq[t] - lataccel)
            else:
                # Neural prediction
                ctx_tensor = torch.tensor(ctx[-CONTEXT_LENGTH:], dtype=torch.float32).unsqueeze(0)
                cur_tensor = torch.tensor(
                    [v_ego, a_ego, roll_seq[t], target_seq[t], lataccel],
                    dtype=torch.float32
                ).unsqueeze(0)
                steer = self.model(ctx_tensor, cur_tensor).item()
            
            steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
            steer_seq.append(steer)
            
            # Simulate dynamics for next context
            k = self.k_steer * np.sqrt(v_ego / 25.0)
            alpha = DEL_T / self.tau
            target_lat = k * steer + roll_seq[t]
            lataccel = lataccel + alpha * (target_lat - lataccel)
            lataccel = np.clip(lataccel, self.prev_lataccel - MAX_ACC_DELTA, 
                              self.prev_lataccel + MAX_ACC_DELTA)
            
            # Update context for next step
            ctx.append([v_ego, a_ego, roll_seq[t], target_seq[t], steer, lataccel])
        
        return np.array(steer_seq)
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel
        
        # Update context
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, self.prev_steer, current_lataccel]
        self.context.append(obs)
        if len(self.context) > CONTEXT_LENGTH:
            self.context = self.context[-CONTEXT_LENGTH:]
        
        # Build future sequences
        if future_plan and len(future_plan.lataccel) >= self.horizon:
            target_seq = np.array(future_plan.lataccel[:self.horizon])
            roll_seq = np.array(future_plan.roll_lataccel[:self.horizon])
        else:
            target_seq = np.full(self.horizon, target_lataccel)
            roll_seq = np.full(self.horizon, roll_lataccel)
        
        # Get neural model's proposed trajectory
        nominal_steer = self.rollout_neural(
            self.context, v_ego, a_ego, roll_seq, target_seq
        )
        
        # Sample perturbations (MPPI style)
        noise = np.random.randn(self.n_samples, self.horizon) * self.noise_std
        candidates = nominal_steer + noise
        candidates = np.clip(candidates, STEER_RANGE[0], STEER_RANGE[1])
        
        # Include nominal trajectory
        candidates = np.vstack([nominal_steer, candidates])
        
        # Evaluate all candidates
        costs = np.zeros(len(candidates))
        for i, steer_seq in enumerate(candidates):
            pred_lat = self.simulate_dynamics(current_lataccel, steer_seq, roll_seq, v_ego)
            costs[i] = self.compute_cost(pred_lat, target_seq, steer_seq)
        
        # MPPI weighting
        costs = costs - np.min(costs)  # shift for numerical stability
        weights = np.exp(-costs / self.temperature)
        weights = weights / np.sum(weights)
        
        # Weighted average of first actions
        steer = np.sum(weights * candidates[:, 0])
        
        # Rate limiting - smooth out steering changes
        steer_delta = steer - self.prev_steer
        steer_delta = np.clip(steer_delta, -self.max_steer_rate, self.max_steer_rate)
        steer = self.prev_steer + steer_delta
        steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
        
        self.prev_steer = steer
        self.prev_lataccel = current_lataccel
        return steer

