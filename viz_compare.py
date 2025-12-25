"""
Visualization tool: Run any controller and compare against neural model predictions.

Usage:
    python viz_compare.py --data_path data/00000.csv --controller pid
    python viz_compare.py --data_path data/00000.csv --controller pid_ff_scheduled_tune
"""

import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import namedtuple

# Import from tinyphysics
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, 
    CONTROL_START_IDX, CONTEXT_LENGTH, ACC_G, STEER_RANGE
)

# Neural model architecture (must match training)
NEURAL_CONTEXT = 10

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


class NeuralPredictor:
    """Wraps the neural model for prediction alongside another controller"""
    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        hidden = checkpoint.get('hidden', 128)
        self.model = SteerTransformer(d_model=hidden)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.context = []
        self.prev_steer = 0.0
    
    @torch.no_grad()
    def predict(self, v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, actual_steer):
        """
        Predict what steer the neural model thinks is correct.
        Uses actual_steer in context (what the real controller did).
        """
        # Add to context using actual controller's steer
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, actual_steer, current_lataccel]
        self.context.append(obs)
        
        if len(self.context) > NEURAL_CONTEXT:
            self.context = self.context[-NEURAL_CONTEXT:]
        
        if len(self.context) < NEURAL_CONTEXT:
            return 0.0  # not enough context yet
        
        ctx = torch.tensor(self.context, dtype=torch.float32).unsqueeze(0)
        cur = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel], 
                          dtype=torch.float32).unsqueeze(0)
        
        pred = self.model(ctx, cur).item()
        return np.clip(pred, STEER_RANGE[0], STEER_RANGE[1])


def run_comparison(data_path: str, controller_name: str, model_path: str, neural_model_path: str):
    """Run simulation with controller and track neural model predictions"""
    
    # Load controller
    controller_module = importlib.import_module(f'controllers.{controller_name}')
    controller = controller_module.Controller()
    
    # Load physics model and simulator
    physics_model = TinyPhysicsModel(model_path, debug=False)
    sim = TinyPhysicsSimulator(physics_model, data_path, controller=controller, debug=False)
    
    # Load neural predictor
    neural = NeuralPredictor(neural_model_path)
    
    # Storage for comparison
    steps = []
    target_lataccels = []
    current_lataccels = []
    controller_steers = []
    neural_steers = []
    ground_truth_steers = []
    
    # Get ground truth steers from data
    df = pd.read_csv(data_path)
    gt_steer = -df['steerCommand'].values  # flip sign like tinyphysics
    
    # Run simulation
    print(f"Running {controller_name} controller with neural comparison...")
    
    for step in range(CONTEXT_LENGTH, len(sim.data)):
        # Get state before step
        state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
        
        # Run one simulation step
        sim.step()
        
        # Get what the controller did
        actual_steer = sim.action_history[-1]
        current_lat = sim.current_lataccel_history[-1]
        
        # Get neural model's prediction (seeing what controller actually did)
        neural_pred = neural.predict(
            state.v_ego, state.a_ego, state.roll_lataccel,
            target, current_lat, actual_steer
        )
        
        # Store for plotting
        steps.append(sim.step_idx - 1)
        target_lataccels.append(target)
        current_lataccels.append(current_lat)
        controller_steers.append(actual_steer)
        neural_steers.append(neural_pred)
        
        # Ground truth (only valid for first 100 steps)
        if sim.step_idx - 1 < len(gt_steer) and not np.isnan(gt_steer[sim.step_idx - 1]):
            ground_truth_steers.append(gt_steer[sim.step_idx - 1])
        else:
            ground_truth_steers.append(np.nan)
    
    # Compute cost
    cost = sim.compute_cost()
    print(f"Cost: lataccel={cost['lataccel_cost']:.4f}, jerk={cost['jerk_cost']:.4f}, total={cost['total_cost']:.4f}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Lateral Acceleration
    ax1 = axes[0]
    ax1.plot(steps, target_lataccels, 'b-', label='Target LatAccel', linewidth=1.5)
    ax1.plot(steps, current_lataccels, 'r-', label='Current LatAccel', alpha=0.8)
    ax1.axvline(x=CONTROL_START_IDX, color='k', linestyle='--', alpha=0.5, label='Control Start')
    ax1.set_ylabel('Lateral Acceleration')
    ax1.set_title(f'Lateral Acceleration Tracking | Controller: {controller_name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steering Commands Comparison
    ax2 = axes[1]
    ax2.plot(steps, controller_steers, 'b-', label=f'{controller_name} Steer', linewidth=1.5)
    ax2.plot(steps, neural_steers, 'g-', label='Neural Predicted Steer', alpha=0.7)
    ax2.axvline(x=CONTROL_START_IDX, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Steer Command')
    ax2.set_title('Steering: Controller vs Neural Model Prediction')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Steering Error (Neural - Controller)
    ax3 = axes[2]
    steer_diff = np.array(neural_steers) - np.array(controller_steers)
    ax3.plot(steps, steer_diff, 'purple', label='Neural - Controller', alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=CONTROL_START_IDX, color='k', linestyle='--', alpha=0.5)
    ax3.fill_between(steps, steer_diff, 0, alpha=0.3, color='purple')
    ax3.set_ylabel('Steer Difference')
    ax3.set_xlabel('Step')
    ax3.set_title(f'Steering Difference | RMSE: {np.sqrt(np.nanmean(steer_diff[CONTROL_START_IDX:]**2)):.4f}')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'viz_compare_{controller_name}.png', dpi=150)
    print(f"Saved: viz_compare_{controller_name}.png")
    plt.show()
    
    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare controller with neural model predictions")
    parser.add_argument("--data_path", type=str, default="data/00000.csv")
    parser.add_argument("--controller", type=str, default="pid")
    parser.add_argument("--model_path", type=str, default="models/tinyphysics.onnx")
    parser.add_argument("--neural_model", type=str, default="models/steer_model.pt")
    args = parser.parse_args()
    
    run_comparison(args.data_path, args.controller, args.model_path, args.neural_model)

