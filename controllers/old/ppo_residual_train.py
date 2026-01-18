"""
Residual PPO Training - Learn corrections to PID controller

The neural network learns a small residual to add to PID's output:
    final_action = PID_action + NN_residual

This is much easier to learn than control from scratch.

Usage:
    python -m controllers.ppo_residual_train --num_segs 100 --iterations 500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

from tinyphysics import (
    TinyPhysicsModel, CONTEXT_LENGTH, CONTROL_START_IDX,
    FUTURE_PLAN_STEPS, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    State, FuturePlan, ACC_G
)
import pandas as pd


# ============================================================================
# PID Controller (embedded)
# ============================================================================

class PIDController:
    def __init__(self, p=0.195, i=0.100, d=-0.053):
        self.p = p
        self.i = i
        self.d = d
        self.error_integral = 0
        self.prev_error = 0

    def reset(self):
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        return action, error, self.error_integral, error_diff


# ============================================================================
# Residual Policy Network
# ============================================================================

class ResidualPolicy(nn.Module):
    """
    Learns a residual correction to PID.

    Input includes:
    - History features (same as before)
    - Current state + PID action + PID internals
    - Future plan

    Output: small residual in [-max_residual, +max_residual]
    """

    def __init__(
        self,
        hist_features=6,
        fut_features=4,
        current_features=10,  # 6 base + 4 PID (action, error, integral, deriv)
        seq_len=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        max_residual=0.5,  # Maximum correction magnitude
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.max_residual = max_residual

        # Embeddings
        self.hist_embed = nn.Linear(hist_features, d_model)
        self.fut_embed = nn.Linear(fut_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.hist_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fut_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
            ),
            num_layers
        )

        # Current state + PID info embedding
        self.current_embed = nn.Sequential(
            nn.Linear(current_features, d_model),
            nn.ReLU(),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head - outputs RESIDUAL (small correction)
        # Initialize to output near-zero so we start as pure PID
        self.actor_mean = nn.Linear(128, 1)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        self.actor_log_std = nn.Parameter(torch.tensor(-4.0))  # Start nearly deterministic (std≈0.018)

        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, hist, current, future):
        h = self.hist_embed(hist) + self.pos_embed
        h = self.hist_transformer(h)
        h_summary = h[:, -1]

        f = self.fut_embed(future) + self.pos_embed
        f = self.fut_transformer(f)
        f_summary = f[:, 0]

        c = self.current_embed(current)

        fused = torch.cat([h_summary, f_summary, c], dim=-1)
        features = self.fusion(fused)
        return features

    def get_action_and_value(self, hist, current, future, deterministic=False):
        features = self.forward(hist, current, future)

        # Output residual scaled to [-max_residual, max_residual]
        mean = torch.tanh(self.actor_mean(features)) * self.max_residual
        std = self.actor_log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            residual = mean
        else:
            residual = dist.sample()

        residual = torch.clamp(residual, -self.max_residual, self.max_residual)
        log_prob = dist.log_prob(residual)
        entropy = dist.entropy()
        value = self.critic(features)

        return residual, log_prob, entropy, value

    def evaluate_actions(self, hist, current, future, residuals):
        features = self.forward(hist, current, future)

        mean = torch.tanh(self.actor_mean(features)) * self.max_residual
        std = self.actor_log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(residuals)
        entropy = dist.entropy()
        values = self.critic(features)

        return log_probs, entropy, values

    def get_policy_stats(self):
        with torch.no_grad():
            return {
                'actor_log_std': self.actor_log_std.item(),
                'actor_std': self.actor_log_std.exp().item(),
                'max_residual': self.max_residual,
            }


# ============================================================================
# Environment with PID
# ============================================================================

class ResidualControlsEnv:
    """Environment where NN learns residual corrections to PID."""

    def __init__(self, model_path: str, context_length: int = 20):
        self.model_path = model_path
        self.context_length = context_length
        self.sim_model = TinyPhysicsModel(model_path, debug=False)
        self.pid = PIDController()

    def reset(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data(data_path)
        self.pid.reset()

        self.step_idx = CONTEXT_LENGTH
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []

        for i in range(self.step_idx):
            state = self._get_state(i)
            self.state_history.append(state)
            self.action_history.append(self.data['steer_command'].values[i])
            self.lataccel_history.append(self.data['target_lataccel'].values[i])
            self.target_history.append(self.data['target_lataccel'].values[i])

        self.current_lataccel = self.lataccel_history[-1]
        self.prev_lataccel = self.lataccel_history[-2] if len(self.lataccel_history) > 1 else 0.0

        from hashlib import md5
        seed = int(md5(data_path.encode()).hexdigest(), 16) % 10**4
        np.random.seed(seed)

        return self._get_obs()

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        return pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values,
        })

    def _get_state(self, idx: int) -> State:
        row = self.data.iloc[idx]
        return State(
            roll_lataccel=row['roll_lataccel'],
            v_ego=row['v_ego'],
            a_ego=row['a_ego'],
        )

    def _get_future_plan(self, idx: int) -> FuturePlan:
        end_idx = min(idx + FUTURE_PLAN_STEPS, len(self.data))
        return FuturePlan(
            lataccel=self.data['target_lataccel'].values[idx+1:end_idx].tolist(),
            roll_lataccel=self.data['roll_lataccel'].values[idx+1:end_idx].tolist(),
            v_ego=self.data['v_ego'].values[idx+1:end_idx].tolist(),
            a_ego=self.data['a_ego'].values[idx+1:end_idx].tolist(),
        )

    def _get_obs(self):
        """Build observation including PID state."""
        # History
        hist = np.zeros((self.context_length, 6), dtype=np.float32)
        for i, idx in enumerate(range(max(0, len(self.state_history) - self.context_length), len(self.state_history))):
            if idx < len(self.state_history):
                s = self.state_history[idx]
                hist[i] = [
                    s.v_ego / 30.0,
                    s.a_ego / 4.0,
                    s.roll_lataccel / 2.0,
                    self.lataccel_history[idx] / 5.0 if idx < len(self.lataccel_history) else 0,
                    self.target_history[idx] / 5.0 if idx < len(self.target_history) else 0,
                    self.action_history[idx] / 2.0 if idx < len(self.action_history) else 0,
                ]

        # Current state
        state = self._get_state(self.step_idx) if self.step_idx < len(self.data) else self.state_history[-1]
        target = self.data['target_lataccel'].values[self.step_idx] if self.step_idx < len(self.data) else 0

        # Get PID action and internals
        pid_action, pid_error, pid_integral, pid_deriv = self.pid.update(target, self.current_lataccel)
        # Don't update PID state yet - this is just for observation
        self.pid.error_integral -= pid_error  # Undo the integral update
        self.pid.prev_error = pid_error - pid_deriv  # Undo prev_error update

        error = target - self.current_lataccel

        # Current features: base state + PID info
        current = np.array([
            state.v_ego / 30.0,
            state.a_ego / 4.0,
            state.roll_lataccel / 2.0,
            self.current_lataccel / 5.0,
            target / 5.0,
            error / 5.0,
            # PID state (normalized)
            pid_action / 2.0,
            pid_error / 5.0,
            pid_integral / 50.0,  # Integral can grow large
            pid_deriv / 2.0,
        ], dtype=np.float32)

        # Future
        future_plan = self._get_future_plan(self.step_idx) if self.step_idx < len(self.data) else None
        future = np.zeros((self.context_length, 4), dtype=np.float32)

        if future_plan:
            future_len = min(self.context_length, len(future_plan.v_ego))
            for i in range(future_len):
                future[i] = [
                    future_plan.v_ego[i] / 30.0 if i < len(future_plan.v_ego) else 0,
                    future_plan.a_ego[i] / 4.0 if i < len(future_plan.a_ego) else 0,
                    future_plan.roll_lataccel[i] / 2.0 if i < len(future_plan.roll_lataccel) else 0,
                    future_plan.lataccel[i] / 5.0 if i < len(future_plan.lataccel) else 0,
                ]

        return {
            'hist': hist,
            'current': current,
            'future': future,
            'pid_action': pid_action,
        }

    def step(self, residual: float):
        """Execute PID + residual action."""
        state = self._get_state(self.step_idx)
        target = self.data['target_lataccel'].values[self.step_idx]

        # Get PID action (this updates PID state)
        pid_action, _, _, _ = self.pid.update(target, self.current_lataccel)

        # Combine PID + residual
        action = pid_action + residual
        action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))

        self.state_history.append(state)
        self.target_history.append(target)
        self.action_history.append(action)

        if self.step_idx < CONTROL_START_IDX:
            action = self.data['steer_command'].values[self.step_idx]
            self.action_history[-1] = action
            new_lataccel = self.data['target_lataccel'].values[self.step_idx]
            # Reset PID during warmup to match ground truth
            self.pid.reset()
        else:
            new_lataccel = self.sim_model.get_current_lataccel(
                sim_states=self.state_history[-CONTEXT_LENGTH:],
                actions=self.action_history[-CONTEXT_LENGTH:],
                past_preds=self.lataccel_history[-CONTEXT_LENGTH:],
            )
            MAX_ACC_DELTA = 0.5
            new_lataccel = np.clip(
                new_lataccel,
                self.current_lataccel - MAX_ACC_DELTA,
                self.current_lataccel + MAX_ACC_DELTA
            )

        self.lataccel_history.append(new_lataccel)

        # Reward
        tracking_error = (target - new_lataccel) ** 2
        jerk = ((new_lataccel - self.current_lataccel) / DEL_T) ** 2
        reward = -(LAT_ACCEL_COST_MULTIPLIER * tracking_error + jerk) * 0.01

        # Penalty for large residuals (prefer minimal corrections)
        # Higher coefficient forces NN to only correct when it really helps
        residual_penalty = 0.5 * residual ** 2
        reward -= residual_penalty

        self.prev_lataccel = self.current_lataccel
        self.current_lataccel = new_lataccel
        self.step_idx += 1

        done = self.step_idx >= len(self.data) - 1
        obs = self._get_obs() if not done else None

        info = {
            'target': target,
            'lataccel': new_lataccel,
            'tracking_error': tracking_error,
            'jerk': jerk,
            'pid_action': pid_action,
            'residual': residual,
            'action': action,
        }

        return obs, reward, done, info

    def compute_cost(self):
        COST_END_IDX = 500
        target = np.array(self.target_history)[CONTROL_START_IDX:COST_END_IDX]
        pred = np.array(self.lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

        if len(target) == 0 or len(pred) == 0:
            return {'lataccel_cost': 0, 'jerk_cost': 0, 'total_cost': 0}

        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

        return {
            'lataccel_cost': lat_accel_cost,
            'jerk_cost': jerk_cost,
            'total_cost': total_cost,
        }


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    def __init__(
        self,
        policy: ResidualPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cpu',
    ):
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t + 1]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return (
            torch.tensor(advantages, dtype=torch.float32, device=self.device),
            torch.tensor(returns, dtype=torch.float32, device=self.device),
        )

    def update(self, rollout):
        hist = torch.tensor(np.array(rollout['hist']), dtype=torch.float32, device=self.device)
        current = torch.tensor(np.array(rollout['current']), dtype=torch.float32, device=self.device)
        future = torch.tensor(np.array(rollout['future']), dtype=torch.float32, device=self.device)
        residuals = torch.tensor(np.array(rollout['residuals']), dtype=torch.float32, device=self.device).unsqueeze(-1)
        old_log_probs = torch.tensor(np.array(rollout['log_probs']), dtype=torch.float32, device=self.device)
        advantages = rollout['advantages']
        returns = rollout['returns']

        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_stats = {'policy_loss': [], 'value_loss': [], 'entropy': [],
                     'clip_frac': [], 'approx_kl': [], 'grad_norm': []}

        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(hist))

            for start in range(0, len(hist), self.batch_size):
                end = min(start + self.batch_size, len(hist))
                idx = indices[start:end]

                log_probs, entropy, values = self.policy.evaluate_actions(
                    hist[idx], current[idx], future[idx], residuals[idx]
                )

                log_ratio = log_probs - old_log_probs[idx].unsqueeze(-1)
                ratio = torch.exp(log_ratio)

                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                surr1 = ratio * advantages[idx].unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx].unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values.squeeze(-1) - returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.policy.parameters() if p.grad is not None) ** 0.5

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_stats['policy_loss'].append(policy_loss.item())
                all_stats['value_loss'].append(value_loss.item())
                all_stats['entropy'].append(entropy.mean().item())
                all_stats['clip_frac'].append(clip_frac)
                all_stats['approx_kl'].append(approx_kl)
                all_stats['grad_norm'].append(grad_norm)

        return {k: np.mean(v) for k, v in all_stats.items()} | {'adv_mean': adv_mean, 'adv_std': adv_std}


# ============================================================================
# Training Loop
# ============================================================================

def collect_rollout(env, policy, seg_paths, rollout_length, device):
    rollout = {
        'hist': [], 'current': [], 'future': [],
        'residuals': [], 'rewards': [], 'log_probs': [], 'values': [], 'dones': [],
    }

    all_residuals = []
    all_pid_actions = []
    all_final_actions = []

    steps = 0
    episode_rewards = []
    episode_costs = []

    while steps < rollout_length:
        seg_path = np.random.choice(seg_paths)
        obs = env.reset(str(seg_path))

        ep_reward = 0
        done = False

        while not done and steps < rollout_length:
            hist_t = torch.tensor(obs['hist'], dtype=torch.float32, device=device).unsqueeze(0)
            curr_t = torch.tensor(obs['current'], dtype=torch.float32, device=device).unsqueeze(0)
            fut_t = torch.tensor(obs['future'], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                residual, log_prob, _, value = policy.get_action_and_value(
                    hist_t, curr_t, fut_t, deterministic=False
                )

            residual_np = residual.squeeze().cpu().numpy()
            next_obs, reward, done, info = env.step(residual_np)

            rollout['hist'].append(obs['hist'])
            rollout['current'].append(obs['current'])
            rollout['future'].append(obs['future'])
            rollout['residuals'].append(residual_np)
            rollout['rewards'].append(reward)
            rollout['log_probs'].append(log_prob.squeeze().cpu().numpy())
            rollout['values'].append(value.squeeze().cpu().numpy())
            rollout['dones'].append(float(done))

            all_residuals.append(info['residual'])
            all_pid_actions.append(info['pid_action'])
            all_final_actions.append(info['action'])

            obs = next_obs
            ep_reward += reward
            steps += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(env.compute_cost())

    # Bootstrap
    if not done and obs is not None:
        hist_t = torch.tensor(obs['hist'], dtype=torch.float32, device=device).unsqueeze(0)
        curr_t = torch.tensor(obs['current'], dtype=torch.float32, device=device).unsqueeze(0)
        fut_t = torch.tensor(obs['future'], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, value = policy.get_action_and_value(hist_t, curr_t, fut_t)
            rollout['values'].append(value.squeeze().cpu().numpy())
    else:
        rollout['values'].append(0.0)

    stats = {
        'residual_mean': np.mean(all_residuals),
        'residual_std': np.std(all_residuals),
        'residual_abs_mean': np.mean(np.abs(all_residuals)),
        'pid_action_mean': np.mean(all_pid_actions),
        'final_action_mean': np.mean(all_final_actions),
    }

    return rollout, episode_rewards, episode_costs, stats


def evaluate(env, policy, eval_paths, device, num_eval=5):
    policy.eval()
    costs = []

    for seg_path in eval_paths[:num_eval]:
        obs = env.reset(str(seg_path))
        done = False

        while not done:
            hist_t = torch.tensor(obs['hist'], dtype=torch.float32, device=device).unsqueeze(0)
            curr_t = torch.tensor(obs['current'], dtype=torch.float32, device=device).unsqueeze(0)
            fut_t = torch.tensor(obs['future'], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                residual, _, _, _ = policy.get_action_and_value(hist_t, curr_t, fut_t, deterministic=True)

            obs, _, done, _ = env.step(residual.squeeze().cpu().numpy())
            if obs is None:
                break

        costs.append(env.compute_cost())

    policy.train()

    return {
        'eval_lataccel_cost': np.mean([c['lataccel_cost'] for c in costs]),
        'eval_jerk_cost': np.mean([c['jerk_cost'] for c in costs]),
        'eval_total_cost': np.mean([c['total_cost'] for c in costs]),
    }


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*70}")
    print(f"Residual PPO Training - Learning corrections to PID")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Max residual: {args.max_residual}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path = Path(str(args.data_path) + '_mini')

    seg_paths = sorted(data_path.glob('*.csv'))[:args.num_segs]
    print(f"Training segments: {len(seg_paths)}")

    eval_paths = seg_paths[-5:] if len(seg_paths) > 10 else seg_paths[:2]
    train_paths = seg_paths[:-5] if len(seg_paths) > 10 else seg_paths

    run_dir = Path('runs') / f"residual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    config = vars(args) | {'device': device}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize
    env = ResidualControlsEnv(args.model_path)
    policy = ResidualPolicy(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_residual=args.max_residual,
    )
    trainer = PPOTrainer(policy, lr=args.lr, device=device, entropy_coef=args.entropy_coef)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"{'='*70}\n")

    # First evaluate pure PID as baseline
    print("Evaluating pure PID baseline...")
    pid_costs = []
    for seg_path in eval_paths[:5]:
        obs = env.reset(str(seg_path))
        done = False
        while not done:
            obs, _, done, _ = env.step(0.0)  # Zero residual = pure PID
            if obs is None:
                break
        pid_costs.append(env.compute_cost())
    pid_baseline = np.mean([c['total_cost'] for c in pid_costs])
    print(f"PID baseline cost: {pid_baseline:.2f}\n")

    # Training
    log_data = []
    best_cost = float('inf')

    header = (
        f"{'Iter':>5} | {'Cost':>8} | {'vsPID':>7} | {'LatAcc':>7} | {'Jerk':>7} | "
        f"{'ResMean':>7} | {'|Res|':>6} | {'Std':>6} | "
        f"{'PiLoss':>8} | {'Ent':>6} | {'KL':>7}"
    )
    print(header)
    print('-' * len(header))

    for iteration in range(args.iterations):
        rollout, ep_rewards, ep_costs, rollout_stats = collect_rollout(
            env, policy, train_paths, args.rollout_length, device
        )

        advantages, returns = trainer.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        rollout['advantages'] = advantages
        rollout['returns'] = returns

        update_stats = trainer.update(rollout)
        policy_stats = policy.get_policy_stats()

        mean_lataccel = np.mean([c['lataccel_cost'] for c in ep_costs])
        mean_jerk = np.mean([c['jerk_cost'] for c in ep_costs])
        mean_cost = np.mean([c['total_cost'] for c in ep_costs])
        vs_pid = mean_cost - pid_baseline

        log_entry = {
            'iteration': iteration,
            'total_cost': mean_cost,
            'vs_pid': vs_pid,
            'lataccel_cost': mean_lataccel,
            'jerk_cost': mean_jerk,
            'residual_mean': rollout_stats['residual_mean'],
            'residual_std': rollout_stats['residual_std'],
            'residual_abs_mean': rollout_stats['residual_abs_mean'],
            **update_stats,
            **policy_stats,
        }

        if iteration % args.eval_freq == 0:
            eval_stats = evaluate(env, policy, eval_paths, device)
            log_entry.update(eval_stats)
            eval_cost = eval_stats['eval_total_cost']
        else:
            eval_cost = None

        log_data.append(log_entry)

        # Console output
        vs_pid_str = f"{vs_pid:+7.1f}"
        print(
            f"{iteration:>5} | {mean_cost:>8.1f} | {vs_pid_str} | {mean_lataccel:>7.2f} | {mean_jerk:>7.2f} | "
            f"{rollout_stats['residual_mean']:>+7.3f} | {rollout_stats['residual_abs_mean']:>6.3f} | "
            f"{policy_stats['actor_std']:>6.3f} | "
            f"{update_stats['policy_loss']:>8.4f} | {update_stats['entropy']:>6.3f} | {update_stats['approx_kl']:>7.4f}"
        )

        # Save best
        cost_to_check = eval_cost if eval_cost is not None else mean_cost
        if cost_to_check < best_cost:
            best_cost = cost_to_check
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'cost': cost_to_check,
                'config': config,
            }, Path('models') / 'ppo_residual_best.pt')
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'cost': cost_to_check,
            }, run_dir / 'best.pt')
            improvement = pid_baseline - cost_to_check
            print(f"  --> Saved best (cost={cost_to_check:.2f}, {improvement:+.1f} vs PID)")

        if iteration % args.save_freq == 0 and iteration > 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
            }, run_dir / f'checkpoint_{iteration}.pt')

        if iteration % 10 == 0:
            pd.DataFrame(log_data).to_csv(run_dir / 'training_log.csv', index=False)

    # Final save
    torch.save({'policy_state_dict': policy.state_dict()}, run_dir / 'final.pt')
    torch.save({'policy_state_dict': policy.state_dict()}, Path('models') / 'ppo_residual_final.pt')
    pd.DataFrame(log_data).to_csv(run_dir / 'training_log.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"PID baseline: {pid_baseline:.2f}")
    print(f"Best cost: {best_cost:.2f} ({pid_baseline - best_cost:+.1f} vs PID)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_segs', type=int, default=100)
    parser.add_argument('--num_segs_rand', action='store_true', help='Randomly sample segments instead of first N')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--rollout_length', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.05)  # Higher to encourage exploration growth
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_residual', type=float, default=0.3, help='Max correction magnitude')
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=50)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
