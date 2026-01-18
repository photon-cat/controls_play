"""
PPO Training for Controls Challenge

Usage:
    python -m controllers.ppo_train --num_segs 100 --iterations 500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from tinyphysics import (
    TinyPhysicsModel, CONTEXT_LENGTH, CONTROL_START_IDX,
    FUTURE_PLAN_STEPS, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    State, FuturePlan, ACC_G
)
import pandas as pd

# ============================================================================
# Model Architecture
# ============================================================================

class TemporalPolicy(nn.Module):
    """Transformer-based policy for controls."""

    def __init__(
        self,
        hist_features=6,
        fut_features=4,
        current_features=6,
        seq_len=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Embeddings
        self.hist_embed = nn.Linear(hist_features, d_model)
        self.fut_embed = nn.Linear(fut_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
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

        # Current state embedding
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

        # Actor head
        self.actor_mean = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.tensor(-0.5))

        # Critic head
        self.critic = nn.Linear(128, 1)

        self.action_scale = 2.0  # STEER_RANGE is [-2, 2]

    def forward(self, hist, current, future):
        # Encode history
        h = self.hist_embed(hist) + self.pos_embed
        h = self.hist_transformer(h)
        h_summary = h[:, -1]

        # Encode future
        f = self.fut_embed(future) + self.pos_embed
        f = self.fut_transformer(f)
        f_summary = f[:, 0]

        # Encode current
        c = self.current_embed(current)

        # Fuse
        fused = torch.cat([h_summary, f_summary, c], dim=-1)
        features = self.fusion(fused)

        return features

    def get_action_and_value(self, hist, current, future, deterministic=False):
        features = self.forward(hist, current, future)

        mean = torch.tanh(self.actor_mean(features)) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        action = torch.clamp(action, -self.action_scale, self.action_scale)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features)

        return action, log_prob, entropy, value

    def evaluate_actions(self, hist, current, future, actions):
        features = self.forward(hist, current, future)

        mean = torch.tanh(self.actor_mean(features)) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(features)

        return log_probs, entropy, values

    def get_policy_stats(self):
        """Get current policy parameters for logging."""
        with torch.no_grad():
            return {
                'actor_log_std': self.actor_log_std.item(),
                'actor_std': self.actor_log_std.exp().item(),
            }


# ============================================================================
# Environment
# ============================================================================

class ControlsEnv:
    """Gym-like wrapper for tinyphysics."""

    def __init__(self, model_path: str, context_length: int = 20):
        self.model_path = model_path
        self.context_length = context_length
        self.sim_model = TinyPhysicsModel(model_path, debug=False)

    def reset(self, data_path: str):
        """Reset with a new segment."""
        self.data_path = data_path
        self.data = self._load_data(data_path)

        # Initialize histories
        self.step_idx = CONTEXT_LENGTH
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.target_history = []

        # Fill initial history from data
        for i in range(self.step_idx):
            state = self._get_state(i)
            self.state_history.append(state)
            self.action_history.append(self.data['steer_command'].values[i])
            self.lataccel_history.append(self.data['target_lataccel'].values[i])
            self.target_history.append(self.data['target_lataccel'].values[i])

        self.current_lataccel = self.lataccel_history[-1]
        self.prev_lataccel = self.lataccel_history[-2] if len(self.lataccel_history) > 1 else 0.0

        # Seed for reproducibility
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
        """Build observation dict with history, current, future."""
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

        state = self._get_state(self.step_idx) if self.step_idx < len(self.data) else self.state_history[-1]
        target = self.data['target_lataccel'].values[self.step_idx] if self.step_idx < len(self.data) else 0
        error = target - self.current_lataccel

        current = np.array([
            state.v_ego / 30.0,
            state.a_ego / 4.0,
            state.roll_lataccel / 2.0,
            self.current_lataccel / 5.0,
            target / 5.0,
            error / 5.0,
        ], dtype=np.float32)

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

        return {'hist': hist, 'current': current, 'future': future}

    def step(self, action: float):
        """Execute action, return obs, reward, done, info."""
        action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))

        state = self._get_state(self.step_idx)
        target = self.data['target_lataccel'].values[self.step_idx]

        self.state_history.append(state)
        self.target_history.append(target)
        self.action_history.append(action)

        if self.step_idx < CONTROL_START_IDX:
            action = self.data['steer_command'].values[self.step_idx]
            self.action_history[-1] = action
            new_lataccel = self.data['target_lataccel'].values[self.step_idx]
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

        tracking_error = (target - new_lataccel) ** 2
        jerk = ((new_lataccel - self.current_lataccel) / DEL_T) ** 2
        reward = -(LAT_ACCEL_COST_MULTIPLIER * tracking_error + jerk) * 0.01

        if len(self.action_history) >= 2:
            action_diff = (action - self.action_history[-2]) ** 2
            reward -= 0.01 * action_diff

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
            'action': action,
        }

        return obs, reward, done, info

    def compute_cost(self):
        """Compute final cost metrics."""
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
        policy: TemporalPolicy,
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
        self.lr = lr

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation."""
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

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        return advantages, returns

    def update(self, rollout):
        """PPO update with detailed stats."""
        hist = torch.tensor(np.array(rollout['hist']), dtype=torch.float32, device=self.device)
        current = torch.tensor(np.array(rollout['current']), dtype=torch.float32, device=self.device)
        future = torch.tensor(np.array(rollout['future']), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(rollout['actions']), dtype=torch.float32, device=self.device).unsqueeze(-1)
        old_log_probs = torch.tensor(np.array(rollout['log_probs']), dtype=torch.float32, device=self.device)
        advantages = rollout['advantages']
        returns = rollout['returns']

        # Normalize advantages
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track stats across epochs
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_clip_fracs = []
        all_approx_kls = []
        all_grad_norms = []

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(len(hist))

            for start in range(0, len(hist), self.batch_size):
                end = min(start + self.batch_size, len(hist))
                batch_idx = indices[start:end]

                b_hist = hist[batch_idx]
                b_current = current[batch_idx]
                b_future = future[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                log_probs, entropy, values = self.policy.evaluate_actions(
                    b_hist, b_current, b_future, b_actions
                )

                # Policy loss with clipping stats
                log_ratio = log_probs - b_old_log_probs.unsqueeze(-1)
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # Clipping fraction
                clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                surr1 = ratio * b_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values.squeeze(-1) - b_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm before clipping
                grad_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.mean().item())
                all_clip_fracs.append(clip_frac)
                all_approx_kls.append(approx_kl)
                all_grad_norms.append(grad_norm)

        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'clip_frac': np.mean(all_clip_fracs),
            'approx_kl': np.mean(all_approx_kls),
            'grad_norm': np.mean(all_grad_norms),
            'adv_mean': adv_mean,
            'adv_std': adv_std,
        }


# ============================================================================
# Training Loop
# ============================================================================

def collect_rollout(env, policy, seg_paths, rollout_length, device):
    """Collect experience with detailed tracking."""
    rollout = {
        'hist': [],
        'current': [],
        'future': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'dones': [],
    }

    # Track detailed stats
    all_tracking_errors = []
    all_jerks = []
    all_actions = []

    steps = 0
    episode_rewards = []
    episode_costs = []
    episodes_completed = 0

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
                action, log_prob, _, value = policy.get_action_and_value(
                    hist_t, curr_t, fut_t, deterministic=False
                )

            action_np = action.squeeze().cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)

            rollout['hist'].append(obs['hist'])
            rollout['current'].append(obs['current'])
            rollout['future'].append(obs['future'])
            rollout['actions'].append(action_np)
            rollout['rewards'].append(reward)
            rollout['log_probs'].append(log_prob.squeeze().cpu().numpy())
            rollout['values'].append(value.squeeze().cpu().numpy())
            rollout['dones'].append(float(done))

            # Track step-level stats
            all_tracking_errors.append(info['tracking_error'])
            all_jerks.append(info['jerk'])
            all_actions.append(info['action'])

            obs = next_obs
            ep_reward += reward
            steps += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(env.compute_cost())
        episodes_completed += 1

    # Bootstrap value
    if not done and obs is not None:
        hist_t = torch.tensor(obs['hist'], dtype=torch.float32, device=device).unsqueeze(0)
        curr_t = torch.tensor(obs['current'], dtype=torch.float32, device=device).unsqueeze(0)
        fut_t = torch.tensor(obs['future'], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, value = policy.get_action_and_value(hist_t, curr_t, fut_t)
            rollout['values'].append(value.squeeze().cpu().numpy())
    else:
        rollout['values'].append(0.0)

    rollout_stats = {
        'episodes_completed': episodes_completed,
        'mean_tracking_error': np.mean(all_tracking_errors),
        'mean_jerk': np.mean(all_jerks),
        'action_mean': np.mean(all_actions),
        'action_std': np.std(all_actions),
        'action_min': np.min(all_actions),
        'action_max': np.max(all_actions),
        'reward_mean': np.mean(rollout['rewards']),
        'reward_std': np.std(rollout['rewards']),
        'reward_min': np.min(rollout['rewards']),
        'reward_max': np.max(rollout['rewards']),
    }

    return rollout, episode_rewards, episode_costs, rollout_stats


def evaluate(env, policy, eval_paths, device, num_eval=5):
    """Evaluate policy deterministically on held-out segments."""
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
                action, _, _, _ = policy.get_action_and_value(
                    hist_t, curr_t, fut_t, deterministic=True
                )

            action_np = action.squeeze().cpu().numpy()
            obs, _, done, _ = env.step(action_np)
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
    print(f"{'='*60}")
    print(f"PPO Training for Controls Challenge")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path = Path(str(args.data_path) + '_mini')

    seg_paths = sorted(data_path.glob('*.csv'))[:args.num_segs]
    print(f"Training segments: {len(seg_paths)}")

    # Split for evaluation
    eval_paths = seg_paths[-5:] if len(seg_paths) > 10 else seg_paths[:2]
    train_paths = seg_paths[:-5] if len(seg_paths) > 10 else seg_paths

    # Create output directory
    run_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Save config
    config = vars(args)
    config['device'] = device
    config['num_train_segs'] = len(train_paths)
    config['num_eval_segs'] = len(eval_paths)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize
    env = ControlsEnv(args.model_path)
    policy = TemporalPolicy(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    trainer = PPOTrainer(
        policy,
        lr=args.lr,
        device=device,
        entropy_coef=args.entropy_coef,
    )

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"{'='*60}\n")

    # Training log
    log_data = []
    best_cost = float('inf')

    # Header for console output
    header = (
        f"{'Iter':>5} | {'Reward':>8} | {'Cost':>8} | {'LatAcc':>7} | {'Jerk':>7} | "
        f"{'PiLoss':>8} | {'VLoss':>8} | {'Entropy':>7} | {'KL':>7} | {'Clip%':>6} | "
        f"{'ActMean':>7} | {'ActStd':>6} | {'GradN':>7}"
    )
    print(header)
    print('-' * len(header))

    for iteration in range(args.iterations):
        # Collect rollout
        rollout, ep_rewards, ep_costs, rollout_stats = collect_rollout(
            env, policy, train_paths, args.rollout_length, device
        )

        # Compute advantages
        advantages, returns = trainer.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
        )
        rollout['advantages'] = advantages
        rollout['returns'] = returns

        # Update policy
        update_stats = trainer.update(rollout)

        # Get policy stats
        policy_stats = policy.get_policy_stats()

        # Compute metrics
        mean_reward = np.mean(ep_rewards)
        mean_lataccel = np.mean([c['lataccel_cost'] for c in ep_costs])
        mean_jerk = np.mean([c['jerk_cost'] for c in ep_costs])
        mean_cost = np.mean([c['total_cost'] for c in ep_costs])

        # Log entry
        log_entry = {
            'iteration': iteration,
            'reward_mean': mean_reward,
            'reward_std': np.std(ep_rewards),
            'total_cost': mean_cost,
            'lataccel_cost': mean_lataccel,
            'jerk_cost': mean_jerk,
            'policy_loss': update_stats['policy_loss'],
            'value_loss': update_stats['value_loss'],
            'entropy': update_stats['entropy'],
            'approx_kl': update_stats['approx_kl'],
            'clip_frac': update_stats['clip_frac'],
            'grad_norm': update_stats['grad_norm'],
            'adv_mean': update_stats['adv_mean'],
            'adv_std': update_stats['adv_std'],
            'action_mean': rollout_stats['action_mean'],
            'action_std': rollout_stats['action_std'],
            'action_min': rollout_stats['action_min'],
            'action_max': rollout_stats['action_max'],
            'actor_std': policy_stats['actor_std'],
            'episodes': rollout_stats['episodes_completed'],
        }

        # Evaluate periodically
        if iteration % args.eval_freq == 0:
            eval_stats = evaluate(env, policy, eval_paths, device)
            log_entry.update(eval_stats)
            eval_cost = eval_stats['eval_total_cost']
        else:
            eval_cost = None

        log_data.append(log_entry)

        # Console output
        print(
            f"{iteration:>5} | {mean_reward:>8.2f} | {mean_cost:>8.1f} | {mean_lataccel:>7.2f} | {mean_jerk:>7.2f} | "
            f"{update_stats['policy_loss']:>8.4f} | {update_stats['value_loss']:>8.2f} | {update_stats['entropy']:>7.4f} | "
            f"{update_stats['approx_kl']:>7.4f} | {update_stats['clip_frac']*100:>5.1f}% | "
            f"{rollout_stats['action_mean']:>7.3f} | {rollout_stats['action_std']:>6.3f} | {update_stats['grad_norm']:>7.2f}"
        )

        # Save best model
        cost_to_check = eval_cost if eval_cost is not None else mean_cost
        if cost_to_check < best_cost:
            best_cost = cost_to_check
            save_path = Path('models') / 'ppo_best.pt'
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'cost': cost_to_check,
                'config': config,
            }, save_path)
            # Also save to run dir
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'iteration': iteration,
                'cost': cost_to_check,
                'config': config,
            }, run_dir / 'best.pt')
            print(f"  --> Saved best model (cost={cost_to_check:.2f})")

        # Periodic checkpoint
        if iteration % args.save_freq == 0 and iteration > 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'iteration': iteration,
                'config': config,
            }, run_dir / f'checkpoint_{iteration}.pt')
            print(f"  --> Saved checkpoint")

        # Save log periodically
        if iteration % 10 == 0:
            log_df = pd.DataFrame(log_data)
            log_df.to_csv(run_dir / 'training_log.csv', index=False)

    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'iteration': args.iterations,
        'config': config,
    }, run_dir / 'final.pt')

    torch.save({
        'policy_state_dict': policy.state_dict(),
        'iteration': args.iterations,
    }, Path('models') / 'ppo_final.pt')

    # Save final log
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(run_dir / 'training_log.csv', index=False)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Outputs saved to: {run_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--num_segs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--rollout_length', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluate every N iterations')
    parser.add_argument('--save_freq', type=int, default=50, help='Save checkpoint every N iterations')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
