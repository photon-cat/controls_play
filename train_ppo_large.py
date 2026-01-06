#!/usr/bin/env python3
"""
PPO training with larger network and reward discovery.

Key changes:
1. Larger network (512 hidden, 3 layers)
2. Log actual cost components to understand reward alignment
3. Try different reward formulations
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import time

# Constants
CONTEXT_LENGTH = 10
LOOKAHEAD_LENGTH = 10
STEER_RANGE = 2.0
ACC_G = 9.81


class LargeSteerPolicy(nn.Module):
    """
    Larger MLP policy with residual connections.
    
    Input: flattened (past_ctx, current, future_ctx) = 104 dims
    """
    def __init__(self, hidden=512, num_layers=3):
        super().__init__()
        
        input_dim = CONTEXT_LENGTH * 6 + 4 + LOOKAHEAD_LENGTH * 4  # 104
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        
        self.log_std = nn.Parameter(torch.tensor(-1.0))  # std ~0.37
        
        # Input normalization
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
    def forward(self, x, deterministic=False):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        h = F.relu(self.input_proj(x_norm))
        
        # Residual blocks
        for block in self.blocks:
            h = h + block(h)  # residual connection
            h = F.relu(h)
        
        mean = self.mean_head(h).squeeze(-1)
        mean = torch.tanh(mean) * STEER_RANGE
        
        std = torch.exp(self.log_std.clamp(-3, 0))
        
        if deterministic:
            return mean, None, None
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = action.clamp(-STEER_RANGE, STEER_RANGE)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, x, action):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        h = F.relu(self.input_proj(x_norm))
        for block in self.blocks:
            h = h + block(h)
            h = F.relu(h)
        
        mean = self.mean_head(h).squeeze(-1)
        mean = torch.tanh(mean) * STEER_RANGE
        
        std = torch.exp(self.log_std.clamp(-3, 0))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class LargeValueNetwork(nn.Module):
    """Larger value network"""
    def __init__(self, hidden=512, num_layers=3):
        super().__init__()
        
        input_dim = CONTEXT_LENGTH * 6 + 4 + LOOKAHEAD_LENGTH * 4
        
        self.input_proj = nn.Linear(input_dim, hidden)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            for _ in range(num_layers)
        ])
        
        self.head = nn.Linear(hidden, 1)
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        h = F.relu(self.input_proj(x_norm))
        for block in self.blocks:
            h = h + block(h)
            h = F.relu(h)
        
        return self.head(h).squeeze(-1)


def flatten_obs(past_ctx, current, future_ctx):
    return np.concatenate([past_ctx.flatten(), current, future_ctx.flatten()])


def run_episode_with_discovery(data_path, policy, device, deterministic=False, reward_type='default'):
    """
    Run episode and collect detailed cost information for reward discovery.
    """
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    
    class PolicyWrapper:
        def __init__(self):
            self.history = []
            self.trajectory = []
            self.prev_steer = 0.0
            self.prev_lataccel = 0.0
            self.step_idx = 0
            
            # For cost discovery
            self.lataccel_errors = []
            self.jerks = []
            self.target_lataccels = []
            self.actual_lataccels = []
            
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            self.step_idx += 1
            roll = state.roll_lataccel
            obs = [state.v_ego, state.a_ego, roll, target_lataccel, current_lataccel, self.prev_steer]
            self.history.append(obs)
            
            # Track for cost discovery
            self.target_lataccels.append(target_lataccel)
            self.actual_lataccels.append(current_lataccel)
            
            if len(self.history) < CONTEXT_LENGTH:
                steer = 0.3 * (target_lataccel - current_lataccel)
                self.prev_steer = np.clip(steer, -STEER_RANGE, STEER_RANGE)
                self.prev_lataccel = current_lataccel
                return self.prev_steer
            
            if not future_plan or len(future_plan.lataccel) < LOOKAHEAD_LENGTH:
                return self.prev_steer
            
            past_ctx = np.array(self.history[-CONTEXT_LENGTH:], dtype=np.float32)
            current = np.array([state.v_ego, state.a_ego, roll, current_lataccel], dtype=np.float32)
            future_ctx = np.stack([
                np.array(future_plan.v_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.a_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.roll_lataccel[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.lataccel[:LOOKAHEAD_LENGTH]),
            ], axis=1).astype(np.float32)
            
            flat_state = flatten_obs(past_ctx, current, future_ctx)
            state_t = torch.tensor(flat_state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, _ = policy(state_t, deterministic=deterministic)
                action = action.item()
                lp = log_prob.item() if log_prob is not None else 0.0
            
            # Compute step costs for discovery
            lataccel_err = current_lataccel - target_lataccel
            jerk = (current_lataccel - self.prev_lataccel) / 0.1
            
            self.lataccel_errors.append(lataccel_err)
            self.jerks.append(jerk)
            
            # Different reward formulations
            if reward_type == 'default':
                # Original: 5 * |error| + |jerk|
                step_cost = 5.0 * abs(lataccel_err) + abs(jerk)
            elif reward_type == 'match':
                # Match challenge cost function ratio (50:1 lataccel:jerk)
                # But scale down for numerical stability
                # Original: 5000*error² + 100*jerk² = 100*(50*error² + jerk²)
                # We use 1/100 scale: 0.5*error² + 0.01*jerk²
                step_cost = 0.5 * (lataccel_err ** 2) + 0.01 * (jerk ** 2)
            elif reward_type == 'challenge':
                # Simplified squared version
                step_cost = 5.0 * lataccel_err**2 + jerk**2
            elif reward_type == 'smooth':
                # Penalize steer rate too
                steer_rate = abs(action - self.prev_steer) / 0.1
                step_cost = 5.0 * abs(lataccel_err) + abs(jerk) + 0.1 * steer_rate
            elif reward_type == 'squared':
                # Squared errors (smoother gradients)
                step_cost = 5.0 * lataccel_err**2 + jerk**2
            else:
                step_cost = 5.0 * abs(lataccel_err) + abs(jerk)
            
            reward = -step_cost
            
            self.trajectory.append((flat_state, action, lp, reward))
            self.prev_steer = action
            self.prev_lataccel = current_lataccel
            
            return action
    
    wrapper = PolicyWrapper()
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    try:
        sim.rollout()
        cost = sim.compute_cost()
        total_cost = cost['total_cost']
        lataccel_cost = cost['lataccel_cost']
        jerk_cost = cost['jerk_cost']
    except Exception as e:
        total_cost = 10000.0
        lataccel_cost = 5000.0
        jerk_cost = 5000.0
    
    if len(wrapper.trajectory) == 0:
        return None
    
    states = np.array([t[0] for t in wrapper.trajectory], dtype=np.float32)
    actions = np.array([t[1] for t in wrapper.trajectory], dtype=np.float32)
    log_probs = np.array([t[2] for t in wrapper.trajectory], dtype=np.float32)
    rewards = np.array([t[3] for t in wrapper.trajectory], dtype=np.float32)
    
    return {
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'rewards': rewards,
        'total_cost': total_cost,
        'lataccel_cost': lataccel_cost,
        'jerk_cost': jerk_cost,
        # Discovery info
        'lataccel_errors': wrapper.lataccel_errors,
        'jerks': wrapper.jerks,
        'mean_abs_error': np.mean(np.abs(wrapper.lataccel_errors)) if wrapper.lataccel_errors else 0,
        'mean_abs_jerk': np.mean(np.abs(wrapper.jerks)) if wrapper.jerks else 0,
    }


def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return returns, advantages


def _episode_worker(args):
    data_path, policy_state, input_mean, input_std, device_str, reward_type, hidden, num_layers = args
    
    policy = LargeSteerPolicy(hidden=hidden, num_layers=num_layers)
    policy.load_state_dict(policy_state)
    policy.input_mean = torch.tensor(input_mean)
    policy.input_std = torch.tensor(input_std)
    policy.eval()
    
    device = torch.device(device_str)
    policy = policy.to(device)
    
    return run_episode_with_discovery(data_path, policy, device, deterministic=False, reward_type=reward_type)


def collect_episodes_parallel(data_paths, policy, device, n_workers=8, reward_type='default', hidden=512, num_layers=3):
    policy_state = policy.state_dict()
    input_mean = policy.input_mean.cpu().numpy()
    input_std = policy.input_std.cpu().numpy()
    device_str = str(device)
    
    args_list = [(p, policy_state, input_mean, input_std, device_str, reward_type, hidden, num_layers) for p in data_paths]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_episode_worker, args) for args in args_list]
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Episode error: {e}")
    
    return results


def ppo_update(policy, value_net, policy_optimizer, value_optimizer,
               states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, n_epochs=4, batch_size=512):
    
    device = next(policy.parameters()).device
    
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    stats = {
        'pg_loss': [], 'value_loss': [], 'entropy': [],
        'clip_frac': [], 'approx_kl': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            new_log_probs, entropy = policy.evaluate(states[batch_idx], actions[batch_idx])
            
            ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
            
            surr1 = ratio * advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch_idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            
            entropy_loss = -0.01 * entropy.mean()
            
            policy_loss = pg_loss + entropy_loss
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optimizer.step()
            
            values = value_net(states[batch_idx])
            value_loss = F.mse_loss(values, returns[batch_idx])
            
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()
            
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
                approx_kl = (old_log_probs[batch_idx] - new_log_probs).mean()
                
            stats['pg_loss'].append(pg_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.mean().item())
            stats['clip_frac'].append(clip_frac.item())
            stats['approx_kl'].append(approx_kl.item())
    
    return {k: np.mean(v) for k, v in stats.items()}


def estimate_normalization(policy, data_paths, device, hidden, num_layers, n_samples=10):
    print("Estimating input normalization...")
    
    all_states = []
    for path in data_paths[:n_samples]:
        result = run_episode_with_discovery(path, policy, device, deterministic=True)
        if result is not None:
            all_states.append(result['states'])
    
    if all_states:
        all_states = np.concatenate(all_states)
        input_mean = all_states.mean(axis=0)
        input_std = all_states.std(axis=0) + 1e-8
        return torch.tensor(input_mean, dtype=torch.float32), torch.tensor(input_std, dtype=torch.float32)
    
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes_per_epoch', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--reward_type', type=str, default='match', 
                        choices=['default', 'match', 'challenge', 'smooth', 'squared'])
    parser.add_argument('--eval_every', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Network: {args.hidden} hidden x {args.num_layers} layers")
    print(f"Reward type: {args.reward_type}")
    
    # Initialize networks
    policy = LargeSteerPolicy(hidden=args.hidden, num_layers=args.num_layers).to(device)
    value_net = LargeValueNetwork(hidden=args.hidden, num_layers=args.num_layers).to(device)
    
    # Count parameters
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())
    print(f"Policy params: {policy_params:,}")
    print(f"Value params: {value_params:,}")
    
    # Data paths
    data_dir = Path('data')
    all_files = sorted(data_dir.glob('*.csv'))
    train_files = all_files[:900]
    eval_files = all_files[900:950]
    
    # Estimate normalization
    input_mean, input_std = estimate_normalization(policy, train_files, device, args.hidden, args.num_layers)
    if input_mean is not None:
        policy.input_mean = input_mean.to(device)
        policy.input_std = input_std.to(device)
        value_net.input_mean = input_mean.to(device)
        value_net.input_std = input_std.to(device)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'models/ppo_large_{args.reward_type}_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log = []
    best_cost = float('inf')
    
    print(f"\nTrain files: {len(train_files)}, Eval files: {len(eval_files)}")
    print("\n" + "="*70)
    print("PPO TRAINING WITH LARGE NETWORK")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        episode_files = np.random.choice(train_files, size=args.episodes_per_epoch, replace=False)
        
        policy.eval()
        episodes = collect_episodes_parallel(
            episode_files, policy, device, args.n_workers, 
            args.reward_type, args.hidden, args.num_layers
        )
        
        if len(episodes) == 0:
            print(f"Epoch {epoch}: No episodes collected!")
            continue
        
        # Aggregate data
        all_states, all_actions, all_log_probs, all_rewards, all_values = [], [], [], [], []
        costs = [ep['total_cost'] for ep in episodes]
        lataccel_costs = [ep['lataccel_cost'] for ep in episodes]
        jerk_costs = [ep['jerk_cost'] for ep in episodes]
        
        # Discovery metrics
        mean_abs_errors = [ep['mean_abs_error'] for ep in episodes]
        mean_abs_jerks = [ep['mean_abs_jerk'] for ep in episodes]
        
        for ep in episodes:
            all_states.acan ppend(ep['states'])
            all_actions.append(ep['actions'])
            all_log_probs.append(ep['log_probs'])
            all_rewards.append(ep['rewards'])
            
            with torch.no_grad():
                states_t = torch.tensor(ep['states'], dtype=torch.float32, device=device)
                values = value_net(states_t).cpu().numpy()
            all_values.append(values)
        
        all_returns, all_advantages = [], []
        for rewards, values in zip(all_rewards, all_values):
            returns, advantages = compute_returns_and_advantages(rewards, values)
            all_returns.append(returns)
            all_advantages.append(advantages)
        
        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        log_probs = np.concatenate(all_log_probs)
        returns = np.concatenate(all_returns)
        advantages = np.concatenate(all_advantages)
        
        # PPO update
        policy.train()
        value_net.train()
        
        update_stats = ppo_update(
            policy, value_net, policy_optimizer, value_optimizer,
            states, actions, log_probs, returns, advantages,
            batch_size=512
        )
        
        epoch_time = time.time() - epoch_start
        
        mean_cost = np.mean(costs)
        mean_lataccel = np.mean(lataccel_costs)
        mean_jerk = np.mean(jerk_costs)
        policy_std = torch.exp(policy.log_std).item()
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Total Cost: {mean_cost:.1f} (lataccel={mean_lataccel:.2f}, jerk={mean_jerk:.2f})")
        print(f"  Mean |error|: {np.mean(mean_abs_errors):.3f}, Mean |jerk|: {np.mean(mean_abs_jerks):.3f}")
        print(f"  Policy std: {policy_std:.3f}")
        print(f"  PG loss: {update_stats['pg_loss']:.4f}, Value loss: {update_stats['value_loss']:.1f}")
        print(f"  Clip frac: {update_stats['clip_frac']:.3f}, KL: {update_stats['approx_kl']:.4f}")
        
        log_entry = {
            'epoch': epoch,
            'mean_cost': float(mean_cost),
            'lataccel_cost': float(mean_lataccel),
            'jerk_cost': float(mean_jerk),
            'mean_abs_error': float(np.mean(mean_abs_errors)),
            'mean_abs_jerk': float(np.mean(mean_abs_jerks)),
            'policy_std': float(policy_std),
            **{k: float(v) for k, v in update_stats.items()}
        }
        log.append(log_entry)
        
        if epoch % args.eval_every == 0:
            policy.eval()
            eval_episodes = collect_episodes_parallel(
                eval_files[:16], policy, device, args.n_workers,
                args.reward_type, args.hidden, args.num_layers
            )
            eval_costs = [ep['total_cost'] for ep in eval_episodes if ep is not None]
            
            if eval_costs:
                eval_mean = np.mean(eval_costs)
                print(f"  [EVAL] Cost: {eval_mean:.1f} ± {np.std(eval_costs):.1f}")
                
                if eval_mean < best_cost:
                    best_cost = eval_mean
                    torch.save({
                        'epoch': epoch,
                        'policy_state': policy.state_dict(),
                        'value_state': value_net.state_dict(),
                        'input_mean': policy.input_mean.cpu(),
                        'input_std': policy.input_std.cpu(),
                        'hidden': args.hidden,
                        'num_layers': args.num_layers,
                        'reward_type': args.reward_type,
                        'eval_cost': eval_mean,
                    }, out_dir / 'best_model.pt')
                    print(f"  [EVAL] New best!")
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'policy_state': policy.state_dict(),
                'value_state': value_net.state_dict(),
                'input_mean': policy.input_mean.cpu(),
                'input_std': policy.input_std.cpu(),
            }, out_dir / f'checkpoint_{epoch:03d}.pt')
    
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Training complete!")
    print(f"Best eval cost: {best_cost:.1f}")
    print(f"Logs saved to: {out_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

