#!/usr/bin/env python3
"""
PPO training with extensive debugging - start from random weights.

Goal: Get basic convergence before fine-tuning from imitation.
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


class SteerPolicy(nn.Module):
    """
    Simple MLP policy for debugging - easier to train than transformer.
    
    Input: flattened (past_ctx, current, future_ctx)
    Output: mean, log_std for Gaussian policy
    """
    def __init__(self, hidden=256):
        super().__init__()
        
        # Input: 10*6 (past) + 4 (current) + 10*4 (future) = 104
        input_dim = CONTEXT_LENGTH * 6 + 4 + LOOKAHEAD_LENGTH * 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor(-1.0))  # learnable log_std, init std=0.37
        
        # Input normalization
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
    def forward(self, x, deterministic=False):
        """
        x: (batch, input_dim) - flattened input
        Returns: (action, log_prob, entropy) or (action, None, None) if deterministic
        """
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        mean = torch.tanh(mean) * STEER_RANGE  # bound to [-2, 2]
        
        std = torch.exp(self.log_std.clamp(-3, 0))  # std in [0.05, 1.0]
        
        if deterministic:
            return mean, None, None
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = action.clamp(-STEER_RANGE, STEER_RANGE)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, x, action):
        """Evaluate log_prob of action given state"""
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        
        h = self.net(x)
        mean = self.mean_head(h).squeeze(-1)
        mean = torch.tanh(mean) * STEER_RANGE
        
        std = torch.exp(self.log_std.clamp(-3, 0))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Separate value network for critic"""
    def __init__(self, hidden=256):
        super().__init__()
        
        input_dim = CONTEXT_LENGTH * 6 + 4 + LOOKAHEAD_LENGTH * 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
    def forward(self, x):
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x).squeeze(-1)


def flatten_obs(past_ctx, current, future_ctx):
    """Flatten observations into single vector"""
    # past_ctx: (10, 6), current: (4,), future_ctx: (10, 4)
    return np.concatenate([past_ctx.flatten(), current, future_ctx.flatten()])


def run_episode(data_path, policy, device, deterministic=False):
    """
    Run single episode, collect trajectory.
    Returns: states, actions, rewards, log_probs, total_cost
    """
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    
    # Wrapper to collect data
    class PolicyWrapper:
        def __init__(self):
            self.history = []  # (v_ego, a_ego, roll, target, measured, steer)
            self.trajectory = []  # (state, action, log_prob)
            self.prev_steer = 0.0
            self.prev_lataccel = 0.0
            
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            roll = state.roll_lataccel
            obs = [state.v_ego, state.a_ego, roll, target_lataccel, current_lataccel, self.prev_steer]
            self.history.append(obs)
            
            # Need enough context
            if len(self.history) < CONTEXT_LENGTH:
                self.prev_steer = 0.0
                return 0.0
            
            # Check future plan
            if not future_plan or len(future_plan.lataccel) < LOOKAHEAD_LENGTH:
                return self.prev_steer
            
            # Build state
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
            
            # Compute reward (negative cost at this step)
            lataccel_err = abs(current_lataccel - target_lataccel)
            jerk = abs(current_lataccel - self.prev_lataccel) / 0.1  # dt=0.1s
            step_cost = 5.0 * lataccel_err + jerk
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
    except Exception as e:
        total_cost = 10000.0
    
    if len(wrapper.trajectory) == 0:
        return None
    
    # Unpack trajectory
    states = np.array([t[0] for t in wrapper.trajectory], dtype=np.float32)
    actions = np.array([t[1] for t in wrapper.trajectory], dtype=np.float32)
    log_probs = np.array([t[2] for t in wrapper.trajectory], dtype=np.float32)
    rewards = np.array([t[3] for t in wrapper.trajectory], dtype=np.float32)
    
    return {
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'rewards': rewards,
        'total_cost': total_cost
    }


def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return returns, advantages


def _episode_worker(args):
    """Worker for parallel episode collection"""
    data_path, policy_state, device_str = args
    
    # Recreate policy in worker
    policy = SteerPolicy()
    policy.load_state_dict(policy_state)
    policy.eval()
    
    device = torch.device(device_str)
    policy = policy.to(device)
    
    return run_episode(data_path, policy, device, deterministic=False)


def collect_episodes_parallel(data_paths, policy, device, n_workers=8):
    """Collect episodes in parallel"""
    policy_state = policy.state_dict()
    device_str = str(device)
    
    args_list = [(p, policy_state, device_str) for p in data_paths]
    
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
               clip_eps=0.2, n_epochs=4, batch_size=256):
    """PPO update with debugging info"""
    
    device = next(policy.parameters()).device
    
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    stats = {
        'pg_loss': [],
        'value_loss': [],
        'entropy': [],
        'clip_frac': [],
        'approx_kl': [],
    }
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            # Policy update
            new_log_probs, entropy = policy.evaluate(states[batch_idx], actions[batch_idx])
            
            ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
            
            # Clipped objective
            surr1 = ratio * advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[batch_idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            
            entropy_loss = -0.01 * entropy.mean()
            
            policy_loss = pg_loss + entropy_loss
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optimizer.step()
            
            # Value update
            values = value_net(states[batch_idx])
            value_loss = F.mse_loss(values, returns[batch_idx])
            
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()
            
            # Stats
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
                approx_kl = (old_log_probs[batch_idx] - new_log_probs).mean()
                
            stats['pg_loss'].append(pg_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.mean().item())
            stats['clip_frac'].append(clip_frac.item())
            stats['approx_kl'].append(approx_kl.item())
    
    return {k: np.mean(v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes_per_epoch', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize policy and value network (RANDOM WEIGHTS)
    policy = SteerPolicy(hidden=args.hidden).to(device)
    value_net = ValueNetwork(hidden=args.hidden).to(device)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    
    # Data paths
    data_dir = Path('data')
    all_files = sorted(data_dir.glob('*.csv'))
    train_files = all_files[:900]  # First 900 for training
    eval_files = all_files[900:950]  # 50 for eval
    
    print(f"Train files: {len(train_files)}, Eval files: {len(eval_files)}")
    
    # Output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'models/ppo_debug_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log = []
    best_cost = float('inf')
    
    print("\n" + "="*70)
    print("PPO TRAINING FROM RANDOM WEIGHTS")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Sample training files
        episode_files = np.random.choice(train_files, size=args.episodes_per_epoch, replace=False)
        
        # Collect episodes
        policy.eval()
        episodes = collect_episodes_parallel(episode_files, policy, device, args.n_workers)
        
        if len(episodes) == 0:
            print(f"Epoch {epoch}: No episodes collected!")
            continue
        
        # Aggregate data
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        
        costs = [ep['total_cost'] for ep in episodes]
        
        for ep in episodes:
            all_states.append(ep['states'])
            all_actions.append(ep['actions'])
            all_log_probs.append(ep['log_probs'])
            all_rewards.append(ep['rewards'])
            
            # Compute values for this episode
            with torch.no_grad():
                states_t = torch.tensor(ep['states'], dtype=torch.float32, device=device)
                values = value_net(states_t).cpu().numpy()
            all_values.append(values)
        
        # Compute returns and advantages for each episode
        all_returns = []
        all_advantages = []
        
        for rewards, values in zip(all_rewards, all_values):
            returns, advantages = compute_returns_and_advantages(rewards, values)
            all_returns.append(returns)
            all_advantages.append(advantages)
        
        # Concatenate all
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
            states, actions, log_probs, returns, advantages
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        mean_reward = np.mean([r.mean() for r in all_rewards])
        policy_std = torch.exp(policy.log_std).item()
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Cost: {mean_cost:.1f} ± {std_cost:.1f} (n={len(episodes)})")
        print(f"  Reward/step: {mean_reward:.3f}")
        print(f"  Policy std: {policy_std:.3f}")
        print(f"  PG loss: {update_stats['pg_loss']:.4f}")
        print(f"  Value loss: {update_stats['value_loss']:.4f}")
        print(f"  Entropy: {update_stats['entropy']:.4f}")
        print(f"  Clip frac: {update_stats['clip_frac']:.3f}")
        print(f"  Approx KL: {update_stats['approx_kl']:.4f}")
        
        log_entry = {
            'epoch': epoch,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'mean_reward': mean_reward,
            'policy_std': policy_std,
            **update_stats
        }
        log.append(log_entry)
        
        # Periodic evaluation
        if epoch % args.eval_every == 0:
            policy.eval()
            eval_episodes = collect_episodes_parallel(eval_files[:16], policy, device, args.n_workers)
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
                        'eval_cost': eval_mean,
                    }, out_dir / 'best_model.pt')
                    print(f"  [EVAL] New best! Saved to {out_dir / 'best_model.pt'}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'policy_state': policy.state_dict(),
                'value_state': value_net.state_dict(),
            }, out_dir / f'checkpoint_{epoch:03d}.pt')
    
    # Save final log (convert numpy types to python types)
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(convert_to_serializable(log), f, indent=2)
    
    print("\n" + "="*70)
    print(f"Training complete!")
    print(f"Best eval cost: {best_cost:.1f}")
    print(f"Logs saved to: {out_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

