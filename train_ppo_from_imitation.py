#!/usr/bin/env python3
"""
PPO fine-tuning from PID imitation with proper value function pre-training.

Key insight: PID imitation achieves ~117 cost. RL from scratch gets ~1000+.
The model CAN learn, the RL algorithm is the problem.

Fix:
1. Load PID imitation model (cost ~117)
2. Pre-train value function FIRST (freeze policy)
3. Very conservative PPO updates (tiny LR, tight clipping)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import time

# Import model architecture from teacher script
from train_steer_model_rl import SteerModelRL, CONTEXT_LENGTH, LOOKAHEAD_LENGTH

STEER_MIN, STEER_MAX = -2.0, 2.0


class ValueNetwork(nn.Module):
    """Value network matching SteerModelRL architecture"""
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.past_proj = nn.Linear(6, d_model)
        self.current_proj = nn.Linear(4, d_model)
        self.future_proj = nn.Linear(4, d_model)
        
        self.pos_emb = nn.Parameter(torch.randn(21, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.register_buffer('past_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('past_std', torch.tensor([15., 1., 0.5, 1., 1., 0.5]))
        self.register_buffer('current_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('current_std', torch.tensor([15., 1., 0.5, 1.]))
        self.register_buffer('future_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('future_std', torch.tensor([15., 1., 0.5, 1.]))
    
    def forward(self, past_ctx, current, future_ctx):
        past_ctx = (past_ctx - self.past_mean) / self.past_std
        current = (current - self.current_mean) / self.current_std
        future_ctx = (future_ctx - self.future_mean) / self.future_std
        
        past_emb = self.past_proj(past_ctx)
        current_emb = self.current_proj(current).unsqueeze(1)
        future_emb = self.future_proj(future_ctx)
        
        seq = torch.cat([past_emb, current_emb, future_emb], dim=1)
        seq = seq + self.pos_emb
        
        out = self.transformer(seq)
        current_out = out[:, 10]
        
        return self.head(current_out).squeeze(-1)


def run_episode(data_path, policy, device, deterministic=False, exploration_std=0.1):
    """Run episode with policy, collect trajectory."""
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    
    class PolicyWrapper:
        def __init__(self):
            self.history = []
            self.trajectory = []
            self.prev_steer = 0.0
            self.prev_lataccel = 0.0
            
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            roll = state.roll_lataccel
            obs = [state.v_ego, state.a_ego, roll, target_lataccel, current_lataccel, self.prev_steer]
            self.history.append(obs)
            
            if len(self.history) < CONTEXT_LENGTH:
                steer = 0.3 * (target_lataccel - current_lataccel)
                self.prev_steer = np.clip(steer, STEER_MIN, STEER_MAX)
                self.prev_lataccel = current_lataccel
                return float(self.prev_steer)
            
            if not future_plan or not hasattr(future_plan, 'lataccel') or len(future_plan.lataccel) < LOOKAHEAD_LENGTH:
                self.prev_lataccel = current_lataccel
                return float(self.prev_steer)
            
            past_ctx = np.array(self.history[-CONTEXT_LENGTH:], dtype=np.float32)
            current = np.array([state.v_ego, state.a_ego, roll, current_lataccel], dtype=np.float32)
            future_ctx = np.stack([
                np.array(future_plan.v_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.a_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.roll_lataccel[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.lataccel[:LOOKAHEAD_LENGTH]),
            ], axis=1).astype(np.float32)
            
            past_t = torch.tensor(past_ctx, dtype=torch.float32, device=device).unsqueeze(0)
            cur_t = torch.tensor(current, dtype=torch.float32, device=device).unsqueeze(0)
            fut_t = torch.tensor(future_ctx, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                mean, std, _ = policy(past_t, cur_t, fut_t, deterministic=True)
                mean = mean.item()
                
                if deterministic:
                    action = mean
                    log_prob = 0.0
                else:
                    # Add exploration noise
                    noise = np.random.randn() * exploration_std
                    action = mean + noise
                    action = np.clip(action, STEER_MIN, STEER_MAX)
                    # Compute log prob
                    log_prob = -0.5 * ((action - mean) / exploration_std) ** 2 - np.log(exploration_std * np.sqrt(2 * np.pi))
            
            # Reward: match challenge cost (scaled)
            lataccel_err = current_lataccel - target_lataccel
            jerk = (current_lataccel - self.prev_lataccel) / 0.1
            # Challenge: 5000*err² + 100*jerk², we scale by 1/1000
            step_cost = 5.0 * (lataccel_err ** 2) + 0.1 * (jerk ** 2)
            reward = -step_cost
            
            self.trajectory.append({
                'past_ctx': past_ctx,
                'current': current,
                'future_ctx': future_ctx,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'mean': mean,
            })
            
            self.prev_steer = float(action)
            self.prev_lataccel = current_lataccel
            
            return float(action)
    
    wrapper = PolicyWrapper()
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    try:
        sim.rollout()
        cost = sim.compute_cost()
        total_cost = cost['total_cost']
    except Exception as e:
        total_cost = 10000.0
    
    return wrapper.trajectory, total_cost


def _episode_worker(args):
    data_path, policy_state, device_str, deterministic, exploration_std = args
    
    policy = SteerModelRL(d_model=128)
    policy.load_state_dict(policy_state)
    policy.eval()
    
    device = torch.device(device_str)
    policy = policy.to(device)
    
    return run_episode(data_path, policy, device, deterministic, exploration_std)


def collect_episodes_parallel(data_paths, policy, device, n_workers=8, deterministic=False, exploration_std=0.1):
    policy_state = policy.state_dict()
    device_str = str(device)
    
    args_list = [(p, policy_state, device_str, deterministic, exploration_std) for p in data_paths]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_episode_worker, args) for args in args_list]
        for future in futures:
            try:
                traj, cost = future.result()
                if traj:
                    results.append({'trajectory': traj, 'cost': cost})
            except Exception as e:
                pass
    
    return results


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imitation_model', type=str, 
                        default='models/teacher_ckpts_20251225_173549/imitation_epoch_003.pt')
    parser.add_argument('--value_pretrain_epochs', type=int, default=5)
    parser.add_argument('--ppo_epochs', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--policy_lr', type=float, default=1e-5, help='Very small for conservative updates')
    parser.add_argument('--value_lr', type=float, default=3e-4)
    parser.add_argument('--clip_eps', type=float, default=0.1, help='Tight clipping')
    parser.add_argument('--exploration_std', type=float, default=0.15)
    parser.add_argument('--eval_every', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load PID imitation model
    print(f"\nLoading imitation model: {args.imitation_model}")
    checkpoint = torch.load(args.imitation_model, map_location='cpu', weights_only=False)
    
    policy = SteerModelRL(d_model=128)
    policy.load_state_dict(checkpoint['model_state'])
    policy = policy.to(device)
    
    # Create value network
    value_net = ValueNetwork(d_model=128)
    value_net = value_net.to(device)
    
    # Data
    data_dir = Path('data')
    all_files = sorted(data_dir.glob('*.csv'))
    train_files = all_files[:900]
    eval_files = all_files[900:950]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'models/ppo_imitation_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial evaluation
    print("\n" + "="*60)
    print("INITIAL EVALUATION (before any training)")
    print("="*60)
    
    policy.eval()
    eval_episodes = collect_episodes_parallel(eval_files[:16], policy, device, args.n_workers, deterministic=True)
    eval_costs = [ep['cost'] for ep in eval_episodes]
    initial_cost = np.mean(eval_costs)
    print(f"Initial cost: {initial_cost:.1f} ± {np.std(eval_costs):.1f}")
    
    # ================================================================
    # PHASE 1: Pre-train value function (freeze policy)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Pre-training value function (policy frozen)")
    print("="*60)
    
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)
    
    for epoch in range(1, args.value_pretrain_epochs + 1):
        episode_files = np.random.choice(train_files, size=args.episodes_per_epoch, replace=False)
        
        policy.eval()
        episodes = collect_episodes_parallel(
            episode_files, policy, device, args.n_workers, 
            deterministic=False, exploration_std=args.exploration_std
        )
        
        if not episodes:
            continue
        
        # Collect all data
        all_past, all_cur, all_fut, all_returns = [], [], [], []
        costs = [ep['cost'] for ep in episodes]
        
        for ep in episodes:
            traj = ep['trajectory']
            rewards = np.array([t['reward'] for t in traj])
            
            # Get values
            past = torch.tensor(np.array([t['past_ctx'] for t in traj]), device=device)
            cur = torch.tensor(np.array([t['current'] for t in traj]), device=device)
            fut = torch.tensor(np.array([t['future_ctx'] for t in traj]), device=device)
            
            with torch.no_grad():
                values = value_net(past, cur, fut).cpu().numpy()
            
            returns, _ = compute_returns_and_advantages(rewards, values)
            
            all_past.append(past.cpu())
            all_cur.append(cur.cpu())
            all_fut.append(fut.cpu())
            all_returns.append(torch.tensor(returns))
        
        # Train value function
        past = torch.cat(all_past).to(device)
        cur = torch.cat(all_cur).to(device)
        fut = torch.cat(all_fut).to(device)
        returns = torch.cat(all_returns).to(device)
        
        value_net.train()
        for _ in range(4):  # Multiple passes
            indices = torch.randperm(len(past))
            for start in range(0, len(past), 256):
                end = min(start + 256, len(past))
                idx = indices[start:end]
                
                values = value_net(past[idx], cur[idx], fut[idx])
                loss = F.mse_loss(values, returns[idx])
                
                value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                value_optimizer.step()
        
        print(f"Value pretrain epoch {epoch}: cost={np.mean(costs):.1f}, value_loss={loss.item():.1f}")
    
    # ================================================================
    # PHASE 2: Conservative PPO fine-tuning
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Conservative PPO fine-tuning")
    print("="*60)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    
    log = []
    best_cost = initial_cost
    
    for epoch in range(1, args.ppo_epochs + 1):
        epoch_start = time.time()
        
        episode_files = np.random.choice(train_files, size=args.episodes_per_epoch, replace=False)
        
        policy.eval()
        episodes = collect_episodes_parallel(
            episode_files, policy, device, args.n_workers,
            deterministic=False, exploration_std=args.exploration_std
        )
        
        if not episodes:
            continue
        
        costs = [ep['cost'] for ep in episodes]
        
        # Collect all data
        all_past, all_cur, all_fut = [], [], []
        all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], []
        
        for ep in episodes:
            traj = ep['trajectory']
            rewards = np.array([t['reward'] for t in traj])
            
            past = torch.tensor(np.array([t['past_ctx'] for t in traj]), device=device)
            cur = torch.tensor(np.array([t['current'] for t in traj]), device=device)
            fut = torch.tensor(np.array([t['future_ctx'] for t in traj]), device=device)
            
            with torch.no_grad():
                values = value_net(past, cur, fut).cpu().numpy()
            
            returns, advantages = compute_returns_and_advantages(rewards, values)
            
            all_past.append(past.cpu())
            all_cur.append(cur.cpu())
            all_fut.append(fut.cpu())
            all_actions.append(torch.tensor([t['action'] for t in traj]))
            all_old_log_probs.append(torch.tensor([t['log_prob'] for t in traj]))
            all_advantages.append(torch.tensor(advantages))
            all_returns.append(torch.tensor(returns))
        
        # Concatenate
        past = torch.cat(all_past).to(device)
        cur = torch.cat(all_cur).to(device)
        fut = torch.cat(all_fut).to(device)
        actions = torch.cat(all_actions).to(device)
        old_log_probs = torch.cat(all_old_log_probs).to(device)
        advantages = torch.cat(all_advantages).to(device)
        returns = torch.cat(all_returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy.train()
        value_net.train()
        
        pg_losses, value_losses, kls = [], [], []
        
        for _ in range(4):
            indices = torch.randperm(len(past))
            
            for start in range(0, len(past), 256):
                end = min(start + 256, len(past))
                idx = indices[start:end]
                
                # Policy update
                mean, std, _ = policy(past[idx], cur[idx], fut[idx], deterministic=True)
                
                # Compute new log prob
                dist = torch.distributions.Normal(mean, args.exploration_std)
                new_log_probs = dist.log_prob(actions[idx])
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                policy_loss = pg_loss - 0.01 * entropy
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_optimizer.step()
                
                # Value update
                values = value_net(past[idx], cur[idx], fut[idx])
                value_loss = F.mse_loss(values, returns[idx])
                
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                value_optimizer.step()
                
                with torch.no_grad():
                    kl = (old_log_probs[idx] - new_log_probs).mean()
                
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                kls.append(kl.item())
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{args.ppo_epochs} ({epoch_time:.1f}s)")
        print(f"  Cost: {np.mean(costs):.1f} ± {np.std(costs):.1f}")
        print(f"  PG loss: {np.mean(pg_losses):.4f}, Value loss: {np.mean(value_losses):.1f}")
        print(f"  KL: {np.mean(kls):.4f}")
        
        log.append({
            'epoch': epoch,
            'mean_cost': float(np.mean(costs)),
            'pg_loss': float(np.mean(pg_losses)),
            'value_loss': float(np.mean(value_losses)),
            'kl': float(np.mean(kls)),
        })
        
        # Evaluation
        if epoch % args.eval_every == 0:
            policy.eval()
            eval_episodes = collect_episodes_parallel(eval_files[:16], policy, device, args.n_workers, deterministic=True)
            eval_costs = [ep['cost'] for ep in eval_episodes]
            
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
                    print(f"  [EVAL] New best!")
    
    # Save log
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Initial cost: {initial_cost:.1f}")
    print(f"Best eval cost: {best_cost:.1f}")
    print(f"Improvement: {(initial_cost - best_cost) / initial_cost * 100:.1f}%")
    print("="*60)


if __name__ == '__main__':
    main()

