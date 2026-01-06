#!/usr/bin/env python3
"""
PPO with Curriculum Learning - start easy, get harder.

Key insight from Canal & Taschin paper:
"Results show a high degree of environment generalization achieved by 
training on randomized maps of increasing difficulty (Curriculum Learning)"

We sort segments by difficulty (variance of target lataccel) and 
train easy→hard.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import time

from train_steer_model_rl import SteerModelRL, CONTEXT_LENGTH, LOOKAHEAD_LENGTH

STEER_MIN, STEER_MAX = -2.0, 2.0


class ValueNetwork(nn.Module):
    """Value network matching policy architecture"""
    def __init__(self, d_model=128):
        super().__init__()
        
        self.past_proj = nn.Linear(6, d_model)
        self.current_proj = nn.Linear(4, d_model)
        self.future_proj = nn.Linear(4, d_model)
        
        self.pos_emb = nn.Parameter(torch.randn(21, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Same normalization as policy
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
        return self.head(out[:, 10]).squeeze(-1)


def compute_segment_difficulty(csv_path):
    """
    Compute difficulty score for a segment.
    Higher = harder (more variation in target lataccel)
    """
    try:
        df = pd.read_csv(csv_path)
        target = df['targetLateralAcceleration'].values
        
        # Difficulty metrics:
        # 1. Variance of target (more change = harder)
        variance = np.var(target)
        
        # 2. Max absolute target (extreme maneuvers = harder)
        max_abs = np.max(np.abs(target))
        
        # 3. Rate of change (quick changes = harder)
        rate_of_change = np.mean(np.abs(np.diff(target)))
        
        # Combined score
        difficulty = variance + 0.5 * max_abs + 2.0 * rate_of_change
        return difficulty
    except:
        return float('inf')  # Put broken files at end


def run_episode(data_path, policy, device, exploration_std=0.1):
    """Run episode, collect trajectory for PPO"""
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    
    class Wrapper:
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
                self.prev_steer = float(np.clip(steer, STEER_MIN, STEER_MAX))
                self.prev_lataccel = current_lataccel
                return self.prev_steer
            
            if not future_plan or not hasattr(future_plan, 'lataccel') or len(future_plan.lataccel) < LOOKAHEAD_LENGTH:
                self.prev_lataccel = current_lataccel
                return self.prev_steer
            
            past_ctx = np.array(self.history[-CONTEXT_LENGTH:], dtype=np.float32)
            current = np.array([state.v_ego, state.a_ego, roll, current_lataccel], dtype=np.float32)
            future_ctx = np.stack([
                np.array(future_plan.v_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.a_ego[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.roll_lataccel[:LOOKAHEAD_LENGTH]),
                np.array(future_plan.lataccel[:LOOKAHEAD_LENGTH]),
            ], axis=1).astype(np.float32)
            
            past_t = torch.tensor(past_ctx, device=device).unsqueeze(0)
            cur_t = torch.tensor(current, device=device).unsqueeze(0)
            fut_t = torch.tensor(future_ctx, device=device).unsqueeze(0)
            
            with torch.no_grad():
                mean, _, _ = policy(past_t, cur_t, fut_t, deterministic=True)
                mean = mean.item()
            
            # Exploration
            noise = np.random.randn() * exploration_std
            action = float(np.clip(mean + noise, STEER_MIN, STEER_MAX))
            
            # Log prob for PPO
            log_prob = -0.5 * ((action - mean) / exploration_std) ** 2
            
            # Reward matching challenge cost
            err = current_lataccel - target_lataccel
            jerk = (current_lataccel - self.prev_lataccel) / 0.1
            reward = -(5.0 * err**2 + 0.1 * jerk**2)
            
            self.trajectory.append({
                'past_ctx': past_ctx, 'current': current, 'future_ctx': future_ctx,
                'action': action, 'log_prob': log_prob, 'reward': reward, 'mean': mean
            })
            
            self.prev_steer = action
            self.prev_lataccel = current_lataccel
            return action
    
    wrapper = Wrapper()
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    try:
        sim.rollout()
        cost = sim.compute_cost()['total_cost']
    except:
        cost = 10000.0
    
    return wrapper.trajectory, cost


def _worker(args):
    data_path, policy_state, device_str, exploration_std = args
    policy = SteerModelRL(d_model=128)
    policy.load_state_dict(policy_state)
    policy.eval()
    return run_episode(data_path, policy, torch.device(device_str), exploration_std)


def collect_parallel(files, policy, device, n_workers, exploration_std):
    policy_state = policy.state_dict()
    args = [(f, policy_state, str(device), exploration_std) for f in files]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for traj, cost in ex.map(_worker, args):
            if traj:
                results.append({'trajectory': traj, 'cost': cost})
    return results


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0 if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    
    return returns, advantages


def ppo_update(policy, value_net, policy_opt, value_opt, 
               episodes, device, clip_eps=0.1, exploration_std=0.1):
    """
    PPO update with proper gradient computation.
    
    The gradient is: ∇_θ L = E[ ∇_θ log π(a|s) * A(s,a) ]
    
    Where:
    - log π(a|s) = log probability of action a in state s
    - A(s,a) = advantage = how much better than expected
    
    PPO clips this to prevent too-large updates.
    """
    
    # Collect all data
    all_past, all_cur, all_fut = [], [], []
    all_actions, all_old_lp, all_returns, all_advs = [], [], [], []
    
    for ep in episodes:
        traj = ep['trajectory']
        rewards = np.array([t['reward'] for t in traj])
        
        past = torch.tensor(np.array([t['past_ctx'] for t in traj]), device=device)
        cur = torch.tensor(np.array([t['current'] for t in traj]), device=device)
        fut = torch.tensor(np.array([t['future_ctx'] for t in traj]), device=device)
        
        with torch.no_grad():
            values = value_net(past, cur, fut).cpu().numpy()
        
        returns, advs = compute_gae(rewards, values)
        
        all_past.append(past.cpu())
        all_cur.append(cur.cpu())
        all_fut.append(fut.cpu())
        all_actions.append(torch.tensor([t['action'] for t in traj]))
        all_old_lp.append(torch.tensor([t['log_prob'] for t in traj]))
        all_returns.append(torch.tensor(returns))
        all_advs.append(torch.tensor(advs))
    
    # Concatenate
    past = torch.cat(all_past).to(device)
    cur = torch.cat(all_cur).to(device)
    fut = torch.cat(all_fut).to(device)
    actions = torch.cat(all_actions).to(device)
    old_lp = torch.cat(all_old_lp).to(device)
    returns = torch.cat(all_returns).to(device)
    advs = torch.cat(all_advs).to(device)
    
    # Normalize advantages (crucial for stability)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    
    # PPO epochs
    policy.train()
    value_net.train()
    
    n = len(past)
    batch_size = 256
    stats = {'pg_loss': [], 'value_loss': [], 'kl': []}
    
    for _ in range(4):  # Multiple passes over data
        idx = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]
            
            # Forward pass through policy
            mean, _, _ = policy(past[b], cur[b], fut[b], deterministic=True)
            
            # Compute new log probability
            # log π(a|s) = -0.5 * ((a - μ) / σ)² - log(σ√2π)
            new_lp = -0.5 * ((actions[b] - mean) / exploration_std) ** 2
            
            # Importance sampling ratio
            ratio = torch.exp(new_lp - old_lp[b])
            
            # PPO clipped objective
            # L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
            surr1 = ratio * advs[b]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs[b]
            pg_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus (encourage exploration)
            entropy = 0.5 * np.log(2 * np.pi * exploration_std**2) + 0.5
            policy_loss = pg_loss - 0.01 * entropy
            
            # Backprop through policy
            policy_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()
            
            # Value function update
            values = value_net(past[b], cur[b], fut[b])
            value_loss = F.mse_loss(values, returns[b])
            
            value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            value_opt.step()
            
            # Stats
            with torch.no_grad():
                kl = (old_lp[b] - new_lp).mean()
            
            stats['pg_loss'].append(pg_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['kl'].append(kl.item())
    
    return {k: np.mean(v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imitation_model', type=str, 
                        default='models/teacher_ckpts_20251225_173549/imitation_epoch_003.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--policy_lr', type=float, default=1e-5)
    parser.add_argument('--value_lr', type=float, default=3e-4)
    parser.add_argument('--exploration_std', type=float, default=0.1)
    parser.add_argument('--curriculum_phases', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load imitation model
    print(f"\nLoading: {args.imitation_model}")
    ckpt = torch.load(args.imitation_model, map_location='cpu', weights_only=False)
    policy = SteerModelRL(d_model=128)
    policy.load_state_dict(ckpt['model_state'])
    policy = policy.to(device)
    
    value_net = ValueNetwork(d_model=128).to(device)
    
    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)
    
    # Load and sort files by difficulty
    data_dir = Path('data')
    all_files = sorted(data_dir.glob('*.csv'))[:900]
    
    print("\nComputing segment difficulties...")
    difficulties = [(f, compute_segment_difficulty(f)) for f in all_files]
    sorted_files = [f for f, d in sorted(difficulties, key=lambda x: x[1])]
    
    # Split into curriculum phases
    n_per_phase = len(sorted_files) // args.curriculum_phases
    phases = [sorted_files[i*n_per_phase:(i+1)*n_per_phase] for i in range(args.curriculum_phases)]
    
    print(f"Curriculum: {args.curriculum_phases} phases, {n_per_phase} segments each")
    print(f"Easiest segment difficulty: {difficulties[0][1]:.3f}")
    print(f"Hardest segment difficulty: {difficulties[-1][1]:.3f}")
    
    # Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'models/ppo_curriculum_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial eval
    eval_files = sorted(data_dir.glob('*.csv'))[900:920]
    policy.eval()
    eval_eps = collect_parallel(eval_files, policy, device, args.n_workers, 0.0)
    initial_cost = np.mean([e['cost'] for e in eval_eps]) if eval_eps else float('inf')
    print(f"\nInitial eval cost: {initial_cost:.1f}")
    
    log = []
    best_cost = initial_cost
    epochs_per_phase = args.epochs // args.curriculum_phases
    
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: Easy → Hard")
    print("="*60)
    
    for phase in range(args.curriculum_phases):
        phase_files = phases[phase]
        
        # Gradually include previous phases too
        available_files = []
        for p in range(phase + 1):
            available_files.extend(phases[p])
        
        print(f"\n--- Phase {phase+1}/{args.curriculum_phases} ---")
        print(f"Available segments: {len(available_files)} (easiest {100*(phase+1)//args.curriculum_phases}%)")
        
        for epoch in range(1, epochs_per_phase + 1):
            global_epoch = phase * epochs_per_phase + epoch
            
            # Sample from available (easier) segments
            episode_files = np.random.choice(available_files, 
                                            size=min(args.episodes_per_epoch, len(available_files)), 
                                            replace=False)
            
            policy.eval()
            episodes = collect_parallel(episode_files, policy, device, 
                                       args.n_workers, args.exploration_std)
            
            if not episodes:
                continue
            
            costs = [e['cost'] for e in episodes]
            
            # PPO update
            stats = ppo_update(policy, value_net, policy_opt, value_opt,
                              episodes, device, exploration_std=args.exploration_std)
            
            print(f"  Epoch {global_epoch}: cost={np.mean(costs):.1f}±{np.std(costs):.1f}, "
                  f"pg={stats['pg_loss']:.4f}, kl={stats['kl']:.4f}")
            
            log.append({
                'epoch': global_epoch, 'phase': phase + 1,
                'cost': float(np.mean(costs)), **{k: float(v) for k, v in stats.items()}
            })
            
            # Eval every 5 epochs
            if global_epoch % 5 == 0:
                policy.eval()
                eval_eps = collect_parallel(eval_files, policy, device, args.n_workers, 0.0)
                if eval_eps:
                    eval_cost = np.mean([e['cost'] for e in eval_eps])
                    print(f"  [EVAL] Cost: {eval_cost:.1f}")
                    
                    if eval_cost < best_cost:
                        best_cost = eval_cost
                        torch.save({
                            'epoch': global_epoch,
                            'policy_state': policy.state_dict(),
                            'value_state': value_net.state_dict(),
                            'eval_cost': eval_cost,
                        }, out_dir / 'best_model.pt')
                        print(f"  [EVAL] New best!")
    
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Initial: {initial_cost:.1f} → Best: {best_cost:.1f}")
    print(f"Improvement: {(initial_cost - best_cost) / initial_cost * 100:.1f}%")
    print("="*60)


if __name__ == '__main__':
    main()

