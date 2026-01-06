#!/usr/bin/env python3
"""
Proper PPO implementation following MATLAB/OpenAI spec.

Key features:
1. Actor outputs BOTH mean AND std (learned exploration)
2. Critic estimates V(S)
3. GAE for advantage estimation
4. Clipped surrogate objective
5. Entropy loss for exploration
6. Curriculum learning (easy → hard)

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json

CONTEXT_LENGTH = 10
LOOKAHEAD_LENGTH = 10
STEER_MIN, STEER_MAX = -2.0, 2.0


class GaussianActor(nn.Module):
    """
    Gaussian policy actor that outputs BOTH mean and std.
    
    For continuous action spaces, the actor outputs:
    - μ(s): mean of Gaussian
    - σ(s): standard deviation of Gaussian
    
    Action is sampled as: a ~ N(μ(s), σ(s)²)
    """
    def __init__(self, d_model=128):
        super().__init__()
        
        # Input projections
        self.past_proj = nn.Linear(6, d_model)
        self.current_proj = nn.Linear(4, d_model)
        self.future_proj = nn.Linear(4, d_model)
        
        self.pos_emb = nn.Parameter(torch.randn(21, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output heads for mean and log_std
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Log std head - learned per-state std
        self.log_std_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Normalization buffers
        self.register_buffer('past_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('past_std', torch.tensor([15., 1., 0.5, 1., 1., 0.5]))
        self.register_buffer('current_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('current_std', torch.tensor([15., 1., 0.5, 1.]))
        self.register_buffer('future_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('future_std', torch.tensor([15., 1., 0.5, 1.]))
    
    def forward(self, past_ctx, current, future_ctx):
        """
        Returns: (mean, std) for Gaussian policy
        """
        # Normalize inputs
        past_ctx = (past_ctx - self.past_mean) / self.past_std
        current = (current - self.current_mean) / self.current_std
        future_ctx = (future_ctx - self.future_mean) / self.future_std
        
        # Project
        past_emb = self.past_proj(past_ctx)
        current_emb = self.current_proj(current).unsqueeze(1)
        future_emb = self.future_proj(future_ctx)
        
        # Transformer
        seq = torch.cat([past_emb, current_emb, future_emb], dim=1)
        seq = seq + self.pos_emb
        out = self.transformer(seq)
        
        # Get output from current position
        h = out[:, 10]
        
        # Mean bounded to action range
        mean = torch.tanh(self.mean_head(h).squeeze(-1)) * 2.0
        
        # Std bounded between 0.05 and 0.5
        log_std = self.log_std_head(h).squeeze(-1)
        std = torch.exp(log_std.clamp(-3, -0.7))  # exp(-3)≈0.05, exp(-0.7)≈0.5
        
        return mean, std
    
    def get_action(self, past_ctx, current, future_ctx, deterministic=False):
        """
        Sample action from policy.
        
        Returns: (action, log_prob, entropy)
        """
        mean, std = self.forward(past_ctx, current, future_ctx)
        
        if deterministic:
            return mean, None, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(STEER_MIN, STEER_MAX)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_action(self, past_ctx, current, future_ctx, action):
        """
        Evaluate log_prob and entropy for given action.
        Used in PPO update.
        """
        mean, std = self.forward(past_ctx, current, future_ctx)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class ValueCritic(nn.Module):
    """Value function critic V(S)"""
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


class PPOAgent:
    """
    PPO Agent following Schulman et al. (2017)
    
    Algorithm:
    1. Collect trajectories with current policy
    2. Compute advantages using GAE
    3. Update policy with clipped surrogate objective
    4. Update value function
    """
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4,
                 clip_eps=0.2, entropy_weight=0.01, gamma=0.99, gae_lambda=0.95):
        self.actor = actor
        self.critic = critic
        
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        
        self.clip_eps = clip_eps
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_gae(self, rewards, values, dones):
        """
        Generalized Advantage Estimation (GAE)
        
        δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        Aₜ = Σₖ (γλ)^k δₜ₊ₖ
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1 or dones[t]:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return returns, advantages
    
    def update(self, experiences, device, num_epochs=4, mini_batch_size=64):
        """
        PPO update step.
        
        L_actor = -min(r*A, clip(r, 1-ε, 1+ε)*A) - w*H
        L_critic = (V - G)² / 2
        
        Where:
        - r = π_new(a|s) / π_old(a|s)  (importance ratio)
        - A = advantage (normalized)
        - H = entropy
        - G = return
        """
        # Unpack experiences
        past = torch.tensor(np.array([e['past_ctx'] for e in experiences]), device=device)
        cur = torch.tensor(np.array([e['current'] for e in experiences]), device=device)
        fut = torch.tensor(np.array([e['future_ctx'] for e in experiences]), device=device)
        actions = torch.tensor(np.array([e['action'] for e in experiences]), device=device)
        old_log_probs = torch.tensor(np.array([e['log_prob'] for e in experiences]), device=device)
        returns = torch.tensor(np.array([e['return'] for e in experiences]), device=device)
        advantages = torch.tensor(np.array([e['advantage'] for e in experiences]), device=device)
        
        # Normalize advantages (crucial for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n = len(experiences)
        stats = {'actor_loss': [], 'critic_loss': [], 'entropy': [], 'kl': [], 'clip_frac': []}
        
        for _ in range(num_epochs):
            indices = torch.randperm(n)
            
            for start in range(0, n, mini_batch_size):
                end = min(start + mini_batch_size, n)
                idx = indices[start:end]
                
                # Get new log probs and entropy
                new_log_probs, entropy = self.actor.evaluate_action(
                    past[idx], cur[idx], fut[idx], actions[idx]
                )
                
                # Importance ratio
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # Clipped surrogate objective
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                
                # Actor loss = -min(surr1, surr2) - entropy_weight * entropy
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy.mean()
                
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
                
                # Critic loss = MSE(V, returns)
                values = self.critic(past[idx], cur[idx], fut[idx])
                critic_loss = F.mse_loss(values, returns[idx])
                
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()
                
                # Stats
                with torch.no_grad():
                    kl = (old_log_probs[idx] - new_log_probs).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()
                
                stats['actor_loss'].append(actor_loss.item())
                stats['critic_loss'].append(critic_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['kl'].append(kl.item())
                stats['clip_frac'].append(clip_frac.item())
        
        return {k: np.mean(v) for k, v in stats.items()}


def compute_difficulty(csv_path):
    """Difficulty = variance + rate of change of target lataccel"""
    try:
        df = pd.read_csv(csv_path)
        target = df['targetLateralAcceleration'].values
        return np.var(target) + 2 * np.mean(np.abs(np.diff(target)))
    except:
        return float('inf')


def run_episode(data_path, actor, critic, device):
    """Run episode, collect experiences"""
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel('models/tinyphysics.onnx', debug=False)
    
    class Wrapper:
        def __init__(self):
            self.history = []
            self.experiences = []
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
                action, log_prob, entropy = actor.get_action(past_t, cur_t, fut_t, deterministic=False)
                value = critic(past_t, cur_t, fut_t)
                
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            # Reward (negative cost)
            err = current_lataccel - target_lataccel
            jerk = (current_lataccel - self.prev_lataccel) / 0.1
            reward = -(5.0 * err**2 + 0.1 * jerk**2)
            
            self.experiences.append({
                'past_ctx': past_ctx,
                'current': current,
                'future_ctx': future_ctx,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'value': value,
                'done': False
            })
            
            self.prev_steer = float(np.clip(action, STEER_MIN, STEER_MAX))
            self.prev_lataccel = current_lataccel
            return self.prev_steer
    
    wrapper = Wrapper()
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    try:
        sim.rollout()
        cost = sim.compute_cost()['total_cost']
    except:
        cost = 10000.0
    
    # Mark last experience as done
    if wrapper.experiences:
        wrapper.experiences[-1]['done'] = True
    
    return wrapper.experiences, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--episodes_per_epoch', type=int, default=16)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--entropy_weight', type=float, default=0.01)
    parser.add_argument('--curriculum_phases', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create networks
    actor = GaussianActor(d_model=128).to(device)
    critic = ValueCritic(d_model=128).to(device)
    
    # Create PPO agent
    agent = PPOAgent(
        actor, critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        entropy_weight=args.entropy_weight
    )
    
    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Load and sort files by difficulty
    data_dir = Path('data')
    all_files = sorted(data_dir.glob('*.csv'))[:900]
    
    print("\nComputing difficulties...")
    diffs = [(f, compute_difficulty(f)) for f in all_files]
    sorted_files = [f for f, _ in sorted(diffs, key=lambda x: x[1])]
    
    # Curriculum phases
    n_per_phase = len(sorted_files) // args.curriculum_phases
    phases = [sorted_files[i*n_per_phase:(i+1)*n_per_phase] for i in range(args.curriculum_phases)]
    
    print(f"Curriculum: {args.curriculum_phases} phases, {n_per_phase} segments each")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f'models/ppo_proper_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log = []
    best_cost = float('inf')
    epochs_per_phase = args.epochs // args.curriculum_phases
    
    print("\n" + "="*60)
    print("PPO TRAINING (Proper Implementation)")
    print("="*60)
    
    for phase in range(args.curriculum_phases):
        # Include all easier phases
        available = []
        for p in range(phase + 1):
            available.extend(phases[p])
        
        print(f"\n--- Phase {phase+1}/{args.curriculum_phases} ({len(available)} segments) ---")
        
        for epoch in range(1, epochs_per_phase + 1):
            global_epoch = phase * epochs_per_phase + epoch
            
            # Collect episodes (sequential for stability)
            files = np.random.choice(available, size=min(args.episodes_per_epoch, len(available)), replace=False)
            
            all_experiences = []
            costs = []
            
            actor.eval()
            critic.eval()
            
            for f in files:
                exps, cost = run_episode(f, actor, critic, device)
                if exps:
                    # Compute returns and advantages
                    rewards = [e['reward'] for e in exps]
                    values = [e['value'] for e in exps]
                    dones = [e['done'] for e in exps]
                    
                    returns, advantages = agent.compute_gae(rewards, values, dones)
                    
                    for i, e in enumerate(exps):
                        e['return'] = returns[i]
                        e['advantage'] = advantages[i]
                    
                    all_experiences.extend(exps)
                    costs.append(cost)
            
            if not all_experiences:
                continue
            
            # PPO update
            actor.train()
            critic.train()
            
            stats = agent.update(all_experiences, device, num_epochs=4, mini_batch_size=64)
            
            print(f"  Epoch {global_epoch}: cost={np.mean(costs):.1f}±{np.std(costs):.1f}, "
                  f"actor={stats['actor_loss']:.4f}, critic={stats['critic_loss']:.1f}, "
                  f"entropy={stats['entropy']:.3f}, kl={stats['kl']:.4f}")
            
            log.append({
                'epoch': global_epoch, 'phase': phase+1,
                'cost': float(np.mean(costs)),
                **{k: float(v) for k, v in stats.items()}
            })
            
            # Save best
            if np.mean(costs) < best_cost:
                best_cost = np.mean(costs)
                torch.save({
                    'epoch': global_epoch,
                    'actor_state': actor.state_dict(),
                    'critic_state': critic.state_dict(),
                    'cost': best_cost
                }, out_dir / 'best_model.pt')
                print(f"  [NEW BEST]")
    
    with open(out_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Best cost: {best_cost:.1f}")
    print(f"Output: {out_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

