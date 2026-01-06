#!/usr/bin/env python3
"""
PPO Fine-tuning for Steering Model

Uses PPO (Proximal Policy Optimization) to fine-tune the PID imitation model.
Key features:
- Clipped objective prevents catastrophic updates
- Value function baseline reduces variance
- Multiple epochs on same data for sample efficiency
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import signal
import atexit
import json
from datetime import datetime


# ============================================================================
# Model Definition (Actor-Critic version of SteerModelRL)
# ============================================================================

class SteerActorCritic(nn.Module):
    """
    Actor-Critic model for PPO.
    Actor: outputs steering action (mean, std)
    Critic: outputs value estimate V(s)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # Input projections (shared)
        self.past_proj = nn.Linear(6, d_model)
        self.current_proj = nn.Linear(4, d_model)
        self.future_proj = nn.Linear(4, d_model)
        
        # Position embeddings: 10 past + 1 current + 10 future = 21 positions
        self.pos_emb = nn.Parameter(torch.randn(21, d_model) * 0.02)
        
        # Shared Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.actor_log_std = nn.Parameter(torch.tensor(-1.5))  # Initial std ~0.22
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Input normalization
        self.register_buffer('past_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('past_std', torch.tensor([15., 1., 0.5, 1., 1., 0.5]))
        self.register_buffer('current_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('current_std', torch.tensor([15., 1., 0.5, 1.]))
        self.register_buffer('future_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('future_std', torch.tensor([15., 1., 0.5, 1.]))
    
    def _get_features(self, past_ctx, current, future_ctx):
        """Shared feature extraction"""
        # Normalize inputs
        past_ctx = (past_ctx - self.past_mean) / self.past_std
        current = (current - self.current_mean) / self.current_std
        future_ctx = (future_ctx - self.future_mean) / self.future_std
        
        # Project to d_model
        past_emb = self.past_proj(past_ctx)
        current_emb = self.current_proj(current).unsqueeze(1)
        future_emb = self.future_proj(future_ctx)
        
        # Concatenate: [past..., current, future...]
        seq = torch.cat([past_emb, current_emb, future_emb], dim=1)
        seq = seq + self.pos_emb
        
        # Transformer
        out = self.transformer(seq)
        
        # Use current position (index 10) for output
        return out[:, 10]
    
    def forward(self, past_ctx, current, future_ctx):
        """Returns action mean, std, and value"""
        features = self._get_features(past_ctx, current, future_ctx)
        
        # Actor
        action_mean = self.actor_mean(features).squeeze(-1)
        action_std = torch.exp(self.actor_log_std.clamp(-3, 0))  # std in [0.05, 1.0]
        
        # Critic
        value = self.critic(features).squeeze(-1)
        
        return action_mean, action_std, value
    
    def act(self, past_ctx, current, future_ctx, deterministic=False):
        """Sample action and return log_prob, value"""
        mean, std, value = self.forward(past_ctx, current, future_ctx)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate(self, past_ctx, current, future_ctx, action):
        """Evaluate log_prob and value for given actions (for PPO update)"""
        mean, std, value = self.forward(past_ctx, current, future_ctx)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


# ============================================================================
# PPO Buffer
# ============================================================================

class PPOBuffer:
    """Stores rollout data for PPO update"""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.past_ctx = []
        self.current = []
        self.future_ctx = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def add(self, past, cur, fut, action, log_prob, value, reward, done):
        self.past_ctx.append(past)
        self.current.append(cur)
        self.future_ctx.append(fut)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get_tensors(self, device):
        """Convert to tensors for training"""
        return {
            'past': torch.tensor(np.array(self.past_ctx), dtype=torch.float32, device=device),
            'current': torch.tensor(np.array(self.current), dtype=torch.float32, device=device),
            'future': torch.tensor(np.array(self.future_ctx), dtype=torch.float32, device=device),
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float32, device=device),
            'log_probs': torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            'values': torch.tensor(np.array(self.values), dtype=torch.float32, device=device),
            'rewards': torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            'dones': torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
        }
    
    def __len__(self):
        return len(self.actions)


# ============================================================================
# PPO Controller for Rollouts
# ============================================================================

class PPOController:
    """Controller wrapper for PPO rollouts"""
    def __init__(self, model, device, buffer=None, deterministic=False):
        self.model = model
        self.device = device
        self.buffer = buffer
        self.deterministic = deterministic
        
        # Context history
        self.past_ctx = []
        self.steer_history = []
        self.lataccel_history = []
        self.prev_action = None
        self.prev_reward = None
    
    def reset(self):
        self.past_ctx = []
        self.steer_history = []
        self.lataccel_history = []
        self.prev_action = None
        self.prev_reward = None
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        from tinyphysics import STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER
        
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll = state.roll_lataccel
        
        # Update histories
        self.lataccel_history.append(current_lataccel)
        if len(self.steer_history) == 0:
            self.steer_history.append(0.0)
        
        # Build context
        if len(self.past_ctx) < 10:
            # Not enough history yet - use simple proportional control
            action = (target_lataccel - current_lataccel) * 0.3
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.steer_history.append(action)
            
            # Build past context entry
            ctx_entry = [v_ego, a_ego, roll, target_lataccel, current_lataccel, action]
            self.past_ctx.append(ctx_entry)
            
            return action
        
        # Build input tensors
        past_np = np.array(self.past_ctx[-10:], dtype=np.float32)
        cur_np = np.array([v_ego, a_ego, roll, current_lataccel], dtype=np.float32)
        
        # Future context - FuturePlan is namedtuple with lists: lataccel, roll_lataccel, v_ego, a_ego
        future_list = []
        for i in range(10):
            if future_plan is not None and hasattr(future_plan, 'lataccel') and i < len(future_plan.lataccel):
                future_list.append([
                    future_plan.v_ego[i],
                    future_plan.a_ego[i],
                    future_plan.roll_lataccel[i],
                    future_plan.lataccel[i]
                ])
            else:
                future_list.append([v_ego, a_ego, roll, target_lataccel])
        fut_np = np.array(future_list, dtype=np.float32)
        
        # To tensor
        past_t = torch.tensor(past_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        cur_t = torch.tensor(cur_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        fut_t = torch.tensor(fut_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action from model
        with torch.no_grad():
            action, log_prob, value = self.model.act(past_t, cur_t, fut_t, deterministic=self.deterministic)
        
        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, STEER_RANGE[0], STEER_RANGE[1])
        
        # Compute reward (negative cost, scaled down for stability)
        lataccel_error = (target_lataccel - current_lataccel) ** 2 * LAT_ACCEL_COST_MULTIPLIER
        if len(self.lataccel_history) >= 2:
            jerk = ((self.lataccel_history[-1] - self.lataccel_history[-2]) / DEL_T) ** 2
        else:
            jerk = 0
        # Scale reward to reasonable range (costs are ~100-10000, so divide by 100)
        reward = -(lataccel_error + jerk) / 100.0
        
        # Store in buffer (if collecting)
        if self.buffer is not None and self.prev_action is not None:
            # Store the PREVIOUS step (now we know its reward)
            self.buffer.add(
                past_np, cur_np, fut_np,
                self.prev_action,
                self.prev_log_prob,
                self.prev_value,
                reward,
                False  # Not terminal
            )
        
        # Save for next step
        self.prev_action = action_np
        self.prev_log_prob = log_prob.cpu().numpy()[0]
        self.prev_value = value.cpu().numpy()[0]
        
        # Update histories
        self.steer_history.append(action_np)
        ctx_entry = [v_ego, a_ego, roll, target_lataccel, current_lataccel, action_np]
        self.past_ctx.append(ctx_entry)
        
        return action_np


# ============================================================================
# PPO Training
# ============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    
    return advantages, returns


def ppo_update(model, optimizer, buffer, device, 
               clip_eps=0.1, value_coef=0.5, entropy_coef=0.05,
               n_epochs=4, batch_size=64, gamma=0.99, lam=0.95):
    """PPO update step"""
    
    data = buffer.get_tensors(device)
    
    # Compute GAE
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones,
        gamma=gamma, lam=lam
    )
    advantages = advantages.to(device)
    returns = returns.to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    n_samples = len(buffer)
    indices = np.arange(n_samples)
    
    total_loss = 0
    total_pg_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0
    
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            # Get batch
            past = data['past'][batch_idx]
            current = data['current'][batch_idx]
            future = data['future'][batch_idx]
            actions = data['actions'][batch_idx]
            old_log_probs = data['log_probs'][batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # Forward pass
            new_log_probs, values, entropy = model.evaluate(past, current, future, actions)
            
            # Policy loss (clipped PPO objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            pg_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, batch_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = pg_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1
    
    return {
        'loss': total_loss / n_updates,
        'pg_loss': total_pg_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'entropy': total_entropy / n_updates,
    }


def run_episode(model, data_path, physics_model_path, device, buffer=None, deterministic=False):
    """Run one episode and return cost"""
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    physics = TinyPhysicsModel(physics_model_path, debug=False)
    controller = PPOController(model, device, buffer=buffer, deterministic=deterministic)
    
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=controller, debug=False)
    sim.rollout()
    
    cost = sim.compute_cost()
    
    # Mark last step as terminal
    if buffer is not None and len(buffer) > 0:
        buffer.dones[-1] = True
    
    return cost['total_cost']


def load_imitation_model(checkpoint_path, model, device):
    """Load weights from imitation model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Map old model keys to new actor-critic model
    new_state_dict = {}
    for key, value in state_dict.items():
        if not hasattr(value, 'shape'):
            continue  # Skip non-tensor values
        
        if key.startswith('mean_head'):
            # Map mean_head to actor_mean
            new_key = key.replace('mean_head', 'actor_mean')
            new_state_dict[new_key] = value
        elif key.startswith('log_std_head'):
            # Skip - we use a single parameter now
            pass
        elif key == 'baseline':
            # Skip
            pass
        else:
            new_state_dict[key] = value
    
    # Load what we can
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='PPO Fine-tuning for Steering Model')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--resume', type=str, default='models/teacher_ckpts_20251225_173549/imitation_epoch_003.pt',
                        help='Path to imitation model checkpoint')
    parser.add_argument('--output', type=str, default='models/steer_model_ppo.pt', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of PPO epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=50, help='Episodes per PPO epoch (more = stable gradients)')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (lower = stable)')
    parser.add_argument('--clip_eps', type=float, default=0.1, help='PPO clip epsilon (tighter = conservative)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy_coef', type=float, default=0.05, help='Entropy coefficient (higher = more exploration)')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    physics_model_path = 'models/tinyphysics.onnx'
    
    # Get data files
    data_path = Path(args.data_path)
    data_files = sorted(data_path.glob('*.csv'))[:5000]  # Use first 5000 segments
    print(f"Found {len(data_files)} data files")
    
    # Create model
    model = SteerActorCritic(d_model=args.hidden).to(device)
    
    # Load imitation model weights
    if args.resume:
        load_imitation_model(args.resume, model, device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Setup checkpointing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = Path(f'models/ppo_checkpoints_{timestamp}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training log
    training_log = []
    best_cost = float('inf')
    
    def save_log():
        log_path = ckpt_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"📝 Training log saved: {log_path}")
    
    signal.signal(signal.SIGINT, lambda s, f: (save_log(), exit(1)))
    atexit.register(save_log)
    
    print(f"\n{'='*60}")
    print(f"PPO FINE-TUNING")
    print(f"{'='*60}")
    print(f"Checkpoints: {ckpt_dir}")
    
    # Evaluate initial model
    print("\nEvaluating initial model...")
    model.eval()
    eval_costs = []
    eval_files = np.random.choice(data_files, min(20, len(data_files)), replace=False)
    for f in tqdm(eval_files, desc="Eval"):
        cost = run_episode(model, f, physics_model_path, device, buffer=None, deterministic=True)
        eval_costs.append(cost)
    initial_cost = np.mean(eval_costs)
    print(f"Initial cost: {initial_cost:.1f} (std={np.std(eval_costs):.1f})")
    best_cost = initial_cost
    
    # PPO Training loop
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Collect rollouts
        model.eval()
        buffer = PPOBuffer()
        episode_costs = []
        
        episode_files = np.random.choice(data_files, args.episodes_per_epoch, replace=False)
        
        for data_file in tqdm(episode_files, desc="Collecting"):
            cost = run_episode(model, data_file, physics_model_path, device, 
                             buffer=buffer, deterministic=False)
            episode_costs.append(cost)
        
        mean_cost = np.mean(episode_costs)
        print(f"  Rollout cost: {mean_cost:.1f} (std={np.std(episode_costs):.1f}), samples={len(buffer)}")
        
        # PPO update
        model.train()
        update_stats = ppo_update(
            model, optimizer, buffer, device,
            clip_eps=args.clip_eps,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            n_epochs=4,
            batch_size=64
        )
        
        print(f"  PPO: loss={update_stats['loss']:.4f}, pg={update_stats['pg_loss']:.4f}, "
              f"value={update_stats['value_loss']:.4f}, entropy={update_stats['entropy']:.4f}")
        
        # Evaluate
        model.eval()
        eval_costs = []
        eval_files = np.random.choice(data_files, min(20, len(data_files)), replace=False)
        for f in tqdm(eval_files, desc="Eval"):
            cost = run_episode(model, f, physics_model_path, device, buffer=None, deterministic=True)
            eval_costs.append(cost)
        eval_cost = np.mean(eval_costs)
        
        print(f"  Eval cost: {eval_cost:.1f} (std={np.std(eval_costs):.1f})")
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'rollout_cost': mean_cost,
            'eval_cost': eval_cost,
            **update_stats
        }
        training_log.append(log_entry)
        
        # Save best
        if eval_cost < best_cost:
            best_cost = eval_cost
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_cost': eval_cost,
            }, args.output)
            print(f"  → Saved best model (cost={eval_cost:.1f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_cost': eval_cost,
        }, ckpt_dir / f'epoch_{epoch+1:03d}.pt')
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Initial cost: {initial_cost:.1f}")
    print(f"Best cost: {best_cost:.1f}")
    print(f"Improvement: {initial_cost - best_cost:.1f} ({100*(initial_cost-best_cost)/initial_cost:.1f}%)")
    print(f"Best model: {args.output}")


if __name__ == '__main__':
    main()

