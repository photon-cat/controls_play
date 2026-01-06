#!/usr/bin/env python3
"""
PPO for steering control - using exact same implementation as CartPole.
Uses PID demonstrations for warm-start (optional).

Usage:
    # Train from scratch
    python3 test_ppo_steering.py --no-warmstart --segments 5
    
    # Train with PID warm-start (recommended)
    python3 test_ppo_steering.py --segments 10 --warmstart-demos 5
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from collections import deque
import json
from datetime import datetime

from steering_env import SteeringEnv, collect_pid_demonstrations, behavior_clone_init


class SteeringPolicy(nn.Module):
    """
    Continuous policy for steering control.
    Same structure as CartPole but adapted for continuous actions.
    
    Conservative exploration to stay near BC-initialized policy.
    """
    def __init__(self, state_dim=104, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh()  # Output in [-1, 1], will scale to [-2, 2]
        )
        self.log_std = nn.Parameter(torch.tensor(-2.5))  # std ~0.082 (conservative!)
    
    def forward(self, x, deterministic=False):
        mean = self.net(x) * 2.0  # Scale to [-2, 2]
        std = torch.exp(self.log_std.clamp(-3, 0))
        
        if deterministic:
            return mean, None, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(-2.0, 2.0)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update"""
        mean = self.net(states) * 2.0
        std = torch.exp(self.log_std.clamp(-3, 0))
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Simple MLP value function"""
    def __init__(self, state_dim=104, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class RolloutBuffer:
    """Store trajectories for PPO updates"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones)
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    values = values.tolist() + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values[:-1])
    return advantages, returns


def ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer, 
               epochs=4, batch_size=64, clip_eps=0.1, value_coef=0.5, entropy_coef=0.01):
    """PPO update - conservative clip for steering (0.1 instead of 0.2)"""
    
    states, actions, rewards, old_log_probs, values, dones = buffer.get()
    
    # Compute advantages and returns
    with torch.no_grad():
        next_value = value_net(states[-1:]).item()
    advantages, returns = compute_gae(rewards, values, dones, next_value)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    n_samples = len(states)
    indices = np.arange(n_samples)
    
    policy_losses = []
    value_losses = []
    entropies = []
    clip_fracs = []
    kls = []
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # Policy loss
            new_log_probs, entropy = policy.evaluate(batch_states, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            batch_values = value_net(batch_states)
            value_loss = F.mse_loss(batch_values, batch_returns)
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            
            # Update
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            optimizer_policy.step()
            optimizer_value.step()
            
            # Logging
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
            
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
                clip_fracs.append(clip_frac)
                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                kls.append(approx_kl)
    
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropies),
        'clip_frac': np.mean(clip_fracs),
        'kl': np.mean(kls)
    }


def train_ppo_steering(args):
    """Train PPO on steering control"""
    
    # Setup
    data_dir = Path(args.data_path)
    model_path = args.model_path
    data_files = sorted(data_dir.glob('*.csv'))[:args.segments]
    
    print("="*60)
    print("PPO STEERING CONTROL")
    print("="*60)
    print(f"Segments: {len(data_files)}")
    print(f"Warm-start: {args.warmstart}")
    print()
    
    # Networks
    policy = SteeringPolicy(state_dim=104, hidden=args.hidden)
    value_net = ValueNetwork(state_dim=104, hidden=args.hidden)
    
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.lr_value)
    
    # Warm-start with PID demonstrations
    if args.warmstart:
        demo_states, demo_actions = collect_pid_demonstrations(
            data_files, model_path, n_demos=args.warmstart_demos
        )
        behavior_clone_init(policy, demo_states, demo_actions, epochs=args.bc_epochs)
    
    # Training
    buffer = RolloutBuffer()
    episode_costs = deque(maxlen=10)
    episode_rewards = deque(maxlen=10)
    
    total_episodes = 0
    
    # Create save directory
    save_dir = Path('models') / f'ppo_steering_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_log = []
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Collect episodes
        epoch_costs = []
        
        for seg_idx, data_file in enumerate(data_files):
            env = SteeringEnv(data_file, model_path)
            state = env.reset()
            episode_reward = 0
            
            # Rollout
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    value = value_net(state_tensor).item()
                    action_tensor, log_prob, entropy = policy(state_tensor)
                    action = action_tensor.item()
                    log_prob = log_prob.item()
                
                next_state, reward, done, info = env.step(action)
                
                buffer.add(state, action, reward, log_prob, value, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Episode complete
            cost = env.get_final_cost()
            episode_costs.append(cost['total_cost'])
            episode_rewards.append(episode_reward)
            epoch_costs.append(cost['total_cost'])
            total_episodes += 1
            
            if (seg_idx + 1) % 5 == 0:
                avg_cost = np.mean(list(episode_costs))
                avg_reward = np.mean(list(episode_rewards))
                print(f"  Seg {seg_idx+1:2d}: cost={cost['total_cost']:7.1f}, "
                      f"reward={episode_reward:7.1f}, avg_cost={avg_cost:7.1f}")
        
        # PPO Update (conservative clip_eps=0.1)
        stats = ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer,
                          epochs=args.ppo_epochs, batch_size=args.batch_size, clip_eps=0.1)
        buffer.clear()
        
        avg_epoch_cost = np.mean(epoch_costs)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg cost: {avg_epoch_cost:.1f}")
        print(f"  Policy loss: {stats['policy_loss']:.4f}")
        print(f"  Value loss: {stats['value_loss']:.2f}")
        print(f"  Entropy: {stats['entropy']:.3f}")
        print(f"  Clip frac: {stats['clip_frac']:.3f}")
        print(f"  KL: {stats['kl']:.4f}")
        
        # Log
        training_log.append({
            'epoch': epoch + 1,
            'avg_cost': avg_epoch_cost,
            'std_cost': np.std(epoch_costs),
            **stats
        })
        
        # Save
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'policy': policy.state_dict(),
                'value': value_net.state_dict(),
                'epoch': epoch + 1
            }, save_dir / f'checkpoint_epoch_{epoch+1:03d}.pt')
            print(f"  Saved checkpoint to {save_dir}")
    
    # Save final
    torch.save({
        'policy': policy.state_dict(),
        'value': value_net.state_dict(),
        'epoch': args.epochs
    }, save_dir / 'final.pt')
    
    with open(save_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete! Saved to {save_dir}")
    print(f"Final avg cost: {avg_epoch_cost:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_path", default="data", help="Path to data directory")
    parser.add_argument("--model_path", default="models/tinyphysics.onnx", help="Physics model")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments per epoch")
    
    # Warm-start
    parser.add_argument("--warmstart", dest='warmstart', action='store_true', help="Use PID warm-start")
    parser.add_argument("--no-warmstart", dest='warmstart', action='store_false')
    parser.set_defaults(warmstart=True)
    parser.add_argument("--warmstart-demos", type=int, default=5, help="PID demos for BC")
    parser.add_argument("--bc-epochs", type=int, default=10, help="Behavior cloning epochs")
    
    # Architecture
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr-policy", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--lr-value", type=float, default=3e-4, help="Value learning rate")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train_ppo_steering(args)

