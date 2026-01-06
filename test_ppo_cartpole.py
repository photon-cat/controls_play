#!/usr/bin/env python3
"""
Minimal PPO implementation on CartPole-v1 to verify algorithm.
This is a sanity check before applying to steering control.

Expected performance:
- CartPole-v1 solves at ~500 reward
- Should reach this in 50-100 episodes with proper PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque


class PolicyNetwork(nn.Module):
    """Simple MLP policy for discrete actions"""
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob.item(), entropy.item()
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update"""
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Simple MLP value function"""
    def __init__(self, state_dim, hidden=64):
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
            torch.FloatTensor(self.states),
            torch.LongTensor(self.actions),
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
               epochs=4, batch_size=64, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    """PPO update with multiple epochs over the buffer"""
    
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


def train_ppo_cartpole(episodes=200, steps_per_update=2048, lr=3e-4):
    """Train PPO on CartPole-v1"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Target: 500 reward (solved)")
    print()
    
    # Networks
    policy = PolicyNetwork(state_dim, action_dim, hidden=64)
    value_net = ValueNetwork(state_dim, hidden=64)
    
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    buffer = RolloutBuffer()
    episode_rewards = deque(maxlen=100)
    
    state, _ = env.reset() if hasattr(env, 'reset') and len(env.reset()) == 2 else (env.reset(), None)
    episode_reward = 0
    episode_count = 0
    
    for step in range(steps_per_update * episodes):
        # Collect experience
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = value_net(state_tensor).item()
        
        action, log_prob, entropy = policy.get_action(state_tensor)
        
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, _ = result
            truncated = False
        else:
            next_state, reward, done, truncated, _ = result
        
        done = done or truncated
        
        buffer.add(state, action, reward, log_prob, value, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode_count:3d}: reward={episode_reward:6.1f}, avg_100={avg_reward:6.1f}")
                
                if avg_reward >= 500:
                    print(f"\n🎉 SOLVED in {episode_count} episodes!")
                    return True
            
            state, _ = env.reset() if hasattr(env, 'reset') and len(env.reset()) == 2 else (env.reset(), None)
            episode_reward = 0
        
        # Update policy
        if (step + 1) % steps_per_update == 0:
            stats = ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer)
            print(f"  Update: policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.2f}, "
                  f"entropy={stats['entropy']:.3f}, clip_frac={stats['clip_frac']:.3f}, kl={stats['kl']:.4f}")
            buffer.clear()
    
    print(f"\nFinal avg reward: {np.mean(episode_rewards):.1f}")
    return False


def train_ppo_pendulum(episodes=50, steps_per_update=2048, lr=3e-4):
    """Train PPO on Pendulum-v1 (continuous control)"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    
    print(f"Environment: Pendulum-v1")
    print(f"State dim: {state_dim}")
    print(f"Target: ~-200 reward (good performance)")
    print()
    
    # Simple continuous policy
    class ContinuousPolicy(nn.Module):
        def __init__(self, state_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Tanh()
            )
            self.log_std = nn.Parameter(torch.tensor(0.0))
        
        def forward(self, x):
            mean = self.net(x) * 2.0  # Scale to [-2, 2]
            return mean
        
        def get_action(self, state, deterministic=False):
            with torch.no_grad():
                mean = self.forward(state).squeeze()
                std = torch.exp(self.log_std.clamp(-3, 0))
                if deterministic:
                    return mean.item(), 0.0, 0.0
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            return action.item(), log_prob.item(), entropy.item()
        
        def evaluate(self, states, actions):
            mean = self.forward(states).squeeze()
            std = torch.exp(self.log_std.clamp(-3, 0))
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy()
            return log_probs, entropy
    
    policy = ContinuousPolicy(state_dim, hidden=64)
    value_net = ValueNetwork(state_dim, hidden=64)
    
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    buffer = RolloutBuffer()
    episode_rewards = deque(maxlen=100)
    
    state, _ = env.reset() if hasattr(env, 'reset') and len(env.reset()) == 2 else (env.reset(), None)
    episode_reward = 0
    episode_count = 0
    
    for step in range(steps_per_update * episodes):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = value_net(state_tensor).item()
        
        action, log_prob, entropy = policy.get_action(state_tensor)
        
        result = env.step([action])
        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, done, truncated, _ = result
            done = done or truncated
        
        buffer.add(state, action, reward, log_prob, value, done)
        
        state = next_state
        episode_reward += reward
        
        if done or (step + 1) % 200 == 0:  # Pendulum has no natural done
            episode_rewards.append(episode_reward)
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode_count:3d}: reward={episode_reward:7.1f}, avg_100={avg_reward:7.1f}")
            
            state, _ = env.reset() if hasattr(env, 'reset') and len(env.reset()) == 2 else (env.reset(), None)
            episode_reward = 0
        
        # Update policy
        if (step + 1) % steps_per_update == 0:
            stats = ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer)
            print(f"  Update: policy_loss={stats['policy_loss']:.4f}, value_loss={stats['value_loss']:.2f}, "
                  f"entropy={stats['entropy']:.3f}, clip_frac={stats['clip_frac']:.3f}, kl={stats['kl']:.4f}")
            buffer.clear()
    
    print(f"\nFinal avg reward: {np.mean(episode_rewards):.1f}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    
    print("="*60)
    print("PPO SANITY CHECK")
    print("="*60)
    print()
    
    if args.env == "cartpole":
        train_ppo_cartpole(episodes=args.episodes)
    else:
        train_ppo_pendulum(episodes=args.episodes)

