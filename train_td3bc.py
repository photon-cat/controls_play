#!/usr/bin/env python3
"""
TD3+BC: Offline reinforcement learning for steering control.

Based on "A Minimalist Approach to Offline Reinforcement Learning" (Fujimoto & Gu, 2021)
TD3 with behavior cloning regularization term.

Key idea: Learn from offline PID dataset without environment interaction.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm


class Actor(nn.Module):
    """Deterministic policy network"""
    def __init__(self, state_dim=104, action_dim=1, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state) * 2.0  # Scale to [-2, 2]


class Critic(nn.Module):
    """Twin Q-networks for stability"""
    def __init__(self, state_dim=104, action_dim=1, hidden=256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


class ReplayBuffer:
    """Simple replay buffer for offline data"""
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions).unsqueeze(1)
        self.rewards = torch.FloatTensor(rewards).unsqueeze(1)
        self.next_states = torch.FloatTensor(next_states)
        self.dones = torch.FloatTensor(dones).unsqueeze(1)
        self.size = len(states)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )


class TD3_BC:
    """TD3 with Behavior Cloning for offline RL"""
    
    def __init__(self, state_dim=104, action_dim=1, hidden=256, device='cpu'):
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2  # Delayed policy updates
        self.alpha = 2.5  # BC weight (key parameter!)
        
        self.total_it = 0
    
    def select_action(self, state):
        """Select action (for evaluation)"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state)
        return action.cpu().numpy().flatten()[0]
    
    def train(self, replay_buffer, batch_size=256):
        """Single training step"""
        self.total_it += 1
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # ============= Critic Update =============
        with torch.no_grad():
            # Select action with target policy + noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-2.0, 2.0)
            
            # Compute target Q
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============= Delayed Actor Update =============
        actor_loss = None
        bc_loss = None
        q_value = None
        
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic.q1_forward(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            
            # TD3+BC objective: maximize Q + BC regularization
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            bc_loss = F.mse_loss(pi, action)
            q_value = q.mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else 0.0,
            'bc_loss': bc_loss.item() if bc_loss is not None else 0.0,
            'q_value': q_value.item() if q_value is not None else 0.0
        }
    
    def save(self, path):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


def train_td3_bc(args):
    """Main training loop"""
    
    # Load dataset
    print("="*60)
    print("TD3+BC OFFLINE RL TRAINING")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    
    data = np.load(args.dataset)
    print(f"\nDataset loaded:")
    print(f"  States: {data['states'].shape}")
    print(f"  Actions: {data['actions'].shape}")
    print(f"  Transitions: {data['num_transitions']}")
    print(f"  Avg cost: {data['costs'].mean():.2f}")
    
    # Normalize states
    if args.normalize:
        states = (data['states'] - data['state_mean']) / data['state_std']
        next_states = (data['next_states'] - data['state_mean']) / data['state_std']
    else:
        states = data['states']
        next_states = data['next_states']
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        states, data['actions'], data['rewards'],
        next_states, data['dones']
    )
    
    # Initialize TD3+BC
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    agent = TD3_BC(
        state_dim=104,
        action_dim=1,
        hidden=args.hidden,
        device=device
    )
    agent.alpha = args.alpha
    
    print(f"\nHyperparameters:")
    print(f"  Alpha (BC weight): {agent.alpha}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden size: {args.hidden}")
    print(f"  Total steps: {args.steps}")
    
    # Training
    save_dir = Path('models') / f'td3bc_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    training_log = []
    
    print("\nTraining...")
    pbar = tqdm(range(args.steps), desc="Training")
    
    for step in pbar:
        # Train
        stats = agent.train(replay_buffer, batch_size=args.batch_size)
        
        # Log
        if (step + 1) % args.log_every == 0:
            pbar.set_postfix({
                'critic_loss': f"{stats['critic_loss']:.2f}",
                'bc_loss': f"{stats['bc_loss']:.4f}",
                'q': f"{stats['q_value']:.2f}"
            })
            
            training_log.append({
                'step': step + 1,
                **stats
            })
        
        # Save
        if (step + 1) % args.save_every == 0:
            agent.save(save_dir / f'checkpoint_{step+1:06d}.pt')
            
            with open(save_dir / 'training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)
            
            print(f"\nSaved checkpoint at step {step+1}")
    
    # Save final model
    agent.save(save_dir / 'final.pt')
    
    with open(save_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Save config
    config = {
        'dataset': str(args.dataset),
        'alpha': args.alpha,
        'hidden': args.hidden,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'normalize': args.normalize
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to offline dataset (.npz)')
    parser.add_argument('--steps', type=int, default=100000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--alpha', type=float, default=2.5, help='BC weight')
    parser.add_argument('--normalize', action='store_true', help='Normalize states')
    parser.add_argument('--log_every', type=int, default=1000, help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=10000, help='Save every N steps')
    args = parser.parse_args()
    
    train_td3_bc(args)

