# PPO Controller Design for Controls Challenge

## Overview

On-policy RL controller using PPO with history context and future plan lookahead.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Observation                          │
├─────────────────┬─────────────────┬─────────────────────┤
│  History (t-20  │   Current (t0)  │  Future (t+1 to     │
│   to t-1)       │                 │   t+20)             │
│  6 feat × 20    │   5 features    │  4 feat × 20        │
└────────┬────────┴────────┬────────┴──────────┬──────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           ▼
                   ┌───────────────┐
                   │  Encoder      │
                   │  (MLP or      │
                   │   Transformer)│
                   └───────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │   Actor     │          │   Critic    │
       │  (Policy)   │          │  (Value)    │
       └──────┬──────┘          └──────┬──────┘
              │                        │
              ▼                        ▼
         action μ, σ               V(s)
              │
              ▼
         u ~ N(μ, σ)
```

---

## Observation Space

### History Features (t-20 to t-1): 120 dims
```python
history_features = [
    'v_ego',           # velocity
    'a_ego',           # forward acceleration
    'road_lataccel',   # roll-induced lateral accel
    'measured_lataccel', # actual lateral accel
    'target_lataccel', # desired lateral accel
    'action',          # steer command taken
]
# Shape: (20, 6) flattened to (120,)
```

### Current Features (t0): 6 dims
```python
current_features = [
    'v_ego',
    'a_ego',
    'road_lataccel',
    'measured_lataccel',
    'target_lataccel',
    'error',  # target - measured (explicit control signal)
]
```

### Future Features (t+1 to t+20): 80 dims
```python
future_features = [
    'v_ego',
    'a_ego',
    'road_lataccel',
    'target_lataccel',
]
# Shape: (20, 4) flattened to (80,)
```

### Total Observation: 206 dims

---

## Action Space

```python
# Continuous, single scalar
action_space = Box(low=-2.0, high=2.0, shape=(1,))

# Policy outputs Gaussian parameters
mu = actor_head(features)      # mean, tanh scaled to [-2, 2]
log_std = learnable_param      # or network output, clamped [-2, 0]
```

---

## Reward Function

```python
def compute_reward(target_lataccel, actual_lataccel, prev_lataccel, dt=0.1):
    # Tracking error (primary objective)
    tracking_error = (target_lataccel - actual_lataccel) ** 2

    # Jerk penalty (smoothness)
    jerk = ((actual_lataccel - prev_lataccel) / dt) ** 2

    # Weighted reward (negative cost)
    # Matches eval: total_cost = lataccel_cost * 50 + jerk_cost
    reward = -(50.0 * tracking_error + jerk)

    # Optional: scale down for stable training
    reward = reward * 0.01

    return reward
```

---

## Network Architecture

```python
import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(
        self,
        history_dim=120,    # 20 steps × 6 features
        current_dim=6,
        future_dim=80,      # 20 steps × 4 features
        hidden_dim=256,
    ):
        super().__init__()

        total_dim = history_dim + current_dim + future_dim  # 206

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1], scale to [-2, 2]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1))  # Learnable std

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.action_scale = 2.0  # Scale tanh output to [-2, 2]

    def forward(self, obs):
        features = self.encoder(obs)
        return features

    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)

        mean = self.actor_mean(features) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        action = torch.clamp(action, -self.action_scale, self.action_scale)
        return action

    def evaluate_actions(self, obs, actions):
        features = self.forward(obs)

        mean = self.actor_mean(features) * self.action_scale
        std = self.actor_log_std.exp().expand_as(mean)

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self.critic(features)

        return log_probs, entropy, values
```

---

## PPO Training Loop

```python
import numpy as np
from collections import deque

class PPOTrainer:
    def __init__(
        self,
        policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        # values has one extra element (bootstrap value)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

        return advantages, returns

    def update(self, rollout_buffer):
        """PPO update step."""
        obs = torch.tensor(rollout_buffer['obs'], dtype=torch.float32)
        actions = torch.tensor(rollout_buffer['actions'], dtype=torch.float32)
        old_log_probs = torch.tensor(rollout_buffer['log_probs'], dtype=torch.float32)
        advantages = rollout_buffer['advantages']
        returns = rollout_buffer['returns']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.update_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((values.squeeze() - batch_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
        }
```

---

## Environment Wrapper

```python
class ControlsEnv:
    """Wraps tinyphysics simulator as a gym-like environment."""

    def __init__(self, sim, data_path, context_length=20):
        self.sim = sim
        self.data_path = data_path
        self.context_length = context_length

        # History buffers
        self.history = {
            'v_ego': deque(maxlen=context_length),
            'a_ego': deque(maxlen=context_length),
            'road_lataccel': deque(maxlen=context_length),
            'measured_lataccel': deque(maxlen=context_length),
            'target_lataccel': deque(maxlen=context_length),
            'action': deque(maxlen=context_length),
        }

    def reset(self, seg_path):
        """Reset env with new segment."""
        self.sim.reset(seg_path)

        # Clear history
        for k in self.history:
            self.history[k].clear()

        self.step_idx = 0
        self.prev_lataccel = 0.0

        # Warmup to fill history
        return self._warmup()

    def _warmup(self):
        """Run warmup steps to fill history buffer."""
        while len(self.history['v_ego']) < self.context_length:
            state, current_lataccel, target, future = self.sim.get_state()

            # Use zero or simple controller during warmup
            action = 0.0

            self.history['v_ego'].append(state.v_ego)
            self.history['a_ego'].append(state.a_ego)
            self.history['road_lataccel'].append(state.road_lataccel)
            self.history['measured_lataccel'].append(current_lataccel)
            self.history['target_lataccel'].append(target)
            self.history['action'].append(action)

            self.sim.step(action)
            self.step_idx += 1

        return self._get_obs()

    def _get_obs(self):
        """Build observation vector."""
        state, current_lataccel, target, future = self.sim.get_state()

        # History (flattened)
        hist = np.concatenate([
            np.array(self.history['v_ego']),
            np.array(self.history['a_ego']),
            np.array(self.history['road_lataccel']),
            np.array(self.history['measured_lataccel']),
            np.array(self.history['target_lataccel']),
            np.array(self.history['action']),
        ])

        # Current
        error = target - current_lataccel
        curr = np.array([
            state.v_ego,
            state.a_ego,
            state.road_lataccel,
            current_lataccel,
            target,
            error,
        ])

        # Future (pad if needed)
        future_len = min(20, len(future.v_ego))
        fut_v = np.zeros(20)
        fut_a = np.zeros(20)
        fut_roll = np.zeros(20)
        fut_target = np.zeros(20)

        fut_v[:future_len] = future.v_ego[:future_len]
        fut_a[:future_len] = future.a_ego[:future_len]
        fut_roll[:future_len] = future.road_lataccel[:future_len]
        fut_target[:future_len] = future.lataccel[:future_len]

        fut = np.concatenate([fut_v, fut_a, fut_roll, fut_target])

        return np.concatenate([hist, curr, fut]).astype(np.float32)

    def step(self, action):
        """Execute action, return (obs, reward, done, info)."""
        state, current_lataccel, target, future = self.sim.get_state()

        # Execute in simulator
        new_lataccel = self.sim.step(action)

        # Compute reward
        tracking_error = (target - new_lataccel) ** 2
        jerk = ((new_lataccel - self.prev_lataccel) / 0.1) ** 2
        reward = -(50.0 * tracking_error + jerk) * 0.01

        # Update history
        self.history['v_ego'].append(state.v_ego)
        self.history['a_ego'].append(state.a_ego)
        self.history['road_lataccel'].append(state.road_lataccel)
        self.history['measured_lataccel'].append(new_lataccel)
        self.history['target_lataccel'].append(target)
        self.history['action'].append(action)

        self.prev_lataccel = new_lataccel
        self.step_idx += 1

        done = self.sim.is_done()
        obs = self._get_obs() if not done else None

        return obs, reward, done, {'lataccel': new_lataccel, 'target': target}
```

---

## Training Script

```python
def train_ppo(
    num_iterations=1000,
    rollout_length=2048,
    num_segments=100,
):
    # Initialize
    policy = PPOPolicy()
    trainer = PPOTrainer(policy)

    # Load segment paths
    seg_paths = glob.glob('./data/*.csv')[:num_segments]

    for iteration in range(num_iterations):
        # Collect rollouts
        rollout_buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
        }

        steps_collected = 0

        while steps_collected < rollout_length:
            # Sample random segment
            seg_path = np.random.choice(seg_paths)
            env = ControlsEnv(sim, seg_path)
            obs = env.reset(seg_path)

            done = False
            while not done and steps_collected < rollout_length:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action = policy.get_action(obs_tensor)
                    _, _, value = policy.evaluate_actions(obs_tensor, action)

                    # Get log prob for this action
                    features = policy.forward(obs_tensor)
                    mean = policy.actor_mean(features) * policy.action_scale
                    std = policy.actor_log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(action)

                action_np = action.squeeze().numpy()
                next_obs, reward, done, info = env.step(action_np)

                rollout_buffer['obs'].append(obs)
                rollout_buffer['actions'].append(action_np)
                rollout_buffer['rewards'].append(reward)
                rollout_buffer['log_probs'].append(log_prob.item())
                rollout_buffer['values'].append(value.item())
                rollout_buffer['dones'].append(float(done))

                obs = next_obs
                steps_collected += 1

        # Bootstrap value for last state
        if not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, _, bootstrap_value = policy.evaluate_actions(
                    obs_tensor,
                    policy.get_action(obs_tensor)
                )
                rollout_buffer['values'].append(bootstrap_value.item())
        else:
            rollout_buffer['values'].append(0.0)

        # Compute advantages
        advantages, returns = trainer.compute_gae(
            rollout_buffer['rewards'],
            rollout_buffer['values'],
            rollout_buffer['dones'],
        )
        rollout_buffer['advantages'] = advantages
        rollout_buffer['returns'] = returns

        # Update policy
        stats = trainer.update(rollout_buffer)

        # Logging
        mean_reward = np.mean(rollout_buffer['rewards'])
        print(f"Iter {iteration}: reward={mean_reward:.4f}, "
              f"policy_loss={stats['policy_loss']:.4f}, "
              f"value_loss={stats['value_loss']:.4f}")

        # Save checkpoint
        if iteration % 100 == 0:
            torch.save(policy.state_dict(), f'checkpoints/ppo_{iteration}.pt')
```

---

## Hyperparameter Recommendations

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | Standard for PPO |
| Gamma | 0.99 | High for continuous control |
| GAE Lambda | 0.95 | Bias-variance tradeoff |
| Clip epsilon | 0.2 | Standard PPO clip |
| Entropy coef | 0.01 | Encourage exploration early |
| Value coef | 0.5 | Balance value/policy loss |
| Update epochs | 10 | Per rollout batch |
| Batch size | 64 | Mini-batch size |
| Rollout length | 2048 | Steps before update |
| Hidden dim | 256 | Network width |

---

## Tips for This Problem

1. **Normalize observations** - v_ego, a_ego, lataccel have different scales. Standardize inputs.

2. **Reward scaling** - The raw cost can be large. Scale rewards to ~[-1, 1] range.

3. **Curriculum** - Start with easier segments (less aggressive maneuvers), increase difficulty.

4. **Action smoothing** - Add a small penalty for `(action_t - action_{t-1})^2` to reduce oscillation.

5. **Exploration schedule** - Start with higher entropy coef (0.05), decay to 0.001.

6. **Segment diversity** - Train on many different segments to generalize.

7. **Early stopping** - If tracking error explodes, reset that rollout.
