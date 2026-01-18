"""
Q-Learning controller with small neural network.
- State: [error, error_rate, roll, target, current_la, prev_steer]
- Action: discretized steer values
- Reward: negative cost (lataccel_error^2 * 50 + jerk^2)
- Learns online during episode
"""
from . import BaseController
import numpy as np

STEER_RANGE = [-2.0, 2.0]
NUM_ACTIONS = 41  # Discretize steer into 41 bins (-2 to 2 in 0.1 steps)


class SimpleQNetwork:
    """Tiny neural network for Q-values."""

    def __init__(self, input_dim, num_actions, hidden_dim=32):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, num_actions) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(num_actions)

    def forward(self, x):
        """Forward pass with ReLU activations."""
        self.x = x
        self.h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        self.h2 = np.maximum(0, self.h1 @ self.W2 + self.b2)  # ReLU
        self.q_values = self.h2 @ self.W3 + self.b3
        return self.q_values

    def backward(self, target_q, action_idx, lr=0.001):
        """Backward pass for single action Q-value update."""
        # Only update the Q-value for the taken action
        td_error = self.q_values[action_idx] - target_q

        # Clip TD error to prevent explosion
        td_error = np.clip(td_error, -10.0, 10.0)

        q_error = np.zeros(self.num_actions)
        q_error[action_idx] = td_error

        # Gradient through output layer
        dW3 = np.outer(self.h2, q_error)
        db3 = q_error
        dh2 = q_error @ self.W3.T

        # Gradient through hidden layer 2 (ReLU)
        dh2 = dh2 * (self.h2 > 0)
        dW2 = np.outer(self.h1, dh2)
        db2 = dh2
        dh1 = dh2 @ self.W2.T

        # Gradient through hidden layer 1 (ReLU)
        dh1 = dh1 * (self.h1 > 0)
        dW1 = np.outer(self.x, dh1)
        db1 = dh1

        # Clip gradients
        max_grad = 1.0
        dW3 = np.clip(dW3, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW1 = np.clip(dW1, -max_grad, max_grad)
        db3 = np.clip(db3, -max_grad, max_grad)
        db2 = np.clip(db2, -max_grad, max_grad)
        db1 = np.clip(db1, -max_grad, max_grad)

        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


class Controller(BaseController):
    def __init__(self):
        # Q-network: state -> Q-values for each action
        # State: [error, error_rate, roll, target, current_la, prev_steer, v_ego_norm]
        self.state_dim = 7
        self.q_net = SimpleQNetwork(self.state_dim, NUM_ACTIONS, hidden_dim=48)

        # Q-learning params
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.5  # Exploration rate (starts high, decays)
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.lr = 0.0005  # Smaller learning rate for stability

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 200
        self.batch_size = 16

        # State tracking
        self.prev_steer = 0.0
        self.prev_lataccel = None
        self.prev_state = None
        self.prev_action_idx = None
        self.step_count = 0

        # Action discretization
        self.actions = np.linspace(STEER_RANGE[0], STEER_RANGE[1], NUM_ACTIONS)

        self._log = {}

    def _get_state(self, target, current, roll, v_ego):
        """Build normalized state vector."""
        error = target - current
        error_rate = (current - self.prev_lataccel) if self.prev_lataccel is not None else 0.0

        # Normalize all inputs to roughly [-1, 1] range
        return np.array([
            np.clip(error / 2.0, -1, 1),  # Error normalized by max expected
            np.clip(error_rate / 0.5, -1, 1),  # Error rate normalized
            np.clip(roll / 2.0, -1, 1),  # Roll normalized
            np.clip(target / 2.0, -1, 1),  # Target normalized
            np.clip(current / 2.0, -1, 1),  # Current normalized
            self.prev_steer / 2.0,  # Prev steer is already in [-2, 2]
            np.clip((v_ego - 25) / 15, -1, 1),  # v_ego normalized
        ])

    def _compute_reward(self, target, current, prev_current):
        """Compute reward (negative cost, scaled for stability)."""
        lataccel_err = (target - current) ** 2
        jerk = ((current - prev_current) / 0.1) ** 2 if prev_current is not None else 0.0

        # Reward is negative cost (higher is better), scaled down
        cost = lataccel_err * 50 + jerk
        reward = -cost / 10.0  # Scale down to prevent large Q-values

        # Bonus for being close to target
        if abs(target - current) < 0.1:
            reward += 0.5

        return np.clip(reward, -10.0, 10.0)

    def _select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            # Explore: random action, but bias toward center
            action_idx = int(np.clip(
                np.random.normal(NUM_ACTIONS // 2, NUM_ACTIONS // 4),
                0, NUM_ACTIONS - 1
            ))
        else:
            # Exploit: choose best Q-value
            q_values = self.q_net.forward(state)
            action_idx = np.argmax(q_values)

        return action_idx

    def _train_step(self):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)

        for idx in indices:
            state, action_idx, reward, next_state, done = self.replay_buffer[idx]

            # Compute target Q-value
            if done:
                target_q = reward
            else:
                next_q = self.q_net.forward(next_state)
                target_q = reward + self.gamma * np.max(next_q)

            # Update Q-network
            self.q_net.forward(state)
            self.q_net.backward(target_q, action_idx, self.lr)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        roll_la = state.roll_lataccel
        v_ego = state.v_ego

        # Build current state
        current_state = self._get_state(target_lataccel, current_lataccel, roll_la, v_ego)

        # Store experience from previous step
        if self.prev_state is not None and self.prev_action_idx is not None:
            reward = self._compute_reward(target_lataccel, current_lataccel, self.prev_lataccel)

            # Add to replay buffer
            self.replay_buffer.append((
                self.prev_state,
                self.prev_action_idx,
                reward,
                current_state,
                False  # Not done (could check if end of episode)
            ))

            # Keep buffer bounded
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)

            # Train
            self._train_step()

        # Select action
        action_idx = self._select_action(current_state)
        u_cmd = self.actions[action_idx]

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store for next step
        self.prev_state = current_state
        self.prev_action_idx = action_idx
        self.prev_lataccel = current_lataccel
        self.prev_steer = u_cmd

        error = target_lataccel - current_lataccel
        q_values = self.q_net.forward(current_state)
        self._log = {
            'q_max': np.max(q_values),
            'q_min': np.min(q_values),
            'epsilon': self.epsilon,
            'action_idx': action_idx,
            'error': error,
            'buffer_size': len(self.replay_buffer),
        }

        return u_cmd
