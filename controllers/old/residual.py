from . import BaseController
import numpy as np


class ResidualPolicy:
    """
    Tiny residual policy: state -> delta_u
    Single hidden layer, tanh output (bounded).
    """

    def __init__(self, state_dim, hidden_dim=16, max_residual=0.3):
        self.max_residual = max_residual

        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.x = x
        self.h = np.tanh(x @ self.W1 + self.b1)
        self.u = np.tanh(self.h @ self.W2 + self.b2)[0]
        return self.max_residual * self.u

    def update(self, grad, lr=1e-3):
        """
        Policy gradient step.
        grad = d(cost)/d(u)
        """
        du = grad * self.max_residual * (1 - self.u ** 2)

        # du is scalar, W2 is (hidden_dim, 1)
        dW2 = self.h.reshape(-1, 1) * du
        db2 = np.array([du])

        # Backprop through hidden layer
        dh = du * self.W2.flatten() * (1 - self.h ** 2)
        dW1 = np.outer(self.x, dh)
        db1 = dh

        # Clip gradients
        dW2 = np.clip(dW2, -1.0, 1.0)
        dW1 = np.clip(dW1, -1.0, 1.0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


class Controller(BaseController):
    """
    PID + Residual RL Controller
    """

    def __init__(self):
        # --- PID ---
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053

        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_lataccel = None

        # --- Residual RL ---
        self.state_dim = 5
        self.policy = ResidualPolicy(self.state_dim)
        self.lr = 5e-4

    def _get_state(self, target, current, roll, v_ego):
        error = target - current
        error_rate = current - self.prev_lataccel if self.prev_lataccel is not None else 0.0

        return np.array([
            np.clip(error / 2.0, -1, 1),
            np.clip(error_rate / 0.5, -1, 1),
            np.clip(roll / 2.0, -1, 1),
            np.clip(target / 2.0, -1, 1),
            np.clip((v_ego - 25) / 15, -1, 1),
        ])

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # --- PID ---
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = (
            self.p * error +
            self.i * self.error_integral +
            self.d * error_diff
        )

        # --- Residual RL ---
        rl_state = self._get_state(
            target_lataccel,
            current_lataccel,
            state.roll_lataccel,
            state.v_ego
        )

        delta_u = self.policy.forward(rl_state)

        # --- Combine ---
        u = u_pid + delta_u

        # --- Online policy update ---
        # Cost: error^2 + small jerk proxy
        if self.prev_lataccel is not None:
            jerk = current_lataccel - self.prev_lataccel
            cost_grad = 2 * error + 0.1 * jerk
            self.policy.update(cost_grad, self.lr)

        self.prev_lataccel = current_lataccel
        return u
