"""
Pure feedforward controller using model-based sampling.
Samples candidate steers and evaluates them using the actual physics model.
"""
from . import BaseController
import numpy as np

CONTEXT_LENGTH = 20
DEL_T = 0.1
H = 3                     # FF horizon
NUM_SAMPLES = 7
SAMPLE_RADIUS = 0.3
STEER_RANGE = (-2.0, 2.0)

W_TRACK = 1.0
W_JERK = 0.1
W_SMOOTH = 0.3


def cost(a_hat, a_prev, a_target, u, u_prev):
    track = (a_target - a_hat)**2
    jerk = ((a_hat - a_prev) / DEL_T)**2
    smooth = (u - u_prev)**2
    return W_TRACK * track + W_JERK * jerk + W_SMOOTH * smooth


class Controller(BaseController):
    def __init__(self):
        self.sim_model = None
        self.state_hist = []
        self.action_hist = []
        self.lataccel_hist = []
        self.prev_action = 0.0
        self.step_count = 0
        self._log = {}

    def set_physics_model(self, model):
        self.sim_model = model

    def observe_action(self, action, step_idx):
        """Receive actual action used (for warmup phase)."""
        # Replace the last action with actual one used
        if self.action_hist:
            self.action_hist[-1] = action
        self.prev_action = action

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1

        # Build history
        self.state_hist.append(state)
        self.lataccel_hist.append(current_lataccel)

        if len(self.state_hist) > CONTEXT_LENGTH:
            self.state_hist = self.state_hist[-CONTEXT_LENGTH:]
        if len(self.lataccel_hist) > CONTEXT_LENGTH:
            self.lataccel_hist = self.lataccel_hist[-CONTEXT_LENGTH:]
        if len(self.action_hist) > CONTEXT_LENGTH:
            self.action_hist = self.action_hist[-CONTEXT_LENGTH:]

        # Need enough history and model
        if self.sim_model is None or len(self.state_hist) < CONTEXT_LENGTH:
            u = self.prev_action  # Use previous (will be replaced by observe_action)
            self.action_hist.append(u)
            return u

        # Check future plan
        if not future_plan.lataccel or len(future_plan.lataccel) < H:
            u = self.prev_action
            self.action_hist.append(u)
            self.prev_action = u
            return u

        # Sample candidates
        candidates = np.clip(
            self.prev_action + np.linspace(-SAMPLE_RADIUS, SAMPLE_RADIUS, NUM_SAMPLES),
            STEER_RANGE[0], STEER_RANGE[1]
        )

        best_u = self.prev_action
        best_J = float("inf")

        for u0 in candidates:
            sim_states = list(self.state_hist[-CONTEXT_LENGTH:])
            sim_actions = list(self.action_hist[-CONTEXT_LENGTH:])
            sim_preds = list(self.lataccel_hist[-CONTEXT_LENGTH:])

            u_prev = sim_actions[-1] if sim_actions else 0.0
            a_prev = sim_preds[-1] if sim_preds else current_lataccel
            J = 0.0

            for k in range(H):
                sim_actions.append(u0)
                sim_actions = sim_actions[-CONTEXT_LENGTH:]

                a_hat = self.sim_model.get_current_lataccel(
                    sim_states,
                    sim_actions,
                    sim_preds
                )

                a_target = future_plan.lataccel[k]
                J += cost(a_hat, a_prev, a_target, u0, u_prev)

                sim_preds.append(a_hat)
                sim_preds = sim_preds[-CONTEXT_LENGTH:]
                sim_states.append(sim_states[-1])
                sim_states = sim_states[-CONTEXT_LENGTH:]

                a_prev = a_hat
                u_prev = u0

            if J < best_J:
                best_J = J
                best_u = u0

        u = np.clip(best_u, STEER_RANGE[0], STEER_RANGE[1])
        self.action_hist.append(u)
        self.prev_action = u

        self._log = {
            'best_cost': best_J,
            'error': target_lataccel - current_lataccel,
        }

        return u
