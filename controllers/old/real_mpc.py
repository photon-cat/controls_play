"""
MPC controller using the actual TinyPhysicsModel for prediction.
- Uses real physics model for accurate rollouts
- Searches over candidate steer sequences
- Minimizes true cost function (lataccel_error^2 * 50 + jerk^2)
"""
from . import BaseController
import numpy as np
from copy import deepcopy

CONTROL_START_IDX = 100
MAX_ACC_DELTA = 0.5
STEER_RANGE = [-2, 2]


class Controller(BaseController):
    def __init__(self):
        self.physics_model = None
        self.data_path = None

        # MPC params
        self.horizon = 5  # Steps to look ahead
        self.n_candidates = 7  # Number of candidate steers to try
        self.steer_delta = 0.15  # Search range around baseline

        # State for simulation
        self.sim_states = []
        self.sim_actions = []
        self.sim_preds = []

        # Previous steer for baseline
        self.prev_steer = 0.0
        self.step_count = 0

        self._log = {}

    def set_physics_model(self, model):
        """Receive the physics model from tinyphysics."""
        self.physics_model = model

    def set_data_path(self, path):
        """Receive data path for potential use."""
        self.data_path = path

    def _simulate_step(self, states, actions, preds, new_action):
        """
        Simulate one step using the physics model.
        Returns predicted lataccel.
        """
        if self.physics_model is None:
            # Fallback: simple gain model
            return preds[-1] if preds else 0.0

        # Add new action and get prediction
        actions_with_new = actions + [new_action]

        pred = self.physics_model.get_current_lataccel(
            sim_states=states[-20:],  # Context length
            actions=actions_with_new[-20:],
            past_preds=preds[-20:]
        )

        # Apply rate limit
        if preds:
            pred = np.clip(pred, preds[-1] - MAX_ACC_DELTA, preds[-1] + MAX_ACC_DELTA)

        return pred

    def _rollout_cost(self, current_states, current_actions, current_preds,
                      candidate_steer, future_targets, future_rolls):
        """
        Simulate forward with constant steer and compute cost.
        """
        # Copy state
        states = list(current_states)
        actions = list(current_actions)
        preds = list(current_preds)

        total_lataccel_cost = 0.0
        total_jerk_cost = 0.0
        prev_pred = preds[-1] if preds else 0.0

        for h in range(min(self.horizon, len(future_targets))):
            # Simulate one step
            pred = self._simulate_step(states, actions, preds, candidate_steer)

            # Compute costs
            target = future_targets[h]
            lataccel_err = (target - pred) ** 2
            jerk = ((pred - prev_pred) / 0.1) ** 2

            total_lataccel_cost += lataccel_err
            total_jerk_cost += jerk

            # Update for next iteration
            preds.append(pred)
            actions.append(candidate_steer)
            # Note: states would need future values, approximate with last
            if h < len(future_rolls):
                # Create approximate state
                last_state = states[-1]
                new_state = type(last_state)(
                    roll_lataccel=future_rolls[h],
                    v_ego=last_state.v_ego,
                    a_ego=last_state.a_ego
                )
                states.append(new_state)
            else:
                states.append(states[-1])

            prev_pred = pred

        # Match tinyphysics cost function
        total_cost = (total_lataccel_cost * 100 / self.horizon) * 50 + (total_jerk_cost * 100 / self.horizon)
        return total_cost

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1

        # Update simulation state
        self.sim_states.append(state)
        self.sim_preds.append(current_lataccel)

        # Keep history bounded
        if len(self.sim_states) > 30:
            self.sim_states.pop(0)
        if len(self.sim_preds) > 30:
            self.sim_preds.pop(0)
        if len(self.sim_actions) > 30:
            self.sim_actions.pop(0)

        # Simple feedforward baseline
        roll_la = state.roll_lataccel
        turning = target_lataccel - roll_la
        # Approximate gain
        gain = 1.5  # Conservative estimate
        baseline_steer = turning / gain

        # Generate candidates around baseline and previous steer
        center = 0.5 * baseline_steer + 0.5 * self.prev_steer  # Blend
        candidates = np.linspace(
            center - self.steer_delta,
            center + self.steer_delta,
            self.n_candidates
        )
        candidates = np.clip(candidates, STEER_RANGE[0], STEER_RANGE[1])

        # Get future targets and rolls
        future_targets = [target_lataccel]  # Current
        future_rolls = [roll_la]
        if future_plan.lataccel:
            future_targets.extend(future_plan.lataccel[:self.horizon])
        if future_plan.roll_lataccel:
            future_rolls.extend(future_plan.roll_lataccel[:self.horizon])

        # Evaluate candidates
        best_steer = baseline_steer
        best_cost = float('inf')

        if self.physics_model is not None and len(self.sim_states) >= 2:
            for steer in candidates:
                cost = self._rollout_cost(
                    self.sim_states, self.sim_actions, self.sim_preds,
                    steer, future_targets, future_rolls
                )
                if cost < best_cost:
                    best_cost = cost
                    best_steer = steer

        u_cmd = np.clip(best_steer, STEER_RANGE[0], STEER_RANGE[1])

        # Store action
        self.sim_actions.append(u_cmd)
        self.prev_steer = u_cmd

        error = target_lataccel - current_lataccel
        self._log = {
            'mpc_baseline': baseline_steer,
            'mpc_best': best_steer,
            'mpc_cost': best_cost if best_cost < float('inf') else -1,
            'error': error,
        }

        return u_cmd
