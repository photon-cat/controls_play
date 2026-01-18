"""
MPC controller with learned dynamics model.
- Fits a dynamics model during warmup (steps 0-99) using labeled steer_command
- Uses the learned model for MPC predictions during control (steps 100+)
- Model: next_lataccel = momentum * current_lataccel + (1-momentum) * (roll + gain * steer)
"""
from . import BaseController
import numpy as np
import pandas as pd

ACC_G = 9.81
CONTROL_START_IDX = 100
MAX_ACC_DELTA = 0.5
STEER_RANGE = [-2.0, 2.0]


class Controller(BaseController):
    def __init__(self):
        # Data collection for model fitting
        self.warmup_data = []  # [(lataccel, steer, roll, v_ego, next_lataccel), ...]
        self.labeled_data = None  # Will hold loaded CSV data

        # Learned model parameters
        self.gain = 2.0  # Initial guess (fits warmup data)
        self.momentum = 0.5  # How much previous lataccel carries forward (NN has memory)
        self.model_fitted = False

        # Lag model: lag steps = base + scale * |steer|
        self.lag_base = 3.0
        self.lag_scale = 2.0

        # Online learning rates
        self.gain_lr = 0.05
        self.momentum_lr = 0.02

        # MPC params
        self.horizon = 12
        self.n_candidates = 21
        self.steer_delta = 0.5  # Search range around baseline

        # State tracking
        self.prev_lataccel = None
        self.prev_steer = 0.0
        self.step_count = 0
        self.data_path = None

        self._log = {}

    def set_data_path(self, path):
        """Load labeled data for model fitting."""
        self.data_path = path
        try:
            df = pd.read_csv(path)
            self.labeled_data = {
                'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
                'v_ego': df['vEgo'].values,
                'target_lataccel': df['targetLateralAcceleration'].values,
                'steer_command': -df['steerCommand'].values,  # Sign convention
            }
        except Exception as e:
            print(f"Warning: Could not load data: {e}")

    def observe_action(self, action, step_idx):
        """Receive actual steer command (labeled during warmup)."""
        # Store for model fitting
        if step_idx < CONTROL_START_IDX and self.prev_lataccel is not None:
            # Collect transition: (prev_la, steer, roll, v_ego) -> current_la
            # Note: We'll fill in current_la in the next update() call
            pass

    def _fit_model(self):
        """
        Fit dynamics model from labeled warmup data using least squares.

        We use the labeled steer_command and target_lataccel from the CSV.
        During warmup, current_lataccel == target_lataccel (they're the same).

        Model: next_la = momentum * curr_la + (1-momentum) * (roll + gain * steer)

        Rearranged: next_la = alpha * curr_la + beta * roll + gamma * steer + bias
        where alpha = momentum, beta = (1-momentum), gamma = (1-momentum) * gain
        """
        if self.labeled_data is None:
            print("Warning: No labeled data available for model fitting")
            return

        # Use steps 20-99 (context starts at 20, control starts at 100)
        start_idx = 21  # Need previous step
        end_idx = CONTROL_START_IDX

        curr_la = self.labeled_data['target_lataccel'][start_idx-1:end_idx-1]
        next_la = self.labeled_data['target_lataccel'][start_idx:end_idx]
        steer = self.labeled_data['steer_command'][start_idx-1:end_idx-1]  # Steer that caused transition
        roll = self.labeled_data['roll_lataccel'][start_idx:end_idx]

        n_samples = len(curr_la)
        if n_samples < 20:
            print(f"Warning: Only {n_samples} samples for fitting")
            return

        # Build design matrix [curr_la, roll, steer, 1]
        X = np.column_stack([curr_la, roll, steer, np.ones(n_samples)])
        y = next_la

        # Least squares fit
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, gamma, bias = coeffs

            # Recover gain parameter (keep momentum at initial value since warmup data has different dynamics)
            # beta should be (1 - momentum), gamma should be (1-momentum) * gain
            # From fit: next_la = alpha*curr + beta*roll + gamma*steer
            # If alpha is near 0, the warmup data shows instant response
            # But NN has memory, so keep momentum at init value and just learn gain
            if abs(1 - alpha) > 1e-6:
                fitted_gain = gamma / (1 - alpha)
                self.gain = float(np.clip(fitted_gain, 0.5, 4.0))
            # Keep momentum at initial value (0.5) since warmup dynamics differ from control

            self.model_fitted = True

            # Compute fit quality
            y_pred = X @ coeffs
            mse = np.mean((y - y_pred) ** 2)

            print(f"Model fit: momentum={self.momentum:.3f}, gain={self.gain:.3f}, MSE={mse:.6f}")
            print(f"  Coeffs: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}, bias={bias:.4f}")

        except np.linalg.LinAlgError as e:
            print(f"Model fitting failed: {e}")

    def _compute_lag(self, steer_mag):
        """Compute lag in steps based on steer magnitude."""
        lag = int(round(self.lag_base + self.lag_scale * steer_mag))
        return min(lag, 8)

    def _predict_lataccel(self, curr_la, steer, roll):
        """Predict next lataccel using learned model."""
        steady_state = roll + self.gain * steer
        next_la = self.momentum * curr_la + (1 - self.momentum) * steady_state
        # Apply rate limit
        next_la = np.clip(next_la, curr_la - MAX_ACC_DELTA, curr_la + MAX_ACC_DELTA)
        return next_la

    def _rollout_cost(self, curr_la, candidate_steer, future_targets, future_rolls, prev_steer):
        """
        Simulate forward with candidate steer and compute cost.
        Uses learned dynamics model for prediction.
        Accounts for lag: new steer doesn't take effect until lag steps later.
        """
        pred_la = curr_la
        prev_la = curr_la

        # Compute lag for this candidate steer
        lag = self._compute_lag(abs(candidate_steer))

        total_lataccel_cost = 0.0
        total_jerk_cost = 0.0

        for h in range(min(self.horizon, len(future_targets))):
            target = future_targets[h]
            roll = future_rolls[h] if h < len(future_rolls) else future_rolls[-1]

            # Before lag: use previous steer effect
            # After lag: use new candidate steer effect
            if h < lag:
                effective_steer = prev_steer
            else:
                effective_steer = candidate_steer

            # Predict next lataccel
            pred_la = self._predict_lataccel(pred_la, effective_steer, roll)

            # Costs
            lataccel_err = (target - pred_la) ** 2
            jerk = ((pred_la - prev_la) / 0.1) ** 2

            total_lataccel_cost += lataccel_err
            total_jerk_cost += jerk
            prev_la = pred_la

        # Match tinyphysics cost: lataccel_cost * 50 + jerk_cost (both scaled by 100/N)
        n = min(self.horizon, len(future_targets))
        if n > 0:
            total_cost = (total_lataccel_cost * 100 / n) * 50 + (total_jerk_cost * 100 / n)
        else:
            total_cost = 0.0

        return total_cost

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        roll_la = state.roll_lataccel
        v_ego = state.v_ego

        # At step 100: fit model from labeled warmup data
        if self.step_count == CONTROL_START_IDX and not self.model_fitted:
            self._fit_model()

        # Feedforward baseline with lag lookahead
        lag = self._compute_lag(abs(self.prev_steer))
        if lag > 0 and future_plan.lataccel and len(future_plan.lataccel) >= lag:
            lookahead_target = future_plan.lataccel[lag - 1]
            lookahead_roll = future_plan.roll_lataccel[lag - 1] if future_plan.roll_lataccel and len(future_plan.roll_lataccel) >= lag else roll_la
        else:
            lookahead_target = target_lataccel
            lookahead_roll = roll_la

        turning = lookahead_target - lookahead_roll
        baseline_steer = turning / max(self.gain, 0.5)

        # Generate candidates around baseline and previous
        center = 0.5 * baseline_steer + 0.5 * self.prev_steer
        candidates = np.linspace(
            center - self.steer_delta,
            center + self.steer_delta,
            self.n_candidates
        )
        candidates = np.clip(candidates, STEER_RANGE[0], STEER_RANGE[1])

        # Build future targets and rolls
        future_targets = [target_lataccel]
        future_rolls = [roll_la]
        if future_plan.lataccel:
            future_targets.extend(list(future_plan.lataccel[:self.horizon]))
        if future_plan.roll_lataccel:
            future_rolls.extend(list(future_plan.roll_lataccel[:self.horizon]))

        # MPC: evaluate candidates
        best_steer = baseline_steer
        best_cost = float('inf')

        for steer in candidates:
            cost = self._rollout_cost(current_lataccel, steer, future_targets, future_rolls, self.prev_steer)
            if cost < best_cost:
                best_cost = cost
                best_steer = steer

        u_cmd = np.clip(best_steer, STEER_RANGE[0], STEER_RANGE[1])

        # Online model update using prediction error
        if self.prev_lataccel is not None:
            # What we predicted vs what happened
            predicted = self._predict_lataccel(self.prev_lataccel, self.prev_steer, roll_la)
            pred_error = current_lataccel - predicted

            # Only update when error is significant and steer is non-trivial
            if abs(self.prev_steer) > 0.05:
                steady_state = roll_la + self.gain * self.prev_steer

                # d(pred)/d(gain) = (1-momentum) * prev_steer
                d_gain = (1 - self.momentum) * self.prev_steer
                self.gain += self.gain_lr * pred_error * d_gain
                self.gain = float(np.clip(self.gain, 0.5, 4.0))

                # d(pred)/d(momentum) = prev_la - steady_state
                d_momentum = self.prev_lataccel - steady_state
                self.momentum += self.momentum_lr * pred_error * d_momentum
                self.momentum = float(np.clip(self.momentum, 0.1, 0.95))

        self.prev_lataccel = current_lataccel
        self.prev_steer = u_cmd

        error = target_lataccel - current_lataccel
        self._log = {
            'trained_gain': self.gain,
            'trained_momentum': self.momentum,
            'trained_model_fitted': int(self.model_fitted),
            'mpc_baseline': baseline_steer,
            'mpc_best': best_steer,
            'mpc_cost': best_cost if best_cost < float('inf') else -1,
            'error': error,
        }

        return u_cmd
