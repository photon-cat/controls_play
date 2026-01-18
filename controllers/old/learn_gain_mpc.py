"""
Adaptive gain controller with MPC instead of PID.
- Starts from a velocity-based gain LUT
- Updates gain online to reduce lateral acceleration error
- Uses MPC to select best steer from candidates
- Uses magnitude-dependent lag lookahead: lag = 3 + 2.5 * |steer|
- Uses second-order dynamics with momentum for prediction
"""
from . import BaseController
import numpy as np

GAIN_LUT = {
    0.5: 1.32, 1.5: 0.96, 2.5: 1.09, 3.5: 1.27, 4.5: 1.53,
    5.5: 1.67, 6.5: 1.78, 7.5: 1.88, 8.5: 1.72, 9.5: 1.33,
    10.5: 1.29, 11.5: 1.48, 12.5: 1.67, 13.5: 1.79, 14.5: 1.40,
    15.5: 1.06, 16.5: 1.16, 17.5: 1.60, 18.5: 1.69, 19.5: 1.48,
    20.5: 1.00, 21.5: 1.18, 22.5: 1.35, 23.5: 1.30, 24.5: 1.12,
    25.5: 1.59, 26.5: 1.60, 27.5: 1.70, 28.5: 1.61, 29.5: 1.57,
    30.5: 1.59, 31.5: 1.15, 32.5: 1.65, 33.5: 1.90, 34.5: 1.73,
    35.5: 1.95, 36.5: 1.35, 37.5: 2.45, 38.5: 1.69, 39.5: 2.43,
    40.5: 2.57,
}

GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])

CONTROL_START_IDX = 100
MAX_ACC_DELTA = 0.5  # Max lataccel change per step (from tinyphysics)


class Controller(BaseController):
    def __init__(self):
        self.gain = None
        self.min_gain = 0.4
        self.max_gain = 4.0

        # Learning params
        self.base_lr = 0.03
        self.lag_base = 3.0
        self.lag_scale = 2.5
        self.turning_threshold = 0.05
        self.min_command = 0.1

        # MPC params
        self.mpc_horizon = 8
        self.mpc_candidates = 11  # Number of steer candidates to try
        self.mpc_steer_range = 0.3  # Search +/- this around feedforward
        self.jerk_weight = 0.1  # Weight on jerk penalty in MPC cost

        # Second-order dynamics params (momentum model) - LEARNED online
        self.momentum = 0.5  # How much velocity carries forward (0=no inertia, 1=full inertia)
        self.response_rate = 0.2  # How quickly system responds to control (damping)
        self.dynamics_lr = 0.01  # Learning rate for momentum/response_rate

        self.prev_u_cmd = 0.0
        self.prev_u_rate = 0.0
        self.step_count = 0
        self.current_lataccel = 0.0
        self.prev_lataccel = None  # None until initialized
        self.prev_velocity = 0.0  # Track velocity for learning

        self.cmd_history = []

        self._log = {}

    def get_gain_lut(self, v_ego):
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def _compute_lag_steps(self, steer_mag):
        """Compute lag based on steer magnitude: lag = 3 + 2.5 * |steer|"""
        lag = int(round(self.lag_base + self.lag_scale * steer_mag))
        return min(lag, 8)

    def _mpc_cost(self, steer_candidates, current_la, current_velocity, future_plan, roll_lataccel, prev_steer):
        """
        Evaluate cost for each candidate steer over horizon.
        Cost = sum of (target - predicted)^2 * 50 + jerk^2

        Uses second-order dynamics with momentum:
        - velocity = momentum * prev_velocity + response_rate * (steady_state - current)
        - lataccel += velocity

        Accounts for lag: new steer doesn't affect lataccel until lag steps later.
        """
        costs = []
        for steer in steer_candidates:
            lag = self._compute_lag_steps(abs(steer))
            total_cost = 0.0
            prev_la = current_la
            pred_la = current_la
            pred_velocity = current_velocity  # Start with current rate of change

            for h in range(min(self.mpc_horizon, len(future_plan.lataccel))):
                # Get future target and roll
                future_target = future_plan.lataccel[h] if h < len(future_plan.lataccel) else future_plan.lataccel[-1]
                future_roll = future_plan.roll_lataccel[h] if h < len(future_plan.roll_lataccel) else roll_lataccel

                # Before lag: response still based on previous steer
                # After lag: response based on new steer
                if h < lag:
                    effective_steer = prev_steer
                else:
                    effective_steer = steer

                # Second-order dynamics: mass-spring-damper model
                # steady_state is where we'd settle if steer held constant
                steady_state = future_roll + self.gain * effective_steer

                # Force toward steady state
                force = steady_state - pred_la

                # Update velocity with momentum and force
                pred_velocity = self.momentum * pred_velocity + self.response_rate * force

                # Clamp velocity to MAX_ACC_DELTA
                pred_velocity = np.clip(pred_velocity, -MAX_ACC_DELTA, MAX_ACC_DELTA)

                # Update position
                pred_la += pred_velocity

                # Lataccel error cost (weighted by 50 like in tinyphysics)
                la_error = (future_target - pred_la) ** 2 * 50

                # Jerk cost (rate of change of lataccel)
                jerk = (pred_velocity / 0.1) ** 2

                total_cost += la_error + self.jerk_weight * jerk
                prev_la = pred_la

            costs.append(total_cost)

        return np.array(costs)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel

        # Compute current velocity (rate of change of lataccel)
        # Initialize properly to avoid huge initial velocity
        if self.prev_lataccel is None:
            current_velocity = 0.0
            self.prev_lataccel = current_lataccel
        else:
            current_velocity = current_lataccel - self.prev_lataccel

        self.current_lataccel = current_lataccel

        if self.gain is None:
            self.gain = self.get_gain_lut(v_ego)

        # Compute feedforward as baseline
        lag_steps = self._compute_lag_steps(abs(self.prev_u_cmd))
        effective_target = target_lataccel
        effective_roll = roll_lataccel
        if lag_steps > 0 and future_plan.lataccel and len(future_plan.lataccel) > lag_steps:
            effective_target = future_plan.lataccel[lag_steps - 1]
            if future_plan.roll_lataccel and len(future_plan.roll_lataccel) > lag_steps:
                effective_roll = future_plan.roll_lataccel[lag_steps - 1]

        turning_lataccel = effective_target - effective_roll
        u_ff = turning_lataccel / max(self.gain, 1e-3)

        # MPC: try candidates around feedforward estimate
        steer_candidates = np.linspace(
            u_ff - self.mpc_steer_range,
            u_ff + self.mpc_steer_range,
            self.mpc_candidates
        )
        steer_candidates = np.clip(steer_candidates, -2.0, 2.0)

        # Evaluate costs and pick best (now with momentum model)
        if future_plan.lataccel and len(future_plan.lataccel) > 0:
            costs = self._mpc_cost(steer_candidates, current_lataccel, current_velocity, future_plan, roll_lataccel, self.prev_u_cmd)
            best_idx = np.argmin(costs)
            u_cmd = steer_candidates[best_idx]
        else:
            u_cmd = u_ff

        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        self.cmd_history.append(u_cmd)
        if len(self.cmd_history) > 50:
            self.cmd_history.pop(0)

        # Online gain learning (same as before)
        turning_now = target_lataccel - roll_lataccel
        turning_actual = current_lataccel - roll_lataccel
        if (
            self.step_count >= CONTROL_START_IDX
            and abs(turning_now) > self.turning_threshold
            and abs(u_cmd) > self.min_command
        ):
            update_weight = 1.0  # No PID, so full weight on FF
            lr = self.base_lr * update_weight / (1.0 + abs(self.prev_u_rate))
            lag_index = min(lag_steps, len(self.cmd_history) - 1)
            u_update = self.cmd_history[-1 - lag_index]
            gain_error = turning_actual - (self.gain * u_update)
            self.gain += lr * gain_error * u_update
            self.gain = float(np.clip(self.gain, self.min_gain, self.max_gain))

        # Online learning for dynamics params (momentum, response_rate)
        # Compare predicted velocity to actual velocity
        if self.step_count >= CONTROL_START_IDX and self.prev_lataccel is not None:
            # What we predicted the velocity would be
            steady_state = roll_lataccel + self.gain * self.prev_u_cmd
            force = steady_state - self.prev_lataccel
            predicted_velocity = self.momentum * self.prev_velocity + self.response_rate * force

            # Actual velocity
            velocity_error = current_velocity - predicted_velocity

            # Gradient descent updates
            # d_error/d_momentum = -prev_velocity, d_error/d_response = -force
            self.momentum += self.dynamics_lr * velocity_error * self.prev_velocity
            self.response_rate += self.dynamics_lr * velocity_error * force

            # Clamp to reasonable ranges
            self.momentum = float(np.clip(self.momentum, 0.0, 0.95))
            self.response_rate = float(np.clip(self.response_rate, 0.05, 0.8))

        self.prev_u_rate = u_cmd - self.prev_u_cmd
        self.prev_u_cmd = u_cmd
        self.prev_lataccel = current_lataccel
        self.prev_velocity = current_velocity

        error = target_lataccel - current_lataccel
        self._log = {
            'learn_gain': self.gain,
            'learn_lag_steps': lag_steps,
            'learn_lr': self.base_lr,
            'learn_error': error,
            'mpc_u_ff': u_ff,
            'mpc_u_cmd': u_cmd,
            'mpc_velocity': current_velocity,
            'mpc_momentum': self.momentum,
            'mpc_response_rate': self.response_rate,
        }

        return u_cmd
