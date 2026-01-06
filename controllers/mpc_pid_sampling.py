"""
Sampling-based MPC controller using PID as center guess.

Strategy:
1. Get PID prediction as center guess
2. Sample 16 steer commands around PID guess
3. Rollout each trajectory using TinyPhysicsModel
4. Pick trajectory with lowest cost (tracking error + jerk)
"""

from . import BaseController
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Constants from tinyphysics
ACC_G = 9.81
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

# MPC parameters
MPC_HORIZON = 10  # steps to look ahead
NUM_SAMPLES = 16  # number of steer samples to try
SAMPLE_RANGE = 0.3  # +/- range around PID guess


class LataccelTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

    def encode(self, value):
        value = np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return np.digitize(value, self.bins, right=True)

    def decode(self, token):
        return self.bins[token]


class TinyPhysicsModel:
    """Learned dynamics model for predicting lateral acceleration"""
    def __init__(self, model_path: str):
        self.tokenizer = LataccelTokenizer()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        provider = 'CPUExecutionProvider'

        with open(model_path, "rb") as f:
            self.ort_session = ort.InferenceSession(f.read(), options, [provider])

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def predict(self, input_data: dict, temperature=1.) -> int:
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        sample = np.random.choice(probs.shape[2], p=probs[0, -1])
        return sample

    def get_current_lataccel(self, sim_states, actions, past_preds):
        """Predict next lateral acceleration given history"""
        tokenized_actions = self.tokenizer.encode(past_preds)
        states = np.column_stack([actions, sim_states])
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
        }
        return self.tokenizer.decode(self.predict(input_data, temperature=0.1))


class SimplePID:
    """Simple PID controller for center guess"""
    def __init__(self):
        self.kp = 0.3
        self.ki = 0.05
        self.kd = 0.1
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel):
        error = target_lataccel - current_lataccel
        self.integral += error * DEL_T
        self.integral = np.clip(self.integral, -1.0, 1.0)  # anti-windup
        derivative = (error - self.prev_error) / DEL_T
        self.prev_error = error

        steer = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class Controller(BaseController):
    def __init__(self):
        # Load physics model
        model_path = Path(__file__).parent.parent / "models" / "tinyphysics.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Physics model not found at {model_path}")

        print(f"Loading MPC-PID-Sampling controller with physics model: {model_path.name}")
        self.physics_model = TinyPhysicsModel(str(model_path))

        # PID for center guess
        self.pid = SimplePID()

        # History buffers
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []

    def reset(self):
        """Reset controller state"""
        self.pid.reset()
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []

    def rollout_trajectory(self, steer_sequence, target_sequence, state_history, action_history, lataccel_history):
        """
        Rollout a trajectory using the physics model.

        Returns: total_cost
        """
        # Copy history to avoid modifying original
        states = state_history.copy()
        actions = action_history.copy()
        lataccels = lataccel_history.copy()

        predicted_lataccels = []
        applied_steers = []

        for i in range(len(steer_sequence)):
            steer = steer_sequence[i]
            target = target_sequence[i]

            # Clip steer to valid range
            steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
            applied_steers.append(steer)

            # Predict next lataccel using physics model
            if len(states) >= CONTEXT_LENGTH:
                sim_states = [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states[-CONTEXT_LENGTH:]]
                sim_actions = actions[-CONTEXT_LENGTH:]
                sim_lataccels = lataccels[-CONTEXT_LENGTH:]

                pred_lataccel = self.physics_model.get_current_lataccel(
                    sim_states, sim_actions, sim_lataccels
                )

                # Apply max delta constraint
                if len(lataccels) > 0:
                    pred_lataccel = np.clip(
                        pred_lataccel,
                        lataccels[-1] - MAX_ACC_DELTA,
                        lataccels[-1] + MAX_ACC_DELTA
                    )
            else:
                # Not enough history, assume perfect tracking
                pred_lataccel = target

            predicted_lataccels.append(pred_lataccel)

            # Update buffers for next prediction
            # (We don't have next state, so we keep last state - simplification)
            actions.append(steer)
            lataccels.append(pred_lataccel)

        # Compute cost
        predicted_lataccels = np.array(predicted_lataccels)
        target_sequence = np.array(target_sequence)
        applied_steers = np.array(applied_steers)

        # Tracking error cost
        tracking_cost = np.sum((predicted_lataccels - target_sequence)**2)

        # Jerk cost (smoothness)
        jerk_cost = np.sum(np.diff(applied_steers)**2) / (DEL_T**2)

        # Total cost
        total_cost = tracking_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

        return total_cost

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get PID center guess
        pid_steer = self.pid.update(target_lataccel, current_lataccel)

        # Add dummy action to history to keep lengths matched during rollout
        # (will be replaced with actual chosen action at the end)
        self.action_history.append(pid_steer)
        self.state_history.append(state)
        self.lataccel_history.append(current_lataccel)

        # Build target sequence for MPC horizon
        if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) >= MPC_HORIZON:
            target_sequence = [target_lataccel] + future_plan.lataccel[:MPC_HORIZON-1]
        else:
            # No future plan, assume constant target
            target_sequence = [target_lataccel] * MPC_HORIZON

        # Sample steer commands around PID guess
        sample_range = np.clip(SAMPLE_RANGE, 0.1, 1.0)
        steer_samples = np.linspace(
            pid_steer - sample_range,
            pid_steer + sample_range,
            NUM_SAMPLES
        )

        # Clip samples to valid range
        steer_samples = np.clip(steer_samples, STEER_RANGE[0], STEER_RANGE[1])

        # Evaluate each sample (constant steer for simplicity)
        best_cost = float('inf')
        best_steer = pid_steer

        for steer_sample in steer_samples:
            # Simple policy: apply same steer for entire horizon
            steer_sequence = [steer_sample] * MPC_HORIZON

            # Rollout and evaluate
            cost = self.rollout_trajectory(
                steer_sequence,
                target_sequence,
                self.state_history,
                self.action_history,
                self.lataccel_history
            )

            if cost < best_cost:
                best_cost = cost
                best_steer = steer_sample

        # Replace dummy action with chosen action
        self.action_history[-1] = best_steer

        # Keep history bounded
        if len(self.state_history) > CONTEXT_LENGTH:
            self.state_history = self.state_history[-CONTEXT_LENGTH:]
            self.action_history = self.action_history[-CONTEXT_LENGTH:]
            self.lataccel_history = self.lataccel_history[-CONTEXT_LENGTH:]

        return best_steer
