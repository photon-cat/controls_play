"""
Tune bicycle model parameters to match TinyPhysics predictions.

Strategy:
1. Collect predictions from TinyPhysics on test data
2. Optimize bicycle model parameters to minimize prediction error
3. Visualize comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import onnxruntime as ort
from collections import namedtuple

from bicycle_model import BicycleModel, SimplifiedBicycleModel

# Constants
ACC_G = 9.81
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
MAX_ACC_DELTA = 0.5


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
    """Wrapper for TinyPhysics ONNX model"""
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
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(np.clip(x_shifted, -100, 100))  # Clip to avoid overflow
        probs = e_x / np.sum(e_x, axis=axis, keepdims=True)
        # Ensure no NaN/inf
        probs = np.nan_to_num(probs, nan=1.0/e_x.shape[axis])
        # Renormalize
        probs = probs / np.sum(probs, axis=axis, keepdims=True)
        return probs

    def predict(self, input_data: dict, temperature=0.1):
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / temperature, axis=-1)
        assert probs.shape[0] == 1
        assert probs.shape[2] == VOCAB_SIZE
        # Get probabilities for last timestep
        p = probs[0, -1]
        # Ensure valid probability distribution
        p = np.clip(p, 0, 1)
        p = p / np.sum(p)
        # Sample
        sample = np.random.choice(probs.shape[2], p=p)
        return sample

    def get_current_lataccel(self, sim_states, actions, past_preds):
        tokenized_actions = self.tokenizer.encode(past_preds)
        states = np.column_stack([actions, sim_states])
        input_data = {
            'states': np.expand_dims(states, axis=0).astype(np.float32),
            'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
        }
        return self.tokenizer.decode(self.predict(input_data))


def collect_tinyphysics_data(physics_model, csv_files, num_samples=1000):
    """
    Collect prediction data from TinyPhysics model.

    Returns: DataFrame with [v_ego, steer_command, roll_lataccel, predicted_lataccel]
    """
    print(f"Collecting data from TinyPhysics on {len(csv_files)} files...")

    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
    data = []

    for csv_file in csv_files[:10]:  # Limit to first 10 files for speed
        df = pd.read_csv(csv_file)

        # Initialize context
        roll_lataccel = np.sin(df['roll'].values) * ACC_G
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        target_lataccel = df['targetLateralAcceleration'].values
        steer_command = -df['steerCommand'].values  # flip sign

        states = [State(roll_lataccel[i], v_ego[i], a_ego[i]) for i in range(CONTEXT_LENGTH)]
        actions = steer_command[:CONTEXT_LENGTH].tolist()
        lataccels = target_lataccel[:CONTEXT_LENGTH].tolist()

        # Rollout predictions
        for i in range(CONTEXT_LENGTH, min(len(df), 200)):
            # Predict
            sim_states = [[s.roll_lataccel, s.v_ego, s.a_ego] for s in states[-CONTEXT_LENGTH:]]
            sim_actions = actions[-CONTEXT_LENGTH:]
            sim_lataccels = lataccels[-CONTEXT_LENGTH:]

            pred_lataccel = physics_model.get_current_lataccel(sim_states, sim_actions, sim_lataccels)

            # Apply max delta constraint
            if len(lataccels) > 0:
                pred_lataccel = np.clip(
                    pred_lataccel,
                    lataccels[-1] - MAX_ACC_DELTA,
                    lataccels[-1] + MAX_ACC_DELTA
                )

            # Store data point
            data.append({
                'v_ego': v_ego[i],
                'steer_command': steer_command[i],
                'roll_lataccel': roll_lataccel[i],
                'tinyphysics_lataccel': pred_lataccel
            })

            # Update buffers
            states.append(State(roll_lataccel[i], v_ego[i], a_ego[i]))
            actions.append(steer_command[i])
            lataccels.append(pred_lataccel)

            if len(data) >= num_samples:
                break

        if len(data) >= num_samples:
            break

    print(f"Collected {len(data)} data points")
    return pd.DataFrame(data)


def tune_simplified_model(data_df):
    """
    Tune simplified bicycle model parameters.

    Returns: optimal parameters
    """
    print("\nTuning simplified bicycle model...")

    def objective(params):
        steer_gain, steer_ratio = params

        model = SimplifiedBicycleModel(steer_gain=steer_gain, steer_ratio=steer_ratio)

        predictions = []
        for _, row in data_df.iterrows():
            pred = model.predict_lataccel(row['v_ego'], row['steer_command'], row['roll_lataccel'])
            predictions.append(pred)

        predictions = np.array(predictions)
        targets = data_df['tinyphysics_lataccel'].values

        mse = np.mean((predictions - targets) ** 2)
        return mse

    # Initial guess
    x0 = [0.7, 15.0]  # steer_gain, steer_ratio

    # Bounds
    bounds = [(0.1, 2.0), (1.0, 30.0)]

    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    print(f"Optimization result: {result.message}")
    print(f"Optimal steer_gain: {result.x[0]:.4f}")
    print(f"Optimal steer_ratio: {result.x[1]:.4f}")
    print(f"Final MSE: {result.fun:.6f}")

    return result.x


def tune_full_model(data_df):
    """
    Tune full bicycle model parameters.

    Returns: optimal parameters
    """
    print("\nTuning full bicycle model...")

    def objective(params):
        L, a_frac, C_f, C_r, steering_ratio = params
        a = a_frac * L
        b = (1 - a_frac) * L

        model = BicycleModel(L=L, a=a, b=b, C_f=C_f, C_r=C_r, steering_ratio=steering_ratio)

        predictions = []
        for _, row in data_df.iterrows():
            pred = model.predict_lataccel(row['v_ego'], row['steer_command'], row['roll_lataccel'])
            predictions.append(pred)

        predictions = np.array(predictions)
        targets = data_df['tinyphysics_lataccel'].values

        mse = np.mean((predictions - targets) ** 2)
        return mse

    # Initial guess: [L, a_frac, C_f, C_r, steering_ratio]
    x0 = [2.7, 0.45, 80000, 80000, 15.0]

    # Bounds
    bounds = [
        (2.0, 4.0),      # L: wheelbase
        (0.3, 0.7),      # a_frac: CG position (fraction of wheelbase)
        (20000, 150000), # C_f: front cornering stiffness
        (20000, 150000), # C_r: rear cornering stiffness
        (5.0, 30.0)      # steering_ratio
    ]

    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})

    print(f"Optimization result: {result.message}")
    print(f"Optimal L: {result.x[0]:.4f}")
    print(f"Optimal a_frac: {result.x[1]:.4f}")
    print(f"Optimal C_f: {result.x[2]:.1f}")
    print(f"Optimal C_r: {result.x[3]:.1f}")
    print(f"Optimal steering_ratio: {result.x[4]:.4f}")
    print(f"Final MSE: {result.fun:.6f}")

    return result.x


def visualize_comparison(data_df, simple_params, full_params):
    """
    Visualize predictions from both models vs TinyPhysics.
    """
    print("\nGenerating comparison plots...")

    # Build models
    simple_model = SimplifiedBicycleModel(steer_gain=simple_params[0], steer_ratio=simple_params[1])

    L, a_frac = full_params[0], full_params[1]
    full_model = BicycleModel(
        L=L, a=a_frac*L, b=(1-a_frac)*L,
        C_f=full_params[2], C_r=full_params[3],
        steering_ratio=full_params[4]
    )

    # Generate predictions
    simple_preds = []
    full_preds = []

    for _, row in data_df.iterrows():
        simple_preds.append(simple_model.predict_lataccel(row['v_ego'], row['steer_command'], row['roll_lataccel']))
        full_preds.append(full_model.predict_lataccel(row['v_ego'], row['steer_command'], row['roll_lataccel']))

    simple_preds = np.array(simple_preds)
    full_preds = np.array(full_preds)
    tinyphysics_preds = data_df['tinyphysics_lataccel'].values

    # Compute errors
    simple_mse = np.mean((simple_preds - tinyphysics_preds) ** 2)
    full_mse = np.mean((full_preds - tinyphysics_preds) ** 2)

    print(f"\nSimplified Model MSE: {simple_mse:.6f}")
    print(f"Full Model MSE: {full_mse:.6f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Predictions over time
    ax1 = axes[0, 0]
    indices = np.arange(min(200, len(tinyphysics_preds)))
    ax1.plot(indices, tinyphysics_preds[indices], 'k-', linewidth=2, label='TinyPhysics', alpha=0.8)
    ax1.plot(indices, simple_preds[indices], 'b--', linewidth=1.5, label='Simplified Model', alpha=0.7)
    ax1.plot(indices, full_preds[indices], 'r:', linewidth=1.5, label='Full Model', alpha=0.7)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Predicted Lateral Accel (m/s²)')
    ax1.set_title('Predictions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter - Simplified
    ax2 = axes[0, 1]
    ax2.scatter(tinyphysics_preds, simple_preds, alpha=0.3, s=10)
    lim = max(abs(tinyphysics_preds).max(), abs(simple_preds).max())
    ax2.plot([-lim, lim], [-lim, lim], 'k--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('TinyPhysics Prediction')
    ax2.set_ylabel('Simplified Model Prediction')
    ax2.set_title(f'Simplified Model (MSE: {simple_mse:.6f})')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Plot 3: Scatter - Full
    ax3 = axes[1, 0]
    ax3.scatter(tinyphysics_preds, full_preds, alpha=0.3, s=10, color='red')
    ax3.plot([-lim, lim], [-lim, lim], 'k--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('TinyPhysics Prediction')
    ax3.set_ylabel('Full Model Prediction')
    ax3.set_title(f'Full Bicycle Model (MSE: {full_mse:.6f})')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    simple_errors = simple_preds - tinyphysics_preds
    full_errors = full_preds - tinyphysics_preds
    ax4.hist(simple_errors, bins=50, alpha=0.6, label='Simplified', color='blue')
    ax4.hist(full_errors, bins=50, alpha=0.6, label='Full', color='red')
    ax4.axvline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Prediction Error (m/s²)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('bicycle_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: bicycle_model_comparison.png")
    plt.show()


def main():
    # Load TinyPhysics model
    physics_model_path = Path("models/tinyphysics.onnx")
    if not physics_model_path.exists():
        print(f"Error: TinyPhysics model not found at {physics_model_path}")
        return

    physics_model = TinyPhysicsModel(str(physics_model_path))

    # Get data files
    data_files = sorted(Path("data").glob("*.csv"))
    if len(data_files) == 0:
        print("Error: No CSV files found in data/")
        return

    # Collect data from TinyPhysics
    data_df = collect_tinyphysics_data(physics_model, data_files, num_samples=500)

    # Save data
    data_df.to_csv('tinyphysics_collected_data.csv', index=False)
    print(f"Data saved to: tinyphysics_collected_data.csv")

    # Tune models
    simple_params = tune_simplified_model(data_df)
    full_params = tune_full_model(data_df)

    # Visualize
    visualize_comparison(data_df, simple_params, full_params)

    # Print final results
    print("\n" + "="*60)
    print("FINAL TUNED PARAMETERS")
    print("="*60)
    print("\nSimplified Model:")
    print(f"  steer_gain = {simple_params[0]:.6f}")
    print(f"  steer_ratio = {simple_params[1]:.6f}")
    print("\nFull Bicycle Model:")
    print(f"  L = {full_params[0]:.4f} m")
    print(f"  a = {full_params[1] * full_params[0]:.4f} m")
    print(f"  b = {(1 - full_params[1]) * full_params[0]:.4f} m")
    print(f"  C_f = {full_params[2]:.1f} N/rad")
    print(f"  C_r = {full_params[3]:.1f} N/rad")
    print(f"  steering_ratio = {full_params[4]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
