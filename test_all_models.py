"""
Test all available neural models on labeled data.
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from controllers.neural import Controller
from collections import namedtuple

# Load data
df = pd.read_csv('data/00000.csv')
num_steps = 100
ACC_G = 9.81

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

# Get ground truth steer commands (flip sign for right-positive convention)
ground_truth = -df['steerCommand'].values[:num_steps]

# Find all models
model_dir = Path('models')
model_files = [
    'steer_model.pt',
    'steer_model_1.pt',
    'steer_model_smooth.pt',
    'steer_model_smooth_2.pt',
    'steer_model_rl.pt',
    'steer_model_rl2.pt',
    'steer_model_rl3.pt',
    'steer_model_rl4.pt',
    'steer_model_teacher_rl.pt',
    'steer_model_ppo.pt',
]

results = []

for model_name in model_files:
    model_path = model_dir / model_name
    if not model_path.exists():
        print(f"Skipping {model_name} (not found)")
        continue

    try:
        print(f"\nTesting {model_name}...")

        # Initialize controller with specific model
        controller = Controller(model_path=str(model_path))

        predictions = []

        # Test on first 100 steps
        for i in range(num_steps):
            row = df.iloc[i]

            # Prepare state
            state = State(
                roll_lataccel=np.sin(row['roll']) * ACC_G,
                v_ego=row['vEgo'],
                a_ego=row['aEgo']
            )

            target_lataccel = row['targetLateralAcceleration']
            current_lataccel = target_lataccel

            # Future plan
            future_start = i + 1
            future_end = min(i + 51, len(df))
            future_plan = FuturePlan(
                lataccel=df['targetLateralAcceleration'].values[future_start:future_end].tolist(),
                roll_lataccel=(np.sin(df['roll'].values[future_start:future_end]) * ACC_G).tolist(),
                v_ego=df['vEgo'].values[future_start:future_end].tolist(),
                a_ego=df['aEgo'].values[future_start:future_end].tolist()
            )

            # Predict
            steer_pred = controller.update(target_lataccel, current_lataccel, state, future_plan)
            predictions.append(steer_pred)

        predictions = np.array(predictions)

        # Calculate metrics
        mse = np.mean((predictions - ground_truth)**2)
        mae = np.mean(np.abs(predictions - ground_truth))
        rmse = np.sqrt(mse)

        results.append({
            'model': model_name,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'pred_min': predictions.min(),
            'pred_max': predictions.max(),
        })

        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    except Exception as e:
        print(f"  Error: {e}")
        continue

# Print summary
print("\n" + "="*80)
print("SUMMARY - Sorted by MSE (lower is better)")
print("="*80)

results_df = pd.DataFrame(results).sort_values('mse')
print(results_df.to_string(index=False))

print(f"\nGround truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
print(f"\nBest model: {results_df.iloc[0]['model']} (MSE: {results_df.iloc[0]['mse']:.6f})")
