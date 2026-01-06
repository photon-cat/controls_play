"""
Test the newly trained model.
"""
import pandas as pd
import numpy as np
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

model_path = 'models/steer_model_new.pt'
print(f"Testing {model_path}...")

# Initialize controller with the new model
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

print(f"\nMetrics (first {num_steps} steps on 00000.csv):")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"\nGround truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
print(f"Prediction range:   [{predictions.min():.3f}, {predictions.max():.3f}]")

print(f"\n{'='*60}")
print("Comparison to best existing model (steer_model_rl3.pt):")
print("  RL3 Model - MSE: 0.037477, MAE: 0.168141")
print(f"  New Model - MSE: {mse:.6f}, MAE: {mae:.6f}")
if mse < 0.037477:
    print("  ✓ NEW MODEL IS BETTER!")
else:
    improvement = ((mse - 0.037477) / 0.037477) * 100
    print(f"  ✗ New model is {improvement:.1f}% worse (still training...)")
print(f"{'='*60}")
