"""
Test neural controller predictions on labeled data.
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from controllers.neural import Controller

# Load data
df = pd.read_csv('data/00000.csv')

# Initialize controller
controller = Controller()

# Use first 100 timesteps
num_steps = 100
predictions = []
ground_truth = []

ACC_G = 9.81

# Simulate the controller step by step
for i in range(num_steps):
    row = df.iloc[i]

    # Prepare state
    from collections import namedtuple
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
    state = State(
        roll_lataccel=np.sin(row['roll']) * ACC_G,
        v_ego=row['vEgo'],
        a_ego=row['aEgo']
    )

    # Get target and current lataccel
    target_lataccel = row['targetLateralAcceleration']
    # For measured lataccel, we'll use the target initially (in real sim it would be from physics)
    current_lataccel = target_lataccel if i == 0 else target_lataccel

    # Future plan (next 50 steps or remaining)
    future_start = i + 1
    future_end = min(i + 51, len(df))
    from collections import namedtuple
    FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
    future_plan = FuturePlan(
        lataccel=df['targetLateralAcceleration'].values[future_start:future_end].tolist(),
        roll_lataccel=(np.sin(df['roll'].values[future_start:future_end]) * ACC_G).tolist(),
        v_ego=df['vEgo'].values[future_start:future_end].tolist(),
        a_ego=df['aEgo'].values[future_start:future_end].tolist()
    )

    # Predict steer command
    steer_pred = controller.update(target_lataccel, current_lataccel, state, future_plan)

    # Get ground truth (note: CSV has left-positive, controller uses right-positive)
    steer_true = -row['steerCommand']

    predictions.append(steer_pred)
    ground_truth.append(steer_true)

# Convert to arrays
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Calculate metrics
mse = np.mean((predictions - ground_truth)**2)
mae = np.mean(np.abs(predictions - ground_truth))
rmse = np.sqrt(mse)

print(f"Model: {controller.model.__class__.__name__}")
print(f"\nPrediction Metrics (first {num_steps} steps):")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"\nGround truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
print(f"Prediction range:   [{predictions.min():.3f}, {predictions.max():.3f}]")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Predictions vs Ground Truth
axes[0].plot(ground_truth, label='Ground Truth', linewidth=2, alpha=0.7)
axes[0].plot(predictions, label='Neural Predictions', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Steer Command')
axes[0].set_title(f'Steer Command Predictions vs Ground Truth (MSE: {mse:.6f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Error over time
error = predictions - ground_truth
axes[1].plot(error, color='red', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Error (Predicted - True)')
axes[1].set_title(f'Prediction Error (MAE: {mae:.6f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_predictions_test.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: neural_predictions_test.png")
plt.show()
