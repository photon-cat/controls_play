"""
Visualize steer_model_rl3.pt predictions vs ground truth.
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from controllers.neural import Controller
from collections import namedtuple

# Load data
df = pd.read_csv('data/00000.csv')
num_steps = 100
ACC_G = 9.81

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

# Initialize controller with RL3 model
model_path = 'models/steer_model_rl3.pt'
controller = Controller(model_path=model_path)

predictions = []
ground_truth = []

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

    # Ground truth (flip sign for right-positive)
    steer_true = -row['steerCommand']

    predictions.append(steer_pred)
    ground_truth.append(steer_true)

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Calculate metrics
mse = np.mean((predictions - ground_truth)**2)
mae = np.mean(np.abs(predictions - ground_truth))
rmse = np.sqrt(mse)
error = predictions - ground_truth

print(f"Model: steer_model_rl3.pt")
print(f"\nMetrics (first {num_steps} steps):")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"\nGround truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
print(f"Prediction range:   [{predictions.min():.3f}, {predictions.max():.3f}]")

# Create comprehensive visualization
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Predictions vs Ground Truth (time series)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(ground_truth, label='Ground Truth', linewidth=2.5, alpha=0.8, color='blue')
ax1.plot(predictions, label='RL3 Predictions', linewidth=2, alpha=0.8, color='red', linestyle='--')
ax1.set_xlabel('Timestep', fontsize=11)
ax1.set_ylabel('Steer Command', fontsize=11)
ax1.set_title(f'steer_model_rl3.pt - Predictions vs Ground Truth (MSE: {mse:.6f}, MAE: {mae:.6f})',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot (predicted vs actual)
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(ground_truth, predictions, alpha=0.6, s=30, color='purple')
# Add diagonal line (perfect prediction)
min_val = min(ground_truth.min(), predictions.min())
max_val = max(ground_truth.max(), predictions.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect fit')
ax2.set_xlabel('Ground Truth', fontsize=11)
ax2.set_ylabel('Predicted', fontsize=11)
ax2.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Plot 3: Error distribution (histogram)
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(error, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero error')
ax3.axvline(x=error.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {error.mean():.4f}')
ax3.set_xlabel('Error (Predicted - True)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Error over time
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(error, color='red', linewidth=1.5, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.fill_between(range(len(error)), error, 0, alpha=0.3, color='red')
ax4.set_xlabel('Timestep', fontsize=11)
ax4.set_ylabel('Error (Predicted - True)', fontsize=11)
ax4.set_title(f'Prediction Error Over Time (MAE: {mae:.6f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.savefig('steer_model_rl3_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: steer_model_rl3_analysis.png")
plt.show()
