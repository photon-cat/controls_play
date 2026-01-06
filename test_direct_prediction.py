"""
Test model using the SAME method as validation (direct prediction, not sequential).
This should match the low validation loss reported during training.
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Load the model directly
model_path = 'models/steer_model_new.pt'
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"Model checkpoint info:")
print(f"  Keys: {list(checkpoint.keys())}")
if 'model_type' in checkpoint:
    print(f"  Model type: {checkpoint['model_type']}")
if 'hidden' in checkpoint:
    print(f"  Hidden dim: {checkpoint['hidden']}")

# Load model architecture
from train_steer_model import SteerTransformer, SteerMLP, CONTEXT_LENGTH, ACC_G

model_type = checkpoint.get('model_type', 'transformer')
hidden = checkpoint.get('hidden', 128)

if model_type == 'mlp':
    model = SteerMLP(hidden=hidden)
else:
    model = SteerTransformer(d_model=hidden)

model.load_state_dict(checkpoint['model_state'])
model.eval()

# Load test data file - same format as training
df = pd.read_csv('data/00000.csv')

# Process same way as training script
valid_mask = ~df['steerCommand'].isna()
df = df[valid_mask].reset_index(drop=True)

roll_lataccel = np.sin(df['roll'].values) * ACC_G
v_ego = df['vEgo'].values
a_ego = df['aEgo'].values
target_lataccel = df['targetLateralAcceleration'].values
steer_command = -df['steerCommand'].values  # flip sign
measured_lataccel = target_lataccel.copy()  # perfect measurement (no noise)

# Build samples like training does
samples = []
for i in range(CONTEXT_LENGTH, min(len(df), 100 + CONTEXT_LENGTH)):  # test on 100 samples
    ctx_slice = slice(i - CONTEXT_LENGTH, i)
    context = np.stack([
        v_ego[ctx_slice],
        a_ego[ctx_slice],
        roll_lataccel[ctx_slice],
        target_lataccel[ctx_slice],
        steer_command[ctx_slice],
        measured_lataccel[ctx_slice],
    ], axis=1)  # shape: (CONTEXT_LENGTH, 6)

    current = np.array([
        v_ego[i],
        a_ego[i],
        roll_lataccel[i],
        target_lataccel[i],
        measured_lataccel[i],
    ])

    target_steer = steer_command[i]
    samples.append((context, current, target_steer))

print(f"\nTesting on {len(samples)} samples (same format as training/validation)")

# Predict on all samples
predictions = []
targets = []

with torch.no_grad():
    for ctx, cur, tgt in samples:
        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        cur_tensor = torch.tensor(cur, dtype=torch.float32).unsqueeze(0)

        pred = model(ctx_tensor, cur_tensor).item()
        predictions.append(pred)
        targets.append(tgt)

predictions = np.array(predictions)
targets = np.array(targets)

# Calculate MSE (same as validation loss)
mse = np.mean((predictions - targets)**2)
mae = np.mean(np.abs(predictions - targets))
rmse = np.sqrt(mse)

print(f"\nDirect Prediction Results (matches validation setup):")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")

print(f"\n{'='*70}")
print("EXPLANATION OF DISCREPANCY:")
print("This test uses DIRECT prediction (like validation in training)")
print("  - Each sample is independent")
print("  - Model sees ground truth context (perfect steer commands)")
print("  - No sequential warmup period")
print()
print("Previous test used SEQUENTIAL prediction (like real controller):")
print("  - First 10 steps use proportional control (warmup)")
print("  - Context contains model's own previous predictions")
print("  - Errors can accumulate")
print()
print("The validation loss during training should match this direct MSE!")
print(f"{'='*70}")
