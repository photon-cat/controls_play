#!/usr/bin/env python3
"""
Generate a synthetic test scenario CSV for testing controllers.

Scenario:
- Car drives at constant 10 m/s on flat road
- First 100 steps: straight (warm-up)
- Next 50 steps: straight (5 seconds at 10Hz)
- Next 50 steps: gentle right turn
- Next 50 steps: gentle left turn
- Total: 250 steps (25 seconds)
"""

import numpy as np
import pandas as pd

# Simulation parameters
FPS = 10  # 10 Hz
DT = 0.1  # 0.1 seconds per step

# Constant vehicle state
V_EGO = 10.0  # m/s
A_EGO = 0.0   # m/s^2 (no acceleration)
ROLL = 0.0    # radians (flat road)

# Number of steps for each phase
WARMUP_STEPS = 100      # First 100 for warm-up
STRAIGHT_STEPS = 50     # 5 seconds straight
RIGHT_TURN_STEPS = 50   # 5 seconds gentle right
LEFT_TURN_STEPS = 50    # 5 seconds gentle left

TOTAL_STEPS = WARMUP_STEPS + STRAIGHT_STEPS + RIGHT_TURN_STEPS + LEFT_TURN_STEPS

# Generate time array
t = np.arange(TOTAL_STEPS) * DT

# Initialize arrays with constants
vEgo = np.full(TOTAL_STEPS, V_EGO)
aEgo = np.full(TOTAL_STEPS, A_EGO)
roll = np.full(TOTAL_STEPS, ROLL)
# steerCommand only for first 100 steps (warm-up), then NaN
steerCommand = np.full(TOTAL_STEPS, np.nan)
steerCommand[:WARMUP_STEPS] = 0.0  # Only first 100 steps have steer command
targetLateralAcceleration = np.zeros(TOTAL_STEPS)

# Define lateral acceleration profiles (gentle maneuvers)
# Gentle turn: 0.5 m/s^2 peak lateral acceleration
# Use smooth sine waves for realistic transitions

def smooth_transition(steps, peak_value):
    """Generate smooth sinusoidal transition."""
    return peak_value * np.sin(np.linspace(0, np.pi, steps))

# Phase indices
warmup_end = WARMUP_STEPS
straight_end = warmup_end + STRAIGHT_STEPS
right_turn_end = straight_end + RIGHT_TURN_STEPS
left_turn_end = right_turn_end + LEFT_TURN_STEPS

# Warm-up + Straight: target lateral accel = 0
# (already initialized to zeros)

# Gentle right turn (positive lateral accel in simulator convention)
targetLateralAcceleration[straight_end:right_turn_end] = smooth_transition(
    RIGHT_TURN_STEPS,
    0.5  # 0.5 m/s^2 peak
)

# Gentle left turn (negative lateral accel)
targetLateralAcceleration[right_turn_end:left_turn_end] = smooth_transition(
    LEFT_TURN_STEPS,
    -0.5  # -0.5 m/s^2 peak (opposite direction)
)

# Create DataFrame
df = pd.DataFrame({
    't': t,
    'vEgo': vEgo,
    'aEgo': aEgo,
    'roll': roll,
    'targetLateralAcceleration': targetLateralAcceleration,
    'steerCommand': steerCommand
})

# Save to CSV
output_file = 'data/test_scenario.csv'
df.to_csv(output_file, index=False)

print(f"Test scenario generated: {output_file}")
print(f"Total steps: {TOTAL_STEPS}")
print(f"Duration: {TOTAL_STEPS * DT:.1f} seconds")
print(f"\nPhases:")
print(f"  0-{warmup_end}: Warm-up (straight, steerCommand=0)")
print(f"  {warmup_end}-{straight_end}: Straight driving")
print(f"  {straight_end}-{right_turn_end}: Gentle right turn (target: 0.5 m/s²)")
print(f"  {right_turn_end}-{left_turn_end}: Gentle left turn (target: -0.5 m/s²)")
