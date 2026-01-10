#!/usr/bin/env python3
"""
Generate synthetic roll test scenario.
- Constant velocity
- Target lataccel = 0
- Roll varies from -10 to +10 degrees in a sweep
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


LABELED_STEPS = 100  # First 100 steps are labeled data for tinyphysics warm-up


def generate_roll_scenario(velocity, roll_max_deg=10, duration_steps=600, output_path=None):
    """
    Generate scenario with constant velocity, zero target lataccel, varying roll.
    First 100 steps are labeled warm-up (no roll, no steer).
    """
    rows = []

    # First 100 steps: labeled warm-up period
    # Steady state - no roll, no steering, just rolling at velocity
    for i in range(LABELED_STEPS):
        t = i / 10.0  # time in seconds
        rows.append({
            't': t,
            'vEgo': velocity,
            'aEgo': 0.0,
            'roll': 0.0,
            'targetLateralAcceleration': 0.0,
            'steerCommand': 0.0,
        })

    # Remaining steps: roll sweep
    for i in range(duration_steps - LABELED_STEPS):
        t = (LABELED_STEPS + i) / 10.0  # time in seconds (continues from labeled)
        t_roll = i / 10.0  # time for roll calculation (starts at 0)

        # Roll sweep: -max -> +max -> -max over duration
        # Use sine wave for smooth transitions
        roll_deg = roll_max_deg * np.sin(2 * np.pi * t_roll / 30)  # 30 second period
        roll_rad = np.radians(roll_deg)

        rows.append({
            't': t,
            'vEgo': velocity,
            'aEgo': 0.0,
            'roll': roll_rad,
            'targetLateralAcceleration': 0.0,  # Target is zero - just counteract roll
            'steerCommand': 0.0,  # Placeholder
        })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Generated roll scenario: {output_path}")
        print(f"  Velocity: {velocity} m/s")
        print(f"  Roll range: ±{roll_max_deg}°")
        print(f"  Duration: {duration_steps} steps ({duration_steps/10:.1f} sec)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate roll test scenario')
    parser.add_argument('--velocity', type=float, default=20.0, help='Constant velocity (m/s)')
    parser.add_argument('--roll_max', type=float, default=10.0, help='Max roll angle (degrees)')
    parser.add_argument('--duration', type=int, default=600, help='Duration in steps')
    parser.add_argument('--output', type=str, default='test_data/roll_test_20ms.csv')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(exist_ok=True)
    generate_roll_scenario(args.velocity, args.roll_max, args.duration, args.output)
