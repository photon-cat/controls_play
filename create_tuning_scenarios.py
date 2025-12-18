#!/usr/bin/env python3
"""
Create synthetic tuning scenarios by splicing real challenging segments
into controlled test conditions.

Scenario structure:
- 0-10s (100 steps): Warm-up at 10 m/s, straight
- 10-12s (20 steps): Ramp up to real segment initial conditions
- 12-18s (60 steps): Real challenging segment
- 18-20s (20 steps): Ramp down to straight
- Total: 200 steps (20 seconds)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Constants
FPS = 10
DT = 0.1
ACC_G = 9.81

# Phase durations (in steps)
WARMUP_STEPS = 100      # 10 seconds at 10 m/s
RAMP_UP_STEPS = 20      # 2 seconds transition
SEGMENT_STEPS = 60      # 6 seconds real data
RAMP_DOWN_STEPS = 20    # 2 seconds transition
TOTAL_STEPS = WARMUP_STEPS + RAMP_UP_STEPS + SEGMENT_STEPS + RAMP_DOWN_STEPS  # 200 steps

# Warm-up conditions
WARMUP_VEGO = 10.0
WARMUP_AEGO = 0.0
WARMUP_ROLL = 0.0
WARMUP_TARGET_LATACCEL = 0.0

def smooth_ramp(n_steps, start_val, end_val):
    """Generate smooth transition using cosine ramp."""
    t = np.linspace(0, np.pi, n_steps)
    ramp = (1 - np.cos(t)) / 2  # 0 to 1
    return start_val + (end_val - start_val) * ramp

def create_tuning_scenario(
    source_csv: str,
    start_idx: int,
    output_path: str,
    data_dir: str = './data'
):
    """
    Create a tuning scenario from a challenging segment.

    Args:
        source_csv: Source CSV filename
        start_idx: Starting index in source file
        output_path: Output CSV path
        data_dir: Directory containing source data
    """
    # Load source segment
    source_path = Path(data_dir) / source_csv
    df_source = pd.read_csv(source_path)

    # Extract the 6-second window
    end_idx = start_idx + SEGMENT_STEPS
    segment_data = df_source.iloc[start_idx:end_idx]

    if len(segment_data) < SEGMENT_STEPS:
        raise ValueError(f"Not enough data in segment: {len(segment_data)} < {SEGMENT_STEPS}")

    # Get initial and final values of segment for ramping
    initial_vego = segment_data['vEgo'].iloc[0]
    initial_aego = segment_data['aEgo'].iloc[0]
    initial_roll = segment_data['roll'].iloc[0]
    initial_target = segment_data['targetLateralAcceleration'].iloc[0]

    final_vego = segment_data['vEgo'].iloc[-1]
    final_aego = segment_data['aEgo'].iloc[-1]
    final_roll = segment_data['roll'].iloc[-1]
    final_target = segment_data['targetLateralAcceleration'].iloc[-1]

    # Initialize arrays
    t = np.arange(TOTAL_STEPS) * DT
    vEgo = np.zeros(TOTAL_STEPS)
    aEgo = np.zeros(TOTAL_STEPS)
    roll = np.zeros(TOTAL_STEPS)
    targetLateralAcceleration = np.zeros(TOTAL_STEPS)
    steerCommand = np.full(TOTAL_STEPS, np.nan)

    # Phase 1: Warm-up (0-100 steps)
    warmup_end = WARMUP_STEPS
    vEgo[:warmup_end] = WARMUP_VEGO
    aEgo[:warmup_end] = WARMUP_AEGO
    roll[:warmup_end] = WARMUP_ROLL
    targetLateralAcceleration[:warmup_end] = WARMUP_TARGET_LATACCEL
    steerCommand[:warmup_end] = 0.0  # Only warm-up has steer command

    # Phase 2: Ramp up (100-120 steps)
    ramp_up_start = warmup_end
    ramp_up_end = ramp_up_start + RAMP_UP_STEPS

    vEgo[ramp_up_start:ramp_up_end] = smooth_ramp(
        RAMP_UP_STEPS, WARMUP_VEGO, initial_vego
    )
    aEgo[ramp_up_start:ramp_up_end] = smooth_ramp(
        RAMP_UP_STEPS, WARMUP_AEGO, initial_aego
    )
    roll[ramp_up_start:ramp_up_end] = smooth_ramp(
        RAMP_UP_STEPS, WARMUP_ROLL, initial_roll
    )
    targetLateralAcceleration[ramp_up_start:ramp_up_end] = smooth_ramp(
        RAMP_UP_STEPS, WARMUP_TARGET_LATACCEL, initial_target
    )

    # Phase 3: Real challenging segment (120-180 steps)
    segment_start = ramp_up_end
    segment_end = segment_start + SEGMENT_STEPS

    vEgo[segment_start:segment_end] = segment_data['vEgo'].values
    aEgo[segment_start:segment_end] = segment_data['aEgo'].values
    roll[segment_start:segment_end] = segment_data['roll'].values
    targetLateralAcceleration[segment_start:segment_end] = segment_data['targetLateralAcceleration'].values

    # Phase 4: Ramp down (180-200 steps)
    ramp_down_start = segment_end
    ramp_down_end = ramp_down_start + RAMP_DOWN_STEPS

    vEgo[ramp_down_start:ramp_down_end] = smooth_ramp(
        RAMP_DOWN_STEPS, final_vego, WARMUP_VEGO
    )
    aEgo[ramp_down_start:ramp_down_end] = smooth_ramp(
        RAMP_DOWN_STEPS, final_aego, WARMUP_AEGO
    )
    roll[ramp_down_start:ramp_down_end] = smooth_ramp(
        RAMP_DOWN_STEPS, final_roll, WARMUP_ROLL
    )
    targetLateralAcceleration[ramp_down_start:ramp_down_end] = smooth_ramp(
        RAMP_DOWN_STEPS, final_target, WARMUP_TARGET_LATACCEL
    )

    # Create DataFrame
    df_output = pd.DataFrame({
        't': t,
        'vEgo': vEgo,
        'aEgo': aEgo,
        'roll': roll,
        'targetLateralAcceleration': targetLateralAcceleration,
        'steerCommand': steerCommand
    })

    # Save
    df_output.to_csv(output_path, index=False)

    # Print summary
    segment_max_lataccel = np.max(np.abs(targetLateralAcceleration[segment_start:segment_end]))
    segment_avg_speed = np.mean(vEgo[segment_start:segment_end])

    print(f"Created: {output_path}")
    print(f"  Source: {source_csv} @ step {start_idx}")
    print(f"  Segment max |lataccel|: {segment_max_lataccel:.2f} m/s²")
    print(f"  Segment avg speed: {segment_avg_speed:.1f} m/s")
    print(f"  Total duration: {TOTAL_STEPS * DT:.1f}s ({TOTAL_STEPS} steps)")

def main():
    parser = argparse.ArgumentParser(description='Create tuning scenarios from challenging segments')
    parser.add_argument('--input', type=str, default='challenging_segments.txt',
                       help='Input file with segment info')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory with source CSV files')
    parser.add_argument('--output_dir', type=str, default='./tuning_data',
                       help='Output directory for scenarios')
    parser.add_argument('--max_scenarios', type=int, default=10,
                       help='Maximum number of scenarios to create')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Read challenging segments
    print(f"Reading challenging segments from: {args.input}")
    segments = []

    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',')
            if len(parts) >= 3:
                segments.append({
                    'file': parts[0],
                    'start_idx': int(parts[1]),
                    'end_idx': int(parts[2]),
                })

    print(f"Found {len(segments)} challenging segments")

    # Create scenarios
    num_created = 0
    for idx, seg in enumerate(segments[:args.max_scenarios], 1):
        output_path = output_dir / f"tuning_scenario_{idx:02d}.csv"

        try:
            create_tuning_scenario(
                source_csv=seg['file'],
                start_idx=seg['start_idx'],
                output_path=str(output_path),
                data_dir=args.data_dir
            )
            num_created += 1
            print()

        except Exception as e:
            print(f"Error creating scenario {idx}: {e}")
            print()

    print(f"{'='*70}")
    print(f"Successfully created {num_created} tuning scenarios in {args.output_dir}/")
    print(f"{'='*70}")
    print(f"\nTo test a controller on these scenarios:")
    print(f"  python tinyphysics.py --model_path ./models/tinyphysics.onnx \\")
    print(f"                        --data_path {args.output_dir} \\")
    print(f"                        --controller <name>")

if __name__ == '__main__':
    main()
