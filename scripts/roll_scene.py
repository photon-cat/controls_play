#take data segment and produce roll test
#insert from t0 to t99 with labeled steer_command as normal
#target_lataccel = 0 from t>99
#from t100-t199 pull vego to vtarget, determane vtarget based on avg vego of the segment rounded to whole int
#from t200-600 have a sine bank keep vego at vtarget this sweep is from roll min to roll maxx default -10 10
#this is to fab scenes to cancel roll

import argparse
import numpy as np
import pandas as pd

DEL_T = 0.1


def create_roll_scene(input_path: str, output_path: str, roll_min: float = -10, roll_max: float = 10):
    """
    Create a synthetic roll test scene from an existing data segment.

    Args:
        input_path: Path to input CSV segment
        output_path: Path to save output CSV
        roll_min: Minimum roll angle in degrees (default -10)
        roll_max: Maximum roll angle in degrees (default 10)
    """
    # Load the original data
    df = pd.read_csv(input_path)

    # Calculate vtarget as avg vego rounded to whole int
    vtarget = round(df['vEgo'].mean())

    # Get the first 100 rows as-is for context
    output_rows = []

    # t0 to t99: keep original data
    for i in range(min(100, len(df))):
        row = df.iloc[i].copy()
        output_rows.append(row)

    # Calculate total length needed (600 steps total)
    total_steps = 600

    # t100 to t199: transition vego to vtarget, target_lataccel = 0
    for i in range(100, 200):
        t = i * DEL_T
        # Linear interpolation of vego from last value to vtarget
        progress = (i - 100) / 100.0
        if i < len(df):
            v_start = df.iloc[99]['vEgo']
        else:
            v_start = vtarget
        v_ego = v_start + progress * (vtarget - v_start)

        # Small random acceleration during transition
        a_ego = (vtarget - v_start) / (100 * DEL_T)  # constant accel to reach target

        # Keep roll from original data if available, else zero
        if i < len(df):
            roll = df.iloc[i]['roll']
        else:
            roll = 0.0

        row = {
            't': t,
            'vEgo': v_ego,
            'aEgo': a_ego,
            'roll': roll,
            'targetLateralAcceleration': 0.0,  # target is 0 after t99
            'steerCommand': np.nan  # empty - controller takes over
        }
        output_rows.append(pd.Series(row))

    # t200 to t599: sine sweep of roll from roll_min to roll_max
    # Convert degrees to radians
    roll_min_rad = np.radians(roll_min)
    roll_max_rad = np.radians(roll_max)
    roll_amplitude = (roll_max_rad - roll_min_rad) / 2
    roll_center = (roll_max_rad + roll_min_rad) / 2

    sweep_duration = 400  # steps from 200 to 599
    # One full period of sine wave over the sweep
    for i in range(200, total_steps):
        t = i * DEL_T
        sweep_progress = (i - 200) / sweep_duration

        # Sine sweep: goes from center -> max -> center -> min -> center
        roll = roll_center + roll_amplitude * np.sin(2 * np.pi * sweep_progress)

        row = {
            't': t,
            'vEgo': vtarget,
            'aEgo': 0.0,  # constant velocity
            'roll': roll,
            'targetLateralAcceleration': 0.0,  # target is 0, controller must cancel roll
            'steerCommand': np.nan  # empty - controller takes over
        }
        output_rows.append(pd.Series(row))

    # Create output dataframe
    output_df = pd.DataFrame(output_rows)

    # Ensure column order matches original format
    output_df = output_df[['t', 'vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steerCommand']]

    # Save to output path
    output_df.to_csv(output_path, index=False)
    print(f"Created roll scene: {output_path}")
    print(f"  vtarget: {vtarget} m/s")
    print(f"  roll sweep: {roll_min} to {roll_max} degrees")
    print(f"  total steps: {len(output_df)}")

    return output_df


def main():
    parser = argparse.ArgumentParser(description='Create synthetic roll test scenes')
    parser.add_argument('input', type=str, help='Input data segment CSV path')
    parser.add_argument('output', type=str, help='Output CSV path')
    parser.add_argument('--roll_min', type=float, default=-10, help='Minimum roll angle in degrees')
    parser.add_argument('--roll_max', type=float, default=10, help='Maximum roll angle in degrees')
    args = parser.parse_args()

    create_roll_scene(args.input, args.output, args.roll_min, args.roll_max)


if __name__ == '__main__':
    main()
