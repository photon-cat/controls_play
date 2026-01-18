#after rollout at t>99 set target_lataccel to 0

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path


def set_target_zero(input_path: str, output_path: str):
    """
    Take real data and set targetLateralAcceleration to 0 after t100.
    Keeps all other data (vEgo, aEgo, roll) as-is from original segment.
    """
    df = pd.read_csv(input_path)

    # Set target to 0 for t >= 100 (index 100+)
    df.loc[100:, 'targetLateralAcceleration'] = 0.0

    # Clear steerCommand after t100 (controller takes over)
    df.loc[100:, 'steerCommand'] = np.nan

    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")

    return df


def process_random_sample(input_dir: str, output_dir: str, n: int = 100, seed: int = 42):
    """Randomly sample n files from input directory and process them."""
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob('*.csv'))
    sampled = random.sample(files, min(n, len(files)))
    print(f"Processing {len(sampled)} randomly sampled files...")

    for f in sampled:
        # Output filename: 00128_zerotar.csv
        out_name = f"{f.stem}_zerotar.csv"
        out_file = output_path / out_name
        set_target_zero(str(f), str(out_file))

    print(f"\nDone! Saved {len(sampled)} files to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Set target_lataccel to 0 after t100')
    parser.add_argument('input', type=str, help='Input CSV file or directory')
    parser.add_argument('output', type=str, help='Output CSV file or directory')
    parser.add_argument('-n', type=int, default=100, help='Number of files to sample (default 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        # Single file: output name is basename_zerotar.csv
        out_path = Path(args.output)
        if out_path.is_dir():
            out_file = out_path / f"{input_path.stem}_zerotar.csv"
        else:
            out_file = out_path
        set_target_zero(args.input, str(out_file))
    elif input_path.is_dir():
        process_random_sample(args.input, args.output, args.n, args.seed)
    else:
        print(f"Error: {args.input} not found")


if __name__ == '__main__':
    main()
