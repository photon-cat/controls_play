#!/usr/bin/env python3
"""
Sample segments from data until we have representation at every 1 m/s velocity bin.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import random

def sample_for_coverage(src_dir, dest_dir, v_min=0, v_max=40, min_points_per_bin=50):
    """Sample segments until we have min_points_per_bin at each velocity bin."""
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Track points per velocity bin (1 m/s bins)
    bins = list(range(v_min, v_max + 1))
    points_per_bin = {b: 0 for b in bins}

    # Get all CSV files and shuffle
    files = list(src_path.glob("*.csv"))
    random.shuffle(files)

    copied_files = []

    print(f"Sampling segments to get {min_points_per_bin}+ points per 1 m/s bin from {v_min}-{v_max} m/s")
    print(f"Total available segments: {len(files)}")

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'vEgo' not in df.columns:
                continue

            # Check first 100 rows (labeled data)
            v_ego = df['vEgo'].iloc[:100].values

            # Count points per bin in this segment
            new_points = {b: 0 for b in bins}
            for v in v_ego:
                bin_idx = int(np.floor(v))
                if v_min <= bin_idx < v_max:
                    new_points[bin_idx] += 1

            # Check if this segment adds value (fills gaps)
            adds_value = False
            for b in bins[:-1]:  # Exclude last bin edge
                if points_per_bin[b] < min_points_per_bin and new_points[b] > 0:
                    adds_value = True
                    break

            if adds_value:
                # Copy file
                dest_file = dest_path / f.name
                shutil.copy(f, dest_file)
                copied_files.append(f.name)

                # Update counts
                for b in bins[:-1]:
                    points_per_bin[b] += new_points[b]

        except Exception as e:
            continue

        # Check if we have enough coverage
        coverage = sum(1 for b in bins[:-1] if points_per_bin[b] >= min_points_per_bin)
        total_bins = len(bins) - 1

        if coverage == total_bins:
            print(f"\nFull coverage achieved with {len(copied_files)} segments!")
            break

    # Print coverage summary
    print(f"\nCopied {len(copied_files)} segments to {dest_dir}")
    print(f"\nCoverage summary (points per 1 m/s bin):")
    print(f"{'Velocity':<10} {'Points':<10} {'Status':<10}")
    print("-" * 30)

    gaps = []
    for b in bins[:-1]:
        status = "✓" if points_per_bin[b] >= min_points_per_bin else "⚠️ LOW" if points_per_bin[b] > 0 else "❌ NONE"
        if points_per_bin[b] < min_points_per_bin:
            gaps.append(b)
        print(f"{b}-{b+1} m/s    {points_per_bin[b]:<10} {status}")

    if gaps:
        print(f"\nGaps (< {min_points_per_bin} points): {gaps}")

    return points_per_bin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample segments for velocity coverage")
    parser.add_argument("--src", type=str, default="data", help="Source directory")
    parser.add_argument("--dest", type=str, default="tuning_data/all_velocities", help="Destination directory")
    parser.add_argument("--v_min", type=int, default=0, help="Min velocity")
    parser.add_argument("--v_max", type=int, default=40, help="Max velocity")
    parser.add_argument("--min_points", type=int, default=50, help="Min points per bin")
    args = parser.parse_args()

    sample_for_coverage(args.src, args.dest, args.v_min, args.v_max, args.min_points)
