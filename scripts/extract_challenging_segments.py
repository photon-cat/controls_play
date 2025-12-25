#!/usr/bin/env python3
"""
Extract challenging segments (rapid turns) from real driving data.

Scans through CSV files to find segments with high lateral acceleration
changes, indicating rapid turns or lane changes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import argparse

def analyze_segment(csv_path: Path) -> Tuple[float, float, float]:
    """
    Analyze a segment for challenging maneuvers.

    Returns:
        (max_lataccel, max_jerk, avg_lataccel_magnitude)
    """
    df = pd.read_csv(csv_path)

    # Get lateral acceleration statistics
    lataccel = df['targetLateralAcceleration'].values

    # Maximum absolute lateral acceleration
    max_lataccel = np.max(np.abs(lataccel))

    # Lateral jerk (rate of change of lateral accel)
    lataccel_diff = np.diff(lataccel)
    jerk = lataccel_diff / 0.1  # Assuming 10 Hz sampling
    max_jerk = np.max(np.abs(jerk))

    # Average magnitude (measures overall turn intensity)
    avg_lataccel_magnitude = np.mean(np.abs(lataccel))

    return max_lataccel, max_jerk, avg_lataccel_magnitude

def find_rapid_turn_windows(csv_path: Path, window_size: int = 60) -> List[Tuple[int, float]]:
    """
    Find 6-second windows (60 steps at 10Hz) with rapid turns.

    Returns:
        List of (start_index, intensity_score) tuples
    """
    df = pd.read_csv(csv_path)
    lataccel = df['targetLateralAcceleration'].values

    if len(lataccel) < window_size:
        return []

    windows = []

    # Slide window through data
    for start_idx in range(0, len(lataccel) - window_size, 10):  # Step by 1 second
        window = lataccel[start_idx:start_idx + window_size]

        # Compute intensity metrics
        max_abs = np.max(np.abs(window))
        jerk = np.diff(window) / 0.1
        max_jerk = np.max(np.abs(jerk))
        avg_magnitude = np.mean(np.abs(window))

        # Combined intensity score
        # Prioritize: high peak + high jerk + sustained turning
        intensity = (
            max_abs * 2.0 +          # Peak lateral accel (weight 2)
            max_jerk * 0.5 +         # Jerk (weight 0.5)
            avg_magnitude * 1.0      # Sustained turning (weight 1)
        )

        # Only consider windows with significant lateral activity
        if max_abs > 0.8 or avg_magnitude > 0.3:
            windows.append((start_idx, intensity))

    return windows

def main():
    parser = argparse.ArgumentParser(description='Extract challenging driving segments')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with CSV files')
    parser.add_argument('--num_segments', type=int, default=20, help='Number of segments to scan')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top challenging windows to extract')
    parser.add_argument('--output', type=str, default='challenging_segments.txt', help='Output file')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob('*.csv'))[:args.num_segments]

    print(f"Scanning {len(csv_files)} segments for challenging maneuvers...")

    all_candidates = []

    for csv_file in csv_files:
        # Overall segment analysis
        max_lat, max_jerk, avg_lat = analyze_segment(csv_file)

        print(f"\n{csv_file.name}:")
        print(f"  Max lateral accel: {max_lat:.3f} m/s²")
        print(f"  Max jerk: {max_jerk:.3f} m/s³")
        print(f"  Avg |lateral accel|: {avg_lat:.3f} m/s²")

        # Find specific windows with rapid turns
        windows = find_rapid_turn_windows(csv_file)

        if windows:
            # Sort by intensity
            windows.sort(key=lambda x: x[1], reverse=True)
            top_windows = windows[:3]  # Top 3 from each file

            print(f"  Found {len(windows)} challenging windows, top 3:")
            for idx, (start, intensity) in enumerate(top_windows, 1):
                end = start + 60
                time_start = start * 0.1
                time_end = end * 0.1
                print(f"    {idx}. Steps {start}-{end} (t={time_start:.1f}-{time_end:.1f}s), intensity={intensity:.2f}")

                all_candidates.append({
                    'file': csv_file.name,
                    'start_idx': start,
                    'end_idx': end,
                    'intensity': intensity,
                    'time_start': time_start
                })

    # Sort all candidates by intensity
    all_candidates.sort(key=lambda x: x['intensity'], reverse=True)

    # Select top N
    top_candidates = all_candidates[:args.top_n]

    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} MOST CHALLENGING SEGMENTS:")
    print(f"{'='*70}")

    with open(args.output, 'w') as f:
        f.write("# Top challenging segments for controller tuning\n")
        f.write("# Format: filename,start_idx,end_idx,intensity,time_start\n\n")

        for idx, seg in enumerate(top_candidates, 1):
            print(f"{idx:2d}. {seg['file']:12s} @ {seg['time_start']:5.1f}s "
                  f"(steps {seg['start_idx']:4d}-{seg['end_idx']:4d}) "
                  f"intensity={seg['intensity']:6.2f}")

            f.write(f"{seg['file']},{seg['start_idx']},{seg['end_idx']},"
                   f"{seg['intensity']:.2f},{seg['time_start']:.1f}\n")

    print(f"\nResults saved to: {args.output}")
    print(f"\nUse these segments to create synthetic tuning scenarios.")

if __name__ == '__main__':
    main()
