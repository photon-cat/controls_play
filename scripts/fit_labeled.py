#!/usr/bin/env python3
"""
Fit steer_command -> lat_accel using labeled data from first 100 timesteps.
Runs simulation to get actual lat_accel produced by each steer_command.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, ACC_G, CONTROL_START_IDX
from controllers import BaseController


class PassthroughController(BaseController):
    """Controller that just returns 0 - we use ground truth in first 100 steps."""
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return 0.0


def load_labeled_data(data_dir, model_path):
    """
    Load labeled data by running simulation on each segment.
    Captures actual lat_accel produced by ground truth steer_commands.
    """
    data_dir = Path(data_dir)
    all_data = []

    model = TinyPhysicsModel(model_path, debug=False)

    for csv_file in sorted(data_dir.glob("*.csv")):
        # Run simulation to get actual lat_accel
        controller = PassthroughController()
        sim = TinyPhysicsSimulator(model, str(csv_file), controller=controller, debug=False)
        sim.rollout()

        # Get data from first 100 timesteps (where ground truth steer is used)
        df = pd.read_csv(csv_file)

        for i in range(CONTROL_START_IDX):
            steer_cmd = sim.action_history[i]
            actual_lataccel = sim.current_lataccel_history[i]
            v_ego = df['vEgo'].iloc[i]
            roll = df['roll'].iloc[i]
            roll_lataccel = np.sin(roll) * ACC_G

            # Turning component = actual lat_accel minus roll contribution
            turning_lataccel = actual_lataccel - roll_lataccel

            all_data.append({
                'v_ego': v_ego,
                'steer_command': steer_cmd,
                'actual_lataccel': actual_lataccel,
                'roll_lataccel': roll_lataccel,
                'turning_lataccel': turning_lataccel,
                'file': csv_file.name,
            })

    return pd.DataFrame(all_data)


def fit_gain_per_velocity_bin(df, v_bins):
    """Fit gain for each velocity bin."""
    results = []

    print(f"\n{'V range (m/s)':<15} {'N points':<10} {'Gain':<12} {'R²':<10}")
    print("-" * 50)

    for i in range(len(v_bins) - 1):
        v_min, v_max = v_bins[i], v_bins[i + 1]
        mask = (df['v_ego'] >= v_min) & (df['v_ego'] < v_max)
        bin_df = df[mask]

        if len(bin_df) < 10:
            print(f"{v_min}-{v_max:<10} {len(bin_df):<10} (insufficient data)")
            continue

        u = bin_df['steer_command'].values
        a = bin_df['turning_lataccel'].values  # Use turning component

        # Filter near-zero steering
        valid = np.abs(u) > 0.05
        if valid.sum() < 10:
            print(f"{v_min}-{v_max:<10} {valid.sum():<10} (insufficient valid data)")
            continue

        u_fit = u[valid]
        a_fit = a[valid]

        # Linear fit through origin: a = gain * u
        gain = np.sum(u_fit * a_fit) / np.sum(u_fit ** 2)

        # R²
        a_pred = gain * u_fit
        ss_res = np.sum((a_fit - a_pred) ** 2)
        ss_tot = np.sum((a_fit - np.mean(a_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        v_mid = (v_min + v_max) / 2
        results.append({
            'v_min': v_min,
            'v_max': v_max,
            'v_mid': v_mid,
            'n_points': valid.sum(),
            'gain': gain,
            'r2': r2,
        })
        print(f"{v_min}-{v_max:<10} {valid.sum():<10} {gain:<12.4f} {r2:<10.4f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Fit gain from labeled data')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                        help='Directories containing CSV segments')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx',
                        help='Path to physics model')
    parser.add_argument('--output', type=str, default='labeled_gains.csv',
                        help='Output CSV file for gains')
    args = parser.parse_args()

    # Load all data
    all_dfs = []
    for data_dir in args.data_dirs:
        print(f"Loading from {data_dir}...")
        df = load_labeled_data(data_dir, args.model_path)
        print(f"  Loaded {len(df)} labeled points")
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} labeled points")

    # Velocity bins - 1 m/s increments
    v_bins = list(range(0, 42, 1))  # 0-1, 1-2, ..., 40-41

    # Fit gains
    gains_df = fit_gain_per_velocity_bin(combined_df, v_bins)

    # Save
    gains_df.to_csv(args.output, index=False)
    print(f"\nSaved gains to {args.output}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot
    ax = axes[0]
    scatter = ax.scatter(combined_df['steer_command'], combined_df['turning_lataccel'],
                         c=combined_df['v_ego'], cmap='viridis', alpha=0.3, s=5)
    plt.colorbar(scatter, ax=ax, label='v_ego (m/s)')
    ax.set_xlabel('Steer Command')
    ax.set_ylabel('Turning Lat Accel (m/s²)')
    ax.set_title('Labeled Data: Steer vs Turning LatAccel')
    ax.grid(True, alpha=0.3)

    # Gain vs velocity
    ax = axes[1]
    ax.plot(gains_df['v_mid'], gains_df['gain'], 'bo-', markersize=8)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Gain (lat_accel / steer_cmd)')
    ax.set_title('Fitted Gain vs Velocity')
    ax.grid(True, alpha=0.3)

    # R² vs velocity
    ax = axes[2]
    ax.bar(gains_df['v_mid'], gains_df['r2'], width=1.5)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('R²')
    ax.set_title('Fit Quality (R²) vs Velocity')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('labeled_fit_results.png', dpi=150)
    print(f"Plot saved to labeled_fit_results.png")
    plt.close()


if __name__ == "__main__":
    main()
