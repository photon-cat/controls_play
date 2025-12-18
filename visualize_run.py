#!/usr/bin/env python3
"""
Visualize a controller run from saved log data.

Usage:
  python visualize_run.py --log_file logs/my_run.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CONTROL_START_IDX = 100

def load_run_log(log_file):
    """Load a saved run log."""
    data = np.load(log_file, allow_pickle=True)
    return {
        'target_lataccel_history': data['target_lataccel_history'],
        'current_lataccel_history': data['current_lataccel_history'],
        'action_history': data['action_history'],
        'state_history': data['state_history'],
        'costs': data['costs'].item(),
        'metadata': data['metadata'].item() if 'metadata' in data else {}
    }

def plot_run(run_data, save_path=None):
    """Generate the same plots as --debug mode."""
    fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

    target = run_data['target_lataccel_history']
    current = run_data['current_lataccel_history']
    actions = run_data['action_history']
    states = run_data['state_history']
    costs = run_data['costs']
    metadata = run_data['metadata']

    # Plot 1: Lateral Acceleration
    ax[0].plot(target, label='Target lataccel', alpha=0.8)
    ax[0].plot(current, label='Current lataccel', alpha=0.8)
    ax[0].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax[0].legend()
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Lateral Acceleration (m/s²)')
    ax[0].set_title(f'Lateral Acceleration | Cost: {costs["total_cost"]:.2f}')
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Steering Actions
    ax[1].plot(actions, label='Steering Command', color='orange')
    ax[1].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax[1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax[1].legend()
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Steering Command')
    ax[1].set_title('Steering Commands')
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Road Roll Lateral Acceleration
    ax[2].plot(states[:, 0], label='Roll Lateral Acceleration', color='green')
    ax[2].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax[2].legend()
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Lateral Accel due to Road Roll (m/s²)')
    ax[2].set_title('Road Roll Effect')
    ax[2].grid(True, alpha=0.3)

    # Plot 4: Velocity
    ax[3].plot(states[:, 1], label='v_ego', color='purple')
    ax[3].axvline(CONTROL_START_IDX, color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax[3].legend()
    ax[3].set_xlabel('Step')
    ax[3].set_ylabel('Velocity (m/s)')
    ax[3].set_title('Vehicle Velocity')
    ax[3].grid(True, alpha=0.3)

    # Add metadata to title
    if metadata:
        fig.suptitle(f"Controller: {metadata.get('controller', 'Unknown')} | "
                    f"Data: {metadata.get('data_path', 'Unknown')} | "
                    f"LatAccel Cost: {costs['lataccel_cost']:.2f} | "
                    f"Jerk Cost: {costs['jerk_cost']:.2f}",
                    fontsize=10, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    return fig

def print_summary(run_data):
    """Print summary statistics."""
    costs = run_data['costs']
    metadata = run_data['metadata']

    print("\n" + "="*60)
    print("RUN SUMMARY")
    print("="*60)

    if metadata:
        print(f"Controller:  {metadata.get('controller', 'Unknown')}")
        print(f"Data Path:   {metadata.get('data_path', 'Unknown')}")
        print(f"Timestamp:   {metadata.get('timestamp', 'Unknown')}")

    print("\nCosts:")
    print(f"  Lateral Accel Cost: {costs['lataccel_cost']:8.3f}")
    print(f"  Jerk Cost:          {costs['jerk_cost']:8.3f}")
    print(f"  Total Cost:         {costs['total_cost']:8.3f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize controller run from log file')
    parser.add_argument('--log_file', type=str, required=True, help='Path to .npz log file')
    parser.add_argument('--save', type=str, help='Save plot to file instead of showing')
    parser.add_argument('--no_plot', action='store_true', help='Only print summary, no plot')

    args = parser.parse_args()

    # Load the run data
    print(f"Loading run log: {args.log_file}")
    run_data = load_run_log(args.log_file)

    # Print summary
    print_summary(run_data)

    # Plot if requested
    if not args.no_plot:
        plot_run(run_data, save_path=args.save)
