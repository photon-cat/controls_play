"""
Convert CSV data to 2D waypoints trajectory.

At each timestep t, compute where the car traveled using:
- v_ego: longitudinal velocity
- a_ego: longitudinal acceleration
- lateral_accel: lateral acceleration
- roll: road banking

Outputs: 2D waypoints (x, y) at each timestep starting from (0, 0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

ACC_G = 9.81
DEL_T = 0.1  # 10 Hz = 0.1 second timestep


def compute_waypoints(df):
    """
    Compute 2D waypoints from vehicle state data.

    Args:
        df: DataFrame with columns [vEgo, aEgo, targetLateralAcceleration, roll]

    Returns:
        waypoints: Nx3 array of [x, y, heading] at each timestep
    """
    # Extract data
    v_ego = df['vEgo'].values
    a_ego = df['aEgo'].values
    target_lataccel = df['targetLateralAcceleration'].values
    roll = df['roll'].values

    # Compute roll contribution to lateral accel
    roll_lataccel = np.sin(roll) * ACC_G

    # Net lateral accel from turning (remove road banking effect)
    turning_lataccel = target_lataccel - roll_lataccel

    # Initialize trajectory
    n_steps = len(df)
    waypoints = np.zeros((n_steps, 3))  # [x, y, heading]

    x, y, heading = 0.0, 0.0, 0.0  # Start at origin facing east (+x)

    for i in range(n_steps):
        # Store current position
        waypoints[i] = [x, y, heading]

        # Get current state
        v = v_ego[i]
        lat_a = turning_lataccel[i]

        # Compute yaw rate from lateral acceleration
        # lateral_accel = v^2 / R = v * yaw_rate
        # Therefore: yaw_rate = lateral_accel / v
        if abs(v) > 0.1:  # Avoid division by zero
            yaw_rate = lat_a / v
        else:
            yaw_rate = 0.0

        # Update heading (integrate yaw rate)
        heading += yaw_rate * DEL_T

        # Update position (integrate velocity in heading direction)
        x += v * np.cos(heading) * DEL_T
        y += v * np.sin(heading) * DEL_T

    return waypoints


def plot_trajectory(waypoints, title="Vehicle Trajectory", save_path=None):
    """Plot the 2D trajectory with heading arrows."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = waypoints[:, 0]
    y = waypoints[:, 1]
    heading = waypoints[:, 2]

    # Plot 1: Full trajectory
    ax1 = axes[0]
    ax1.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, label='Path')
    ax1.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=10)
    ax1.plot(x[-1], y[-1], 'ro', markersize=12, label='End', zorder=10)

    # Add heading arrows (every 50 points)
    arrow_spacing = 50
    for i in range(0, len(waypoints), arrow_spacing):
        dx = np.cos(heading[i]) * 5  # Arrow length
        dy = np.sin(heading[i]) * 5
        ax1.arrow(x[i], y[i], dx, dy, head_width=2, head_length=1.5,
                 fc='red', ec='red', alpha=0.6)

    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Heading over time
    ax2 = axes[1]
    timesteps = np.arange(len(waypoints)) * DEL_T
    ax2.plot(timesteps, np.degrees(heading), 'purple', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Heading (degrees)', fontsize=12)
    ax2.set_title('Heading Angle Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main(args):
    # Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Compute waypoints
    print("\nComputing waypoints...")
    waypoints = compute_waypoints(df)

    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"  Total timesteps: {len(waypoints)}")
    print(f"  Duration: {len(waypoints) * DEL_T:.1f} seconds")
    print(f"  Start position: ({waypoints[0, 0]:.2f}, {waypoints[0, 1]:.2f})")
    print(f"  End position: ({waypoints[-1, 0]:.2f}, {waypoints[-1, 1]:.2f})")

    total_distance = np.sum(np.sqrt(np.diff(waypoints[:, 0])**2 + np.diff(waypoints[:, 1])**2))
    print(f"  Total distance traveled: {total_distance:.2f} m")
    print(f"  Final heading: {np.degrees(waypoints[-1, 2]):.2f} degrees")

    # Save waypoints if requested
    if args.output:
        output_path = Path(args.output)
        np.save(output_path, waypoints)
        print(f"\nWaypoints saved to: {output_path}")

        # Also save as CSV for easy inspection
        csv_output = output_path.with_suffix('.csv')
        waypoints_df = pd.DataFrame(waypoints, columns=['x', 'y', 'heading'])
        waypoints_df['t'] = np.arange(len(waypoints)) * DEL_T
        waypoints_df = waypoints_df[['t', 'x', 'y', 'heading']]
        waypoints_df.to_csv(csv_output, index=False)
        print(f"Waypoints CSV saved to: {csv_output}")

    # Plot if requested
    if args.plot:
        plot_title = f"Trajectory: {csv_path.name}"
        save_path = args.plot if args.plot != "show" else None
        plot_trajectory(waypoints, title=plot_title, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to 2D waypoints")
    parser.add_argument("csv_path", type=str, help="Path to CSV file (e.g., data/00010.csv)")
    parser.add_argument("--output", type=str, help="Output path for waypoints .npy file")
    parser.add_argument("--plot", type=str, nargs='?', const="show",
                       help="Plot trajectory (provide path to save, or 'show' to display)")

    args = parser.parse_args()
    main(args)
