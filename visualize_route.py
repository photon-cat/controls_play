#!/usr/bin/env python3
"""
Visualize route trajectory from CSV data.

Shows:
- Top-down 2D path of the vehicle
- Speed profile along the path
- Lateral acceleration heatmap
- Interactive animation
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.patches as patches


def integrate_trajectory(df):
    """
    Integrate velocity and lateral acceleration to get 2D path.
    
    Returns:
        x, y: Position arrays
        heading: Heading angle array
    """
    dt = 0.1
    num_steps = len(df)
    
    # Initialize
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    heading = np.zeros(num_steps)
    
    # Current state
    vx = df['vEgo'].values[0]
    vy = 0.0
    theta = 0.0  # heading angle
    
    for i in range(1, num_steps):
        # Get lateral acceleration
        lat_accel = df['targetLateralAcceleration'].values[i]
        v_ego = df['vEgo'].values[i]
        
        # Update heading based on lateral acceleration
        # lat_accel = v^2 / R, so angular velocity omega = v / R = lat_accel / v
        if v_ego > 0.1:
            omega = lat_accel / v_ego
        else:
            omega = 0.0
        
        theta += omega * dt
        heading[i] = theta
        
        # Update position
        x[i] = x[i-1] + v_ego * np.cos(theta) * dt
        y[i] = y[i-1] + v_ego * np.sin(theta) * dt
    
    return x, y, heading


def plot_route_static(df, output_file=None):
    """Create static visualization of the route."""
    
    # Integrate trajectory
    x, y, heading = integrate_trajectory(df)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main plot: Top-down view with speed coloring
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Create line segments colored by speed
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize speed for coloring
    speeds = df['vEgo'].values
    lc = LineCollection(segments, cmap='viridis', linewidth=3)
    lc.set_array(speeds)
    lc.set_clim(speeds.min(), speeds.max())
    
    ax_main.add_collection(lc)
    ax_main.autoscale()
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.set_title('Top-Down Route View (colored by speed)', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # Colorbar for speed
    cbar = plt.colorbar(lc, ax=ax_main)
    cbar.set_label('Speed (m/s)', fontsize=11)
    
    # Mark start and end
    ax_main.plot(x[0], y[0], 'go', markersize=15, label='Start', zorder=5)
    ax_main.plot(x[-1], y[-1], 'ro', markersize=15, label='End', zorder=5)
    ax_main.legend(fontsize=11)
    
    # Speed profile over time
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.plot(df['t'], df['vEgo'], 'b-', linewidth=2)
    ax_speed.fill_between(df['t'], 0, df['vEgo'], alpha=0.3)
    ax_speed.set_xlabel('Time (s)', fontsize=11)
    ax_speed.set_ylabel('Speed (m/s)', fontsize=11)
    ax_speed.set_title('Speed Profile', fontsize=12, fontweight='bold')
    ax_speed.grid(True, alpha=0.3)
    
    # Lateral acceleration over time
    ax_lataccel = fig.add_subplot(gs[1, 1])
    ax_lataccel.plot(df['t'], df['targetLateralAcceleration'], 'r-', linewidth=2)
    ax_lataccel.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_lataccel.fill_between(df['t'], 0, df['targetLateralAcceleration'], 
                              alpha=0.3, color='red')
    ax_lataccel.set_xlabel('Time (s)', fontsize=11)
    ax_lataccel.set_ylabel('Lateral Accel (m/s²)', fontsize=11)
    ax_lataccel.set_title('Lateral Acceleration Profile', fontsize=12, fontweight='bold')
    ax_lataccel.grid(True, alpha=0.3)
    
    # Statistics
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')
    
    stats_text = f"""
    ROUTE STATISTICS
    ────────────────────────────────
    Duration:          {df['t'].iloc[-1]:.1f} s
    Total Distance:    {np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum():.1f} m
    
    Speed:
      Average:         {df['vEgo'].mean():.1f} m/s ({df['vEgo'].mean()*2.237:.1f} mph)
      Maximum:         {df['vEgo'].max():.1f} m/s
      Minimum:         {df['vEgo'].min():.1f} m/s
    
    Lateral Accel:
      Maximum:         {df['targetLateralAcceleration'].abs().max():.2f} m/s²
      Mean Absolute:   {df['targetLateralAcceleration'].abs().mean():.2f} m/s²
    
    Longitudinal Accel:
      Maximum:         {df['aEgo'].max():.2f} m/s²
      Minimum:         {df['aEgo'].min():.2f} m/s² (braking)
    """
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', 
                  facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Route Trajectory Visualization', fontsize=16, fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    
    plt.show()


def animate_route(df, speed=1.0):
    """Create animated visualization of the route."""
    
    # Integrate trajectory
    x, y, heading = integrate_trajectory(df)
    
    # Setup figure
    fig, (ax_map, ax_data) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Map axis
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('X Position (m)', fontsize=11)
    ax_map.set_ylabel('Y Position (m)', fontsize=11)
    ax_map.set_title('Live Route View', fontsize=13, fontweight='bold')
    ax_map.grid(True, alpha=0.3)
    
    # Full path (faded)
    ax_map.plot(x, y, 'b-', alpha=0.2, linewidth=1, label='Full path')
    
    # Dynamic elements
    trail_line, = ax_map.plot([], [], 'b-', linewidth=3, label='Trail')
    vehicle_dot, = ax_map.plot([], [], 'ro', markersize=12, label='Vehicle')
    
    # Vehicle direction arrow
    arrow = patches.FancyArrow(0, 0, 0, 0, width=3, head_width=6, 
                               head_length=4, fc='red', ec='darkred')
    arrow_patch = ax_map.add_patch(arrow)
    
    ax_map.legend(fontsize=10)
    
    # Data axis
    ax_data.set_xlabel('Time (s)', fontsize=11)
    ax_data.set_ylabel('Value', fontsize=11)
    ax_data.set_title('Telemetry', fontsize=13, fontweight='bold')
    ax_data.grid(True, alpha=0.3)
    
    speed_line, = ax_data.plot([], [], 'b-', linewidth=2, label='Speed (m/s)')
    lataccel_line, = ax_data.plot([], [], 'r-', linewidth=2, label='Lat Accel (m/s²)')
    current_time, = ax_data.plot([], [], 'k--', linewidth=2, alpha=0.5)
    
    ax_data.legend(fontsize=10)
    
    # Info text
    info_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Animation parameters
    trail_length = 50  # Show last 50 points in trail
    
    def init():
        trail_line.set_data([], [])
        vehicle_dot.set_data([], [])
        speed_line.set_data([], [])
        lataccel_line.set_data([], [])
        current_time.set_data([], [])
        return trail_line, vehicle_dot, speed_line, lataccel_line, current_time, arrow_patch, info_text
    
    def update(frame):
        nonlocal arrow_patch
        
        # Update trail
        start_idx = max(0, frame - trail_length)
        trail_line.set_data(x[start_idx:frame+1], y[start_idx:frame+1])
        
        # Update vehicle position
        vehicle_dot.set_data([x[frame]], [y[frame]])
        
        # Update arrow
        arrow_patch.remove()
        arrow_length = 20
        dx = arrow_length * np.cos(heading[frame])
        dy = arrow_length * np.sin(heading[frame])
        new_arrow = patches.FancyArrow(x[frame], y[frame], dx, dy,
                                       width=3, head_width=8, head_length=6,
                                       fc='red', ec='darkred', zorder=10)
        arrow_patch = ax_map.add_patch(new_arrow)
        
        # Update data plots
        speed_line.set_data(df['t'].values[:frame+1], df['vEgo'].values[:frame+1])
        lataccel_line.set_data(df['t'].values[:frame+1], 
                               df['targetLateralAcceleration'].values[:frame+1])
        
        # Current time line
        t_now = df['t'].values[frame]
        current_time.set_data([t_now, t_now], ax_data.get_ylim())
        
        # Auto-scale data axis
        if frame > 0:
            ax_data.set_xlim(0, df['t'].values[-1])
            ax_data.set_ylim(-3, max(df['vEgo'].max(), 3))
        
        # Update info text
        info_text.set_text(
            f"Time: {t_now:.1f}s\n"
            f"Speed: {df['vEgo'].values[frame]:.1f} m/s\n"
            f"Lat Accel: {df['targetLateralAcceleration'].values[frame]:+.2f} m/s²\n"
            f"Position: ({x[frame]:.0f}, {y[frame]:.0f}) m"
        )
        
        return trail_line, vehicle_dot, speed_line, lataccel_line, current_time, arrow_patch, info_text
    
    # Calculate frame interval for desired playback speed
    frame_interval = (1000 * 0.1) / speed  # milliseconds per frame
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(df), interval=frame_interval,
                        blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description="Visualize route from CSV data")
    parser.add_argument("csv_file", type=str, help="Path to CSV file")
    parser.add_argument("--animate", action="store_true", help="Show animated view")
    parser.add_argument("--speed", type=float, default=1.0, 
                       help="Animation speed multiplier (default: 1.0)")
    parser.add_argument("--output", type=str, help="Save static plot to file")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} data points ({df['t'].iloc[-1]:.1f} seconds)")
    
    if args.animate:
        print(f"Starting animation at {args.speed}x speed...")
        animate_route(df, speed=args.speed)
    else:
        print("Generating static visualization...")
        plot_route_static(df, output_file=args.output)


if __name__ == "__main__":
    main()

