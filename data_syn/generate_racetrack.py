#!/usr/bin/env python3
"""
Generate synthetic racetrack trajectory data for TinyPhysics.

Creates CSV files with realistic racing patterns:
- High-speed straights
- Tight corners with high lateral acceleration
- Chicanes and S-curves
- Variable speed profiles
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_racetrack_trajectory(
    duration: float = 60.0,
    dt: float = 0.1,
    track_type: str = "oval",
    speed_profile: str = "racing"
) -> pd.DataFrame:
    """
    Generate a synthetic racetrack trajectory.
    
    Args:
        duration: Duration in seconds
        dt: Timestep (0.1s = 10Hz)
        track_type: Type of track (oval, circuit, figure8, chicane)
        speed_profile: Speed profile (racing, cruise, aggressive)
    
    Returns:
        DataFrame with trajectory data
    """
    num_steps = int(duration / dt)
    t = np.arange(num_steps) * dt
    
    # Initialize arrays
    v_ego = np.zeros(num_steps)
    a_ego = np.zeros(num_steps)
    roll = np.zeros(num_steps)
    target_lataccel = np.zeros(num_steps)
    steer_command = np.zeros(num_steps)
    
    # Generate based on track type
    if track_type == "oval":
        v_ego, a_ego, roll, target_lataccel, steer_command = _generate_oval(t, speed_profile)
    
    elif track_type == "circuit":
        v_ego, a_ego, roll, target_lataccel, steer_command = _generate_circuit(t, speed_profile)
    
    elif track_type == "figure8":
        v_ego, a_ego, roll, target_lataccel, steer_command = _generate_figure8(t, speed_profile)
    
    elif track_type == "chicane":
        v_ego, a_ego, roll, target_lataccel, steer_command = _generate_chicane(t, speed_profile)
    
    else:
        raise ValueError(f"Unknown track type: {track_type}")
    
    # Create DataFrame
    df = pd.DataFrame({
        't': t,
        'vEgo': v_ego,
        'aEgo': a_ego,
        'roll': roll,
        'targetLateralAcceleration': target_lataccel,
        'steerCommand': steer_command,
    })
    
    return df


def _generate_oval(t, speed_profile):
    """Generate oval track (straights + banked turns)."""
    num_steps = len(t)
    
    # Speed profile
    if speed_profile == "racing":
        base_speed = 45.0  # m/s (~100 mph)
        corner_speed_drop = 15.0
    elif speed_profile == "aggressive":
        base_speed = 50.0
        corner_speed_drop = 20.0
    else:  # cruise
        base_speed = 35.0
        corner_speed_drop = 10.0
    
    # Track cycle: straight (15s) -> turn (8s) -> straight (15s) -> turn (8s)
    cycle_length = 46.0
    phase = (t % cycle_length) / cycle_length
    
    v_ego = np.zeros(num_steps)
    a_ego = np.zeros(num_steps)
    roll = np.zeros(num_steps)
    target_lataccel = np.zeros(num_steps)
    steer_command = np.zeros(num_steps)
    
    for i in range(num_steps):
        p = phase[i]
        
        if p < 0.33:  # Straight 1
            v_ego[i] = base_speed
            a_ego[i] = 0.1 * np.random.randn()
            target_lataccel[i] = 0.05 * np.sin(2 * np.pi * t[i] * 0.2)  # Small corrections
            roll[i] = 0.02 * np.sin(2 * np.pi * t[i] * 0.3)
        
        elif p < 0.5:  # Turn 1 (left)
            turn_phase = (p - 0.33) / 0.17
            v_ego[i] = base_speed - corner_speed_drop * np.sin(turn_phase * np.pi)
            a_ego[i] = -0.8 if turn_phase < 0.3 else 1.2  # Brake then accelerate
            target_lataccel[i] = 2.5 * np.sin(turn_phase * np.pi)  # Peak lat accel in turn
            roll[i] = 0.15 * np.sin(turn_phase * np.pi)  # Banking
        
        elif p < 0.83:  # Straight 2
            v_ego[i] = base_speed
            a_ego[i] = 0.1 * np.random.randn()
            target_lataccel[i] = -0.05 * np.sin(2 * np.pi * t[i] * 0.2)
            roll[i] = -0.02 * np.sin(2 * np.pi * t[i] * 0.3)
        
        else:  # Turn 2 (right)
            turn_phase = (p - 0.83) / 0.17
            v_ego[i] = base_speed - corner_speed_drop * np.sin(turn_phase * np.pi)
            a_ego[i] = -0.8 if turn_phase < 0.3 else 1.2
            target_lataccel[i] = -2.5 * np.sin(turn_phase * np.pi)
            roll[i] = -0.15 * np.sin(turn_phase * np.pi)
        
        # Steering follows lateral acceleration with some dynamics
        steer_command[i] = -target_lataccel[i] / 3.5 + 0.05 * np.random.randn()
    
    # Smooth transitions
    v_ego = _smooth(v_ego, window=5)
    a_ego = np.gradient(v_ego, 0.1)
    
    return v_ego, a_ego, roll, target_lataccel, steer_command


def _generate_circuit(t, speed_profile):
    """Generate road circuit (varying corners and straights)."""
    num_steps = len(t)
    
    base_speed = 40.0 if speed_profile == "racing" else 30.0
    
    v_ego = np.ones(num_steps) * base_speed
    a_ego = np.zeros(num_steps)
    roll = np.zeros(num_steps)
    target_lataccel = np.zeros(num_steps)
    
    # Create interesting corner sequence
    corners = [
        (5, 10, 1.5, "slow right"),
        (15, 20, -2.0, "tight left"),
        (25, 28, 0.8, "fast right"),
        (32, 38, -1.8, "medium left"),
        (42, 46, 1.2, "right"),
        (50, 56, -2.5, "hairpin left"),
    ]
    
    for start, end, lat_accel_peak, desc in corners:
        start_idx = int(start / 0.1)
        end_idx = int(end / 0.1)
        if end_idx > num_steps:
            continue
        
        corner_length = end_idx - start_idx
        corner_phase = np.linspace(0, np.pi, corner_length)
        
        # Apply corner profile
        target_lataccel[start_idx:end_idx] = lat_accel_peak * np.sin(corner_phase)
        
        # Slow down for tight corners
        speed_reduction = abs(lat_accel_peak) * 3
        v_ego[start_idx:end_idx] -= speed_reduction * np.sin(corner_phase)
        
        # Banking
        roll[start_idx:end_idx] = np.sign(lat_accel_peak) * 0.1 * np.sin(corner_phase)
    
    # Add noise
    target_lataccel += 0.05 * np.random.randn(num_steps)
    roll += 0.01 * np.random.randn(num_steps)
    
    # Compute acceleration
    a_ego = np.gradient(v_ego, 0.1)
    
    # Steering
    steer_command = -target_lataccel / 3.0 + 0.03 * np.random.randn(num_steps)
    
    return v_ego, a_ego, roll, target_lataccel, steer_command


def _generate_figure8(t, speed_profile):
    """Generate figure-8 pattern (continuous alternating turns)."""
    num_steps = len(t)
    
    base_speed = 35.0
    omega = 2 * np.pi / 20.0  # 20 second cycle
    
    # Figure-8 creates sinusoidal lateral acceleration
    target_lataccel = 2.0 * np.sin(omega * t)
    
    # Speed varies with lateral load
    v_ego = base_speed - 5 * np.abs(target_lataccel) / 2.0
    v_ego += np.random.randn(num_steps) * 0.5
    
    # Roll follows turns
    roll = 0.08 * np.sin(omega * t + np.pi/4)
    
    # Acceleration from speed changes
    a_ego = np.gradient(v_ego, 0.1)
    
    # Steering
    steer_command = -target_lataccel / 3.2
    
    return v_ego, a_ego, roll, target_lataccel, steer_command


def _generate_chicane(t, speed_profile):
    """Generate chicane pattern (quick left-right-left transitions)."""
    num_steps = len(t)
    
    base_speed = 38.0
    v_ego = np.ones(num_steps) * base_speed
    target_lataccel = np.zeros(num_steps)
    
    # Place chicanes periodically
    chicane_period = 150  # Every 15 seconds
    chicane_length = 30   # 3 seconds each
    
    for i in range(0, num_steps, chicane_period):
        if i + chicane_length < num_steps:
            # Three quick transitions
            third = chicane_length // 3
            
            # Left
            target_lataccel[i:i+third] = 2.5
            # Right
            target_lataccel[i+third:i+2*third] = -3.0
            # Left
            target_lataccel[i+2*third:i+chicane_length] = 2.0
            
            # Speed reduction in chicane
            v_ego[i:i+chicane_length] *= 0.85
    
    # Smooth transitions
    target_lataccel = _smooth(target_lataccel, window=3)
    v_ego = _smooth(v_ego, window=5)
    
    # Roll and acceleration
    roll = 0.06 * target_lataccel / 3.0 + 0.01 * np.random.randn(num_steps)
    a_ego = np.gradient(v_ego, 0.1)
    
    # Steering
    steer_command = -target_lataccel / 3.5
    
    return v_ego, a_ego, roll, target_lataccel, steer_command


def _smooth(data, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic racetrack data")
    parser.add_argument("--track", type=str, default="oval", 
                       choices=["oval", "circuit", "figure8", "chicane"],
                       help="Type of track pattern")
    parser.add_argument("--speed", type=str, default="racing",
                       choices=["racing", "cruise", "aggressive"],
                       help="Speed profile")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration in seconds")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (default: data_syn/track_XXXXX.csv)")
    parser.add_argument("--num-tracks", type=int, default=1,
                       help="Number of tracks to generate")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("data_syn")
    output_dir.mkdir(exist_ok=True)
    
    # Generate tracks
    for i in range(args.num_tracks):
        print(f"Generating track {i+1}/{args.num_tracks}...")
        
        df = generate_racetrack_trajectory(
            duration=args.duration,
            track_type=args.track,
            speed_profile=args.speed
        )
        
        # Determine output filename
        if args.output and args.num_tracks == 1:
            output_file = output_dir / args.output
        else:
            # Find next available number
            existing = list(output_dir.glob("*.csv"))
            next_num = len(existing)
            output_file = output_dir / f"{args.track}_{next_num:05d}.csv"
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        
        # Print stats
        print(f"  Duration: {df['t'].iloc[-1]:.1f}s")
        print(f"  Avg speed: {df['vEgo'].mean():.1f} m/s")
        print(f"  Max lataccel: {df['targetLateralAcceleration'].abs().max():.2f} m/s²")
        print()
    
    print(f"Generated {args.num_tracks} track(s) in {output_dir}/")


if __name__ == "__main__":
    main()

