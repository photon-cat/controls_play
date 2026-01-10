#!/usr/bin/env python3
"""
Linearize the relationship between steer_command and lat_accel.

Generates scenarios at different velocities (1-10 m/s), runs simulations
with a sweep controller, and fits a model mapping (velocity, steer_command) -> lat_accel.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
import importlib

def generate_scenario(velocity, output_path, duration_steps=500, roll_deg=0.0):
    """Generate a constant velocity scenario for linearization."""
    DT = 0.1
    WARMUP_STEPS = 100

    total_steps = WARMUP_STEPS + duration_steps

    t = np.arange(total_steps) * DT
    vEgo = np.full(total_steps, velocity)
    aEgo = np.zeros(total_steps)
    roll_rad = np.deg2rad(roll_deg)
    roll = np.full(total_steps, roll_rad)
    steerCommand = np.full(total_steps, np.nan)
    steerCommand[:WARMUP_STEPS] = 0.0
    targetLateralAcceleration = np.zeros(total_steps)

    df = pd.DataFrame({
        't': t,
        'vEgo': vEgo,
        'aEgo': aEgo,
        'roll': roll,
        'targetLateralAcceleration': targetLateralAcceleration,
        'steerCommand': steerCommand
    })

    df.to_csv(output_path, index=False)
    return output_path


def run_linearization(model_path, velocities, output_dir, roll_deg=0.0):
    """Run linearization for given velocities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Load sweep controller
    controller_module = importlib.import_module('controllers.sweep')

    for v in velocities:
        print(f"Running linearization at v={v:.1f} m/s, roll={roll_deg:.1f}°...")

        # Generate scenario
        scenario_path = output_dir / f"scenario_v{v:.1f}_roll{roll_deg:.1f}.csv"
        generate_scenario(v, scenario_path, roll_deg=roll_deg)

        # Create model and controller
        model = TinyPhysicsModel(model_path, debug=False)
        controller = controller_module.Controller()

        # Run simulation
        sim = TinyPhysicsSimulator(model, str(scenario_path), controller=controller, debug=False)
        sim.rollout()

        # Collect data after control starts
        for i in range(CONTROL_START_IDX, len(sim.action_history)):
            all_data.append({
                'velocity': v,
                'roll_deg': roll_deg,
                'steer_command': sim.action_history[i],
                'lat_accel': sim.current_lataccel_history[i],
                'target_lat_accel': sim.target_lataccel_history[i],
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / 'linearization_data.csv', index=False)

    return df


def run_roll_sweep(model_path, velocity, roll_values, output_dir):
    """Run linearization for different roll values at fixed velocity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Load sweep controller
    controller_module = importlib.import_module('controllers.sweep')

    for roll_deg in roll_values:
        print(f"Running linearization at v={velocity:.1f} m/s, roll={roll_deg:.1f}°...")

        # Generate scenario
        scenario_path = output_dir / f"scenario_v{velocity:.1f}_roll{roll_deg:.1f}.csv"
        generate_scenario(velocity, scenario_path, roll_deg=roll_deg)

        # Create model and controller
        model = TinyPhysicsModel(model_path, debug=False)
        controller = controller_module.Controller()

        # Run simulation
        sim = TinyPhysicsSimulator(model, str(scenario_path), controller=controller, debug=False)
        sim.rollout()

        # Collect data after control starts
        for i in range(CONTROL_START_IDX, len(sim.action_history)):
            all_data.append({
                'velocity': velocity,
                'roll_deg': roll_deg,
                'steer_command': sim.action_history[i],
                'lat_accel': sim.current_lataccel_history[i],
                'target_lat_accel': sim.target_lataccel_history[i],
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / 'roll_sweep_data.csv', index=False)

    return df


def fit_per_roll(df, velocity):
    """Fit linear model per roll value."""
    roll_values = sorted(df['roll_deg'].unique())
    gains = []

    print(f"\nPer-roll linear fits at v={velocity:.1f} m/s:")
    print(f"{'Roll (deg)':<15} {'Gain':<15} {'Offset':<15} {'R²':<10}")
    print("-" * 55)

    for roll_deg in roll_values:
        r_df = df[df['roll_deg'] == roll_deg]
        u = r_df['steer_command'].values
        a = r_df['lat_accel'].values

        # Filter near-zero steering
        mask = np.abs(u) > 0.1
        if mask.sum() < 10:
            continue

        u_fit = u[mask]
        a_fit = a[mask]

        # Linear fit with offset: a = gain * u + offset
        A = np.vstack([u_fit, np.ones(len(u_fit))]).T
        result = np.linalg.lstsq(A, a_fit, rcond=None)
        gain, offset = result[0]

        # R²
        a_pred = gain * u_fit + offset
        ss_res = np.sum((a_fit - a_pred) ** 2)
        ss_tot = np.sum((a_fit - np.mean(a_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        gains.append({'roll_deg': roll_deg, 'gain': gain, 'offset': offset, 'r2': r2})
        print(f"{roll_deg:<15.1f} {gain:<15.4f} {offset:<15.4f} {r2:<10.4f}")

    return pd.DataFrame(gains)


def fit_model(df):
    """Fit a model: lat_accel = k * v^2 * steer_command + offset"""
    from scipy.optimize import curve_fit

    # Model: lat_accel = k * v^n * steer_command
    def model_func(X, k, n):
        v, u = X
        return k * (v ** n) * u

    # Extract data
    v = df['velocity'].values
    u = df['steer_command'].values
    a = df['lat_accel'].values

    # Filter out near-zero steering (noisy)
    mask = np.abs(u) > 0.1
    v_fit = v[mask]
    u_fit = u[mask]
    a_fit = a[mask]

    # Fit
    try:
        popt, pcov = curve_fit(model_func, (v_fit, u_fit), a_fit, p0=[0.1, 2.0], maxfev=10000)
        k, n = popt

        # Calculate R²
        a_pred = model_func((v_fit, u_fit), k, n)
        ss_res = np.sum((a_fit - a_pred) ** 2)
        ss_tot = np.sum((a_fit - np.mean(a_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"\nFitted model: lat_accel = {k:.6f} * v^{n:.3f} * steer_command")
        print(f"R² = {r2:.4f}")

        return {'k': k, 'n': n, 'r2': r2}
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def fit_per_velocity(df):
    """Fit linear model per velocity: lat_accel = gain(v) * steer_command"""
    velocities = sorted(df['velocity'].unique())
    gains = []

    print("\nPer-velocity linear fits:")
    print(f"{'Velocity (m/s)':<15} {'Gain':<15} {'R²':<10}")
    print("-" * 40)

    for v in velocities:
        v_df = df[df['velocity'] == v]
        u = v_df['steer_command'].values
        a = v_df['lat_accel'].values

        # Filter near-zero
        mask = np.abs(u) > 0.1
        if mask.sum() < 10:
            continue

        u_fit = u[mask]
        a_fit = a[mask]

        # Linear fit through origin: a = gain * u
        gain = np.sum(u_fit * a_fit) / np.sum(u_fit ** 2)

        # R²
        a_pred = gain * u_fit
        ss_res = np.sum((a_fit - a_pred) ** 2)
        ss_tot = np.sum((a_fit - np.mean(a_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        gains.append({'velocity': v, 'gain': gain, 'r2': r2})
        print(f"{v:<15.1f} {gain:<15.4f} {r2:<10.4f}")

    return pd.DataFrame(gains)


def plot_results(df, gains_df, output_dir):
    """Plot linearization results."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Scatter of all data
    ax = axes[0, 0]
    velocities = sorted(df['velocity'].unique())
    for v in velocities:
        v_df = df[df['velocity'] == v]
        ax.scatter(v_df['steer_command'], v_df['lat_accel'], alpha=0.3, label=f'v={v:.0f}', s=5)
    ax.set_xlabel('Steer Command')
    ax.set_ylabel('Lateral Acceleration (m/s²)')
    ax.set_title('Steer Command vs Lat Accel (all velocities)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Gain vs velocity
    ax = axes[0, 1]
    ax.plot(gains_df['velocity'], gains_df['gain'], 'bo-', markersize=8)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Gain (lat_accel / steer_cmd)')
    ax.set_title('Steering Gain vs Velocity')
    ax.grid(True, alpha=0.3)

    # Plot 3: R² vs velocity
    ax = axes[1, 0]
    ax.bar(gains_df['velocity'], gains_df['r2'], width=0.6)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('R²')
    ax.set_title('Linear Fit Quality (R²) vs Velocity')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 4: Gain * v^2 (should be ~constant for bicycle model)
    ax = axes[1, 1]
    gains_df['gain_normalized'] = gains_df['gain'] / (gains_df['velocity'] ** 2)
    ax.plot(gains_df['velocity'], gains_df['gain_normalized'], 'go-', markersize=8)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Gain / v²')
    ax.set_title('Normalized Gain (should be ~constant)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'linearization_results.png', dpi=150)
    print(f"\nPlot saved to {output_dir / 'linearization_results.png'}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linearize steer_command to lat_accel')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx', help='Path to physics model')
    parser.add_argument('--v_min', type=float, default=1.0, help='Min velocity (m/s)')
    parser.add_argument('--v_max', type=float, default=10.0, help='Max velocity (m/s)')
    parser.add_argument('--v_step', type=float, default=1.0, help='Velocity step (m/s)')
    parser.add_argument('--output_dir', type=str, default='linearization_output', help='Output directory')
    parser.add_argument('--mode', type=str, default='velocity', choices=['velocity', 'roll'], help='Sweep mode')
    parser.add_argument('--velocity', type=float, default=10.0, help='Fixed velocity for roll sweep (m/s)')
    parser.add_argument('--roll_values', type=str, default='0,2,4,6,8,10', help='Comma-separated roll values (deg)')
    args = parser.parse_args()

    if args.mode == 'roll':
        # Roll sweep mode
        roll_values = [float(x) for x in args.roll_values.split(',')]
        print(f"Roll sweep at v={args.velocity} m/s, rolls: {roll_values}°")

        df = run_roll_sweep(args.model_path, args.velocity, roll_values, args.output_dir)
        gains_df = fit_per_roll(df, args.velocity)
        gains_df.to_csv(Path(args.output_dir) / 'roll_gains.csv', index=False)

        # Plot roll results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        for roll_deg in roll_values:
            r_df = df[df['roll_deg'] == roll_deg]
            ax.scatter(r_df['steer_command'], r_df['lat_accel'], alpha=0.3, label=f'roll={roll_deg}°', s=5)
        ax.set_xlabel('Steer Command')
        ax.set_ylabel('Lateral Acceleration (m/s²)')
        ax.set_title(f'Steer Command vs Lat Accel (v={args.velocity} m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(gains_df['roll_deg'], gains_df['gain'], 'bo-', markersize=8)
        ax.set_xlabel('Roll (deg)')
        ax.set_ylabel('Gain')
        ax.set_title('Gain vs Roll')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(gains_df['roll_deg'], gains_df['offset'], 'ro-', markersize=8)
        ax.set_xlabel('Roll (deg)')
        ax.set_ylabel('Offset (m/s²)')
        ax.set_title('Offset vs Roll')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / 'roll_sweep_results.png', dpi=150)
        print(f"\nPlot saved to {Path(args.output_dir) / 'roll_sweep_results.png'}")
        plt.show()

    else:
        # Velocity sweep mode
        velocities = np.arange(args.v_min, args.v_max + args.v_step/2, args.v_step)
        print(f"Linearizing at velocities: {velocities}")

        # Run linearization
        df = run_linearization(args.model_path, velocities, args.output_dir)

        # Fit models
        global_fit = fit_model(df)
        gains_df = fit_per_velocity(df)

        # Save gains
        gains_df.to_csv(Path(args.output_dir) / 'gains.csv', index=False)

        # Plot
        plot_results(df, gains_df, args.output_dir)
