#!/usr/bin/env python3
"""
Test autotune concept:
1. Run first 100 timesteps (labeled data) to get initial state
2. Create synthetic test scenario: velocity sweep with sine steering
3. Run tinyphysics to characterize vehicle response
4. Fit gain LUT from results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator,
    CONTROL_START_IDX, ACC_G, CONTEXT_LENGTH
)
from controllers import BaseController


class DataCaptureController(BaseController):
    """Controller that captures data during labeled phase, then uses PID to characterize."""

    def __init__(self):
        self.captured_data = []
        self.step = 0

        # PID gains for characterization
        self.kp = 0.3
        self.ki = 0.1
        self.kd = 0.0

        # PID state
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Target lataccel sweep params
        self.sweep_amplitude = 2.0  # m/s^2
        self.sweep_freq = 0.2  # Hz (slow for good tracking)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step += 1

        # Capture during labeled phase
        if self.step <= CONTROL_START_IDX:
            self.captured_data.append({
                'step': self.step,
                'target_lataccel': target_lataccel,
                'current_lataccel': current_lataccel,
                'v_ego': state.v_ego,
                'a_ego': state.a_ego,
                'roll_lataccel': state.roll_lataccel,
            })
            return 0.0  # Will be overwritten by ground truth

        # After labeled: PID tracks sine wave target lataccel
        t = (self.step - CONTROL_START_IDX) / 10.0  # time in seconds
        synthetic_target = self.sweep_amplitude * np.sin(2 * np.pi * self.sweep_freq * t)

        # PID control
        error = synthetic_target - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10, 10)
        error_diff = error - self.prev_error
        self.prev_error = error

        u = self.kp * error + self.ki * self.error_integral + self.kd * error_diff
        u = np.clip(u, -2.0, 2.0)

        return u


def create_synthetic_scenario(labeled_df, v_avg, v_range=5.0, duration_steps=400):
    """
    Create synthetic scenario CSV for testing.
    - Starts at v_avg, sweeps to v_avg+v_range, then to v_avg-v_range, back to v_avg
    - Zero roll for simplicity
    - Target lataccel = 0 (we're just characterizing response)
    """
    n_total = CONTROL_START_IDX + duration_steps

    # Copy labeled data for first 100 steps
    rows = []
    for i in range(min(CONTROL_START_IDX, len(labeled_df))):
        row = labeled_df.iloc[i]
        rows.append({
            'vEgo': row['vEgo'],
            'aEgo': row['aEgo'],
            'roll': row['roll'],
            'targetLateralAcceleration': row['targetLateralAcceleration'],
            'steerCommand': row['steerCommand'],
        })

    # Synthetic scenario for steps 100-500
    v_min = max(1.0, v_avg - v_range)
    v_max = min(40.0, v_avg + v_range)

    for i in range(duration_steps):
        t = i / 10.0  # time in seconds

        # Velocity profile: sweep up then down
        # Period = 40 seconds (400 steps)
        phase = (i / duration_steps) * 2 * np.pi
        v = v_avg + v_range * np.sin(phase)
        v = np.clip(v, v_min, v_max)

        # Acceleration to achieve velocity change
        if i > 0:
            a = (v - rows[-1]['vEgo']) * 10  # dv/dt
        else:
            a = 0

        rows.append({
            'vEgo': v,
            'aEgo': np.clip(a, -3, 3),
            'roll': 0.0,  # No roll for clean characterization
            'targetLateralAcceleration': 0.0,
            'steerCommand': 0.0,  # Placeholder
        })

    return pd.DataFrame(rows)


def run_characterization(model_path, data_path, output_dir):
    """Run characterization on a single segment."""

    # Load original data
    original_df = pd.read_csv(data_path)
    print(f"Loaded {data_path}: {len(original_df)} rows")

    # Get avg velocity from labeled data
    labeled_v = original_df['vEgo'].iloc[:CONTROL_START_IDX]
    v_avg = labeled_v.mean()
    v_min_labeled = labeled_v.min()
    v_max_labeled = labeled_v.max()
    print(f"Labeled data: v_avg={v_avg:.1f}, v_range=[{v_min_labeled:.1f}, {v_max_labeled:.1f}]")

    # Create synthetic scenario
    synthetic_df = create_synthetic_scenario(original_df, v_avg, v_range=5.0)

    # Save synthetic scenario
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    synthetic_path = output_dir / "synthetic_scenario.csv"
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Created synthetic scenario: {synthetic_path}")

    # Run simulation with sine sweep controller
    model = TinyPhysicsModel(model_path, debug=False)
    controller = DataCaptureController()

    sim = TinyPhysicsSimulator(model, str(synthetic_path), controller=controller, debug=False)
    cost = sim.rollout()

    print(f"Simulation complete. Cost: {cost}")
    print(f"Captured {len(controller.captured_data)} labeled data points")

    # Collect characterization data (steps 100-500)
    # Reconstruct synthetic target from controller params
    sweep_amplitude = controller.sweep_amplitude
    sweep_freq = controller.sweep_freq

    char_data = []
    for i in range(CONTROL_START_IDX, len(sim.action_history)):
        steer = sim.action_history[i]
        lataccel = sim.current_lataccel_history[i]
        v_ego = sim.state_history[i].v_ego
        roll_lataccel = sim.state_history[i].roll_lataccel

        # Reconstruct synthetic target
        t = (i - CONTROL_START_IDX) / 10.0
        synthetic_target = sweep_amplitude * np.sin(2 * np.pi * sweep_freq * t)

        # Turning component (remove roll) - this is what we're actually achieving
        turning_lataccel = lataccel - roll_lataccel

        char_data.append({
            'step': i,
            'v_ego': v_ego,
            'steer': steer,
            'lataccel': lataccel,
            'turning_lataccel': turning_lataccel,
            'roll_lataccel': roll_lataccel,
            'synthetic_target': synthetic_target,
            'tracking_error': synthetic_target - lataccel,
        })

    char_df = pd.DataFrame(char_data)
    print(f"\nCharacterization data: {len(char_df)} points")
    print(f"  v_ego range: [{char_df['v_ego'].min():.1f}, {char_df['v_ego'].max():.1f}]")
    print(f"  steer range: [{char_df['steer'].min():.2f}, {char_df['steer'].max():.2f}]")
    print(f"  lataccel range: [{char_df['lataccel'].min():.2f}, {char_df['lataccel'].max():.2f}]")
    print(f"  tracking error: mean={char_df['tracking_error'].mean():.3f}, std={char_df['tracking_error'].std():.3f}")

    # Fit gains per velocity bin
    print("\n--- Fitting Gains ---")
    gains = fit_gains(char_df)

    # Plot results
    plot_results(char_df, gains, output_dir)

    return gains, char_df


def fit_gains(char_df, v_bin_size=2):
    """Fit gain = lataccel/steer for each velocity bin."""

    results = []

    # Filter for significant steering
    valid = char_df[np.abs(char_df['steer']) > 0.1].copy()
    print(f"Valid points (|steer| > 0.1): {len(valid)}")

    # Group by velocity bins
    v_min = int(valid['v_ego'].min())
    v_max = int(valid['v_ego'].max()) + v_bin_size

    print(f"\n{'V bin':<12} {'N pts':<8} {'Gain':<10} {'R²':<10}")
    print("-" * 45)

    for v_lo in range(v_min, v_max, v_bin_size):
        v_hi = v_lo + v_bin_size
        mask = (valid['v_ego'] >= v_lo) & (valid['v_ego'] < v_hi)
        bin_df = valid[mask]

        if len(bin_df) < 5:
            continue

        u = bin_df['steer'].values
        a = bin_df['turning_lataccel'].values

        # Linear fit through origin: a = gain * u
        denom = np.sum(u ** 2)
        if denom < 0.01:
            continue

        gain = np.sum(u * a) / denom

        # R²
        a_pred = gain * u
        ss_res = np.sum((a - a_pred) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        v_mid = (v_lo + v_hi) / 2
        results.append({
            'v_mid': v_mid,
            'v_lo': v_lo,
            'v_hi': v_hi,
            'n_points': len(bin_df),
            'gain': gain,
            'r2': r2,
        })

        print(f"{v_lo}-{v_hi:<8} {len(bin_df):<8} {gain:<10.3f} {r2:<10.3f}")

    gains_df = pd.DataFrame(results)

    # Print as LUT
    print("\n--- Gain LUT ---")
    print("GAIN_LUT = {")
    for _, row in gains_df.iterrows():
        print(f"    {row['v_mid']}: {row['gain']:.2f},")
    print("}")

    return gains_df


def plot_results(char_df, gains_df, output_dir):
    """Plot characterization results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Steer vs turning lataccel colored by velocity
    ax = axes[0, 0]
    scatter = ax.scatter(char_df['steer'], char_df['turning_lataccel'],
                        c=char_df['v_ego'], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='v_ego (m/s)')
    ax.set_xlabel('Steer Command')
    ax.set_ylabel('Turning Lat Accel (m/s²)')
    ax.set_title('Steer vs Turning LatAccel')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # 2. Gain vs velocity
    ax = axes[0, 1]
    ax.bar(gains_df['v_mid'], gains_df['gain'], width=1.5, alpha=0.7)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Gain (lataccel / steer)')
    ax.set_title('Fitted Gain vs Velocity')
    ax.grid(True, alpha=0.3)

    # 3. R² vs velocity
    ax = axes[1, 0]
    colors = ['green' if r2 > 0.5 else 'red' for r2 in gains_df['r2']]
    ax.bar(gains_df['v_mid'], gains_df['r2'], width=1.5, color=colors, alpha=0.7)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('R²')
    ax.set_title('Fit Quality (R²) vs Velocity')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='orange', linestyle='--', label='R²=0.5 threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Time series - tracking performance
    ax = axes[1, 1]
    ax.plot(char_df['step'], char_df['synthetic_target'], 'b--', label='Target', alpha=0.7)
    ax.plot(char_df['step'], char_df['lataccel'], 'r-', label='Actual LatAccel', alpha=0.7)
    ax.plot(char_df['step'], char_df['steer'], 'g-', label='Steer', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('LatAccel / Steer')
    ax.set_title('Tracking Performance')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'autotune_results.png', dpi=150)
    print(f"\nPlot saved to {output_dir / 'autotune_results.png'}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test autotune concept')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx')
    parser.add_argument('--data_path', type=str, default='data/00000.csv')
    parser.add_argument('--output_dir', type=str, default='autotune_test')
    args = parser.parse_args()

    gains, char_df = run_characterization(args.model_path, args.data_path, args.output_dir)
