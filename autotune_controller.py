#!/usr/bin/env python3
"""
Auto-tuning system for controllers using relay feedback method.

This script runs the simulator with a relay controller to identify
the system's critical gain (Ku) and period (Pu), then computes
optimal PID gains using Ziegler-Nichols tuning rules.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple, Dict
import matplotlib.pyplot as plt

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import BaseController

class RelayController(BaseController):
    """
    Relay (bang-bang) controller for system identification.

    Oscillates between +/- relay_amplitude to induce limit cycle,
    allowing measurement of ultimate gain and period.
    """
    def __init__(self, relay_amplitude=0.5):
        self.relay_amplitude = relay_amplitude
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel

        # Bang-bang control
        if error > 0:
            return self.relay_amplitude
        else:
            return -self.relay_amplitude

def find_oscillation_parameters(
    errors: np.ndarray,
    outputs: np.ndarray,
    dt: float = 0.1
) -> Tuple[float, float]:
    """
    Find oscillation period and amplitude from relay feedback test.

    Returns:
        (period, amplitude): Ultimate period Pu and error amplitude
    """
    # Find zero crossings to measure period
    zero_crossings = []
    for i in range(1, len(errors)):
        if errors[i-1] * errors[i] < 0:  # Sign change
            zero_crossings.append(i)

    if len(zero_crossings) < 4:
        raise ValueError("Not enough oscillations detected")

    # Measure period (average time between alternate crossings)
    periods = []
    for i in range(2, len(zero_crossings)):
        period = (zero_crossings[i] - zero_crossings[i-2]) * dt
        periods.append(period)

    avg_period = np.mean(periods[-5:])  # Use last 5 for stability

    # Measure error amplitude
    peaks = []
    for i in range(1, len(errors)-1):
        if abs(errors[i]) > abs(errors[i-1]) and abs(errors[i]) > abs(errors[i+1]):
            peaks.append(abs(errors[i]))

    avg_amplitude = np.mean(peaks[-5:]) if peaks else np.std(errors)

    return avg_period, avg_amplitude

def compute_ziegler_nichols_gains(
    Ku: float,
    Pu: float,
    controller_type: str = "PID"
) -> Dict[str, float]:
    """
    Compute PID gains using Ziegler-Nichols ultimate cycling method.

    Args:
        Ku: Ultimate gain (relay amplitude / error amplitude)
        Pu: Ultimate period (oscillation period)
        controller_type: "P", "PI", or "PID"

    Returns:
        Dictionary with kp, ki, kd gains
    """
    if controller_type == "P":
        kp = 0.5 * Ku
        ki = 0.0
        kd = 0.0
    elif controller_type == "PI":
        kp = 0.45 * Ku
        ki = 0.54 * Ku / Pu
        kd = 0.0
    elif controller_type == "PID":
        kp = 0.6 * Ku
        ki = 1.2 * Ku / Pu
        kd = 0.075 * Ku * Pu
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    return {'kp': kp, 'ki': ki, 'kd': kd}

def run_relay_test(
    model_path: str,
    data_path: str,
    relay_amplitude: float = 0.5,
    test_duration: int = 300  # steps
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run relay feedback test to identify system dynamics.

    Returns:
        (time, errors, outputs): Arrays of time, tracking errors, and control outputs
    """
    # Initialize
    model = TinyPhysicsModel(model_path, debug=False)
    relay_controller = RelayController(relay_amplitude=relay_amplitude)
    sim = TinyPhysicsSimulator(model, data_path, controller=relay_controller, debug=False)

    # Run simulation
    errors = []
    outputs = []
    times = []

    # Limit to available data
    max_steps = min(test_duration, len(sim.data) - sim.step_idx - 1)

    for step in range(max_steps):
        if sim.step_idx >= len(sim.data) - 1:
            break

        sim.step()

        if sim.step_idx >= 100:  # After warm-up
            error = sim.target_lataccel_history[-1] - sim.current_lataccel_history[-1]
            errors.append(error)
            outputs.append(sim.action_history[-1])
            times.append(sim.step_idx * 0.1)

    return np.array(times), np.array(errors), np.array(outputs)

def autotune(
    model_path: str,
    data_path: str,
    relay_amplitude: float = 0.5,
    controller_type: str = "PID",
    plot: bool = True
) -> Dict[str, float]:
    """
    Perform automatic tuning to find optimal PID gains.
    """
    print("Running relay feedback test...")
    times, errors, outputs = run_relay_test(model_path, data_path, relay_amplitude)

    print("Analyzing oscillations...")
    Pu, error_amplitude = find_oscillation_parameters(errors, outputs)

    # Ultimate gain = 4 * relay_amplitude / (pi * error_amplitude)
    Ku = (4 * relay_amplitude) / (np.pi * error_amplitude)

    print(f"\nSystem Identification:")
    print(f"  Ultimate Period (Pu): {Pu:.3f} seconds")
    print(f"  Ultimate Gain (Ku):   {Ku:.3f}")
    print(f"  Error Amplitude:      {error_amplitude:.3f}")

    # Compute gains
    gains = compute_ziegler_nichols_gains(Ku, Pu, controller_type)

    print(f"\nZiegler-Nichols {controller_type} Gains:")
    print(f"  Kp: {gains['kp']:.4f}")
    print(f"  Ki: {gains['ki']:.4f}")
    print(f"  Kd: {gains['kd']:.4f}")

    # Apply conservative detuning (ZN tends to be aggressive)
    print(f"\nConservative (0.7x) Gains (recommended):")
    conservative_gains = {k: v * 0.7 for k, v in gains.items()}
    print(f"  Kp: {conservative_gains['kp']:.4f}")
    print(f"  Ki: {conservative_gains['ki']:.4f}")
    print(f"  Kd: {conservative_gains['kd']:.4f}")

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Error plot
        ax1.plot(times, errors, label='Tracking Error')
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(error_amplitude, color='r', linestyle='--', alpha=0.5, label=f'Amplitude: {error_amplitude:.3f}')
        ax1.axhline(-error_amplitude, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (m/s²)')
        ax1.set_title(f'Relay Feedback Test - Period: {Pu:.3f}s, Ku: {Ku:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Control output
        ax2.plot(times, outputs, label='Control Output', color='orange')
        ax2.axhline(relay_amplitude, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(-relay_amplitude, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Output')
        ax2.set_title('Relay Controller Output')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('autotune_relay_test.png', dpi=150)
        print(f"\nPlot saved to: autotune_relay_test.png")
        plt.close()

    return conservative_gains

def main():
    parser = argparse.ArgumentParser(description='Auto-tune PID controller')
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx')
    parser.add_argument('--data_path', type=str, default='./tuning_data/tuning_scenario_01.csv',
                       help='Test data path')
    parser.add_argument('--relay_amplitude', type=float, default=0.5,
                       help='Relay amplitude for identification')
    parser.add_argument('--controller_type', type=str, default='PID',
                       choices=['P', 'PI', 'PID'])
    parser.add_argument('--no_plot', action='store_true')

    args = parser.parse_args()

    gains = autotune(
        args.model_path,
        args.data_path,
        relay_amplitude=args.relay_amplitude,
        controller_type=args.controller_type,
        plot=not args.no_plot
    )

    print("\n" + "="*60)
    print("To use these gains, update your controller with:")
    print(f"  self.kp = {gains['kp']:.4f}")
    print(f"  self.ki = {gains['ki']:.4f}")
    print(f"  self.kd = {gains['kd']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
