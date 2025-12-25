#!/usr/bin/env python3
"""
Extended version of tinyphysics.py with automatic logging and visualization support.

Usage:
  python tinyphysics_logging.py --model_path ./models/tinyphysics.onnx \
                                --data_path ./data/00003.csv \
                                --controller pid_enhanced \
                                --log_dir logs \
                                --visualize
"""

import argparse
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, get_available_controllers
from visualize_run import load_run_log, plot_run

def run_with_logging(data_path, controller_type, model_path, log_dir=None, visualize=False):
    """Run a rollout and save log data."""
    # Setup
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
    controller = importlib.import_module(f'controllers.{controller_type}').Controller()
    sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=False)

    # Run
    costs = sim.rollout()

    log_file = None
    plot_file = None

    # Save log if requested
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)

        # Create log filename
        data_name = Path(data_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{controller_type}_{data_name}_{timestamp}.npz"

        # Save data
        np.savez(
            log_file,
            target_lataccel_history=np.array(sim.target_lataccel_history),
            current_lataccel_history=np.array(sim.current_lataccel_history),
            action_history=np.array(sim.action_history),
            state_history=np.array(sim.state_history),
            costs=costs,
            metadata={
                'controller': controller_type,
                'data_path': str(data_path),
                'timestamp': timestamp,
                'model_path': model_path
            }
        )

        print(f"Log saved to: {log_file}")

        # Generate visualization if requested
        if visualize:
            try:
                # Load the data we just saved
                run_data = load_run_log(log_file)

                # Create plot filename
                plot_file = log_dir / f"{controller_type}_{data_name}_{timestamp}.png"

                # Generate and save plot
                plot_run(run_data, save_path=str(plot_file))
                print(f"Plot saved to: {plot_file}")

            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")

    return costs, sim.target_lataccel_history, sim.current_lataccel_history, log_file, plot_file

if __name__ == "__main__":
    available_controllers = get_available_controllers()

    parser = argparse.ArgumentParser(description='Run TinyPhysics simulation with logging')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--controller", default='pid', choices=available_controllers)
    parser.add_argument("--log_dir", type=str, default='logs', help='Directory to save logs (default: logs)')
    parser.add_argument("--no_log", action='store_true', help='Disable logging')
    parser.add_argument("--visualize", action='store_true', help='Automatically generate and save visualization plot')

    args = parser.parse_args()

    # Run with logging
    log_dir = None if args.no_log else args.log_dir
    costs, _, _, log_file, plot_file = run_with_logging(
        args.data_path,
        args.controller,
        args.model_path,
        log_dir=log_dir,
        visualize=args.visualize
    )

    # Print results
    print(f"\nResults:")
    print(f"  Lateral Accel Cost: {costs['lataccel_cost']:.3f}")
    print(f"  Jerk Cost:          {costs['jerk_cost']:.3f}")
    print(f"  Total Cost:         {costs['total_cost']:.3f}")

    if log_file:
        print(f"\nLog file: {log_file}")
        if plot_file:
            print(f"Plot file: {plot_file}")
        else:
            print(f"\nTo visualize this run:")
            print(f"  python visualize_run.py --log_file {log_file}")
