#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from functools import partial
import importlib
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

def run_single_csv(csv_path, controller_type, model_path):
    """Runs a single simulation and returns the costs and the speed group."""
    try:
        # Load model and controller for each process
        tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
        module = importlib.import_module(f'controllers.{controller_type}')
        controller = module.Controller()
        
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(csv_path), controller=controller, debug=False)
        costs = sim.rollout()
        
        # Get speed group from parent directory name
        speed_group = Path(csv_path).parent.name
        
        return {
            'speed_group': speed_group,
            'lataccel_cost': costs['lataccel_cost'],
            'jerk_cost': costs['jerk_cost'],
            'total_cost': costs['total_cost']
        }
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    tune_dir = Path("tune")
    model_path = "models/tinyphysics.onnx"
    controller_type = "pid_ff_scheduled"
    
    if not tune_dir.exists():
        print(f"Directory {tune_dir} not found.")
        return

    # Find all CSV files in tune/ subdirectories
    csv_files = sorted(list(tune_dir.glob("**/*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {tune_dir}.")
        return

    print(f"Found {len(csv_files)} CSV files. Running simulations...")

    # Run simulations in parallel
    run_func = partial(run_single_csv, controller_type=controller_type, model_path=model_path)
    results = process_map(run_func, csv_files, max_workers=os.cpu_count(), chunksize=1)
    
    # Filter out failed runs
    results = [r for r in results if r is not None]
    
    # Create DataFrame and aggregate
    df = pd.DataFrame(results)
    
    # Sort speed groups naturally (by numeric value)
    def extract_speed(name):
        try:
            # name format is usually speed_X_Y
            parts = name.split('_')
            return int(parts[1])
        except:
            return 999

    report = df.groupby('speed_group').agg({
        'lataccel_cost': 'mean',
        'jerk_cost': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    report['sort_key'] = report['speed_group'].apply(extract_speed)
    report = report.sort_values('sort_key').drop(columns=['sort_key'])

    # Print report
    print("\n" + "="*80)
    print(f"Performance Report for {controller_type}")
    print("="*80)
    print(f"{'Speed Group':<15} | {'Lat Accel Cost':<15} | {'Jerk Cost':<15} | {'Total Cost':<15}")
    print("-" * 80)
    
    for _, row in report.iterrows():
        print(f"{row['speed_group']:<15} | {row['lataccel_cost']:<15.4f} | {row['jerk_cost']:<15.4f} | {row['total_cost']:<15.4f}")
    
    print("-" * 80)
    avg_total = report['total_cost'].mean()
    print(f"{'AVERAGE':<15} | {'':<15} | {'':<15} | {avg_total:<15.4f}")
    print("="*80)

    # Save to CSV
    report.to_csv("speed_performance_report.csv", index=False)
    print(f"\nReport saved to speed_performance_report.csv")

if __name__ == "__main__":
    main()
