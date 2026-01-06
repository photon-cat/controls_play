#!/usr/bin/env python3
"""
Fast parallel benchmark runner for controllers.

Usage:
    python benchmark.py --controller pid --num_segs 64 --workers 16
    python benchmark.py --controller neural --num_segs 100 --workers 16
"""

import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm


def run_single_segment(args):
    """Run a single segment - worker function for parallel execution."""
    data_path, controller_type, model_path, neural_model = args
    
    # Import inside worker to avoid pickle issues
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    # Load controller
    if controller_type == 'pid':
        from controllers.pid import Controller
        controller = Controller()
    elif controller_type == 'pid_ff':
        from controllers.pid_ff_scheduled_tune import Controller
        controller = Controller()
    elif controller_type == 'td3bc':
        from controllers.td3bc import Controller
        controller = Controller()
    elif controller_type == 'neural':
        from controllers.neural import Controller
        controller = Controller(model_path=neural_model)
    elif controller_type == 'ppo_debug':
        from controllers.ppo_debug import Controller
        controller = Controller(model_path=neural_model)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")
    
    # Load physics model
    physics = TinyPhysicsModel(model_path, debug=False)
    
    # Run simulation
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=controller, debug=False)
    sim.rollout()
    
    cost = sim.compute_cost()
    return {
        'path': str(data_path),
        'lataccel_cost': cost['lataccel_cost'],
        'jerk_cost': cost['jerk_cost'],
        'total_cost': cost['total_cost']
    }


def benchmark_parallel(data_dir, controller_type, model_path, num_segs=64, workers=16, neural_model=None):
    """Run benchmark with parallel workers."""
    
    data_path = Path(data_dir)
    files = sorted(data_path.glob('*.csv'))[:num_segs]
    
    print(f"Benchmarking {controller_type} on {len(files)} segments with {workers} workers...")
    if neural_model:
        print(f"Neural model: {neural_model}")
    
    # Prepare args for workers
    args_list = [(f, controller_type, model_path, neural_model) for f in files]
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_segment, args): args[0] for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    lataccel_costs = [r['lataccel_cost'] for r in results]
    jerk_costs = [r['jerk_cost'] for r in results]
    total_costs = [r['total_cost'] for r in results]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {controller_type}")
    print(f"{'='*60}")
    print(f"Segments: {len(results)}")
    if len(results) == 0:
        print("ERROR: No segments completed successfully!")
        return None
    print(f"Time: {elapsed:.1f}s ({elapsed/len(results):.2f}s/seg, {len(results)/elapsed:.1f} seg/s)")
    print(f"")
    print(f"Lataccel Cost: {np.mean(lataccel_costs):.2f} ± {np.std(lataccel_costs):.2f}")
    print(f"Jerk Cost:     {np.mean(jerk_costs):.2f} ± {np.std(jerk_costs):.2f}")
    print(f"Total Cost:    {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")
    print(f"{'='*60}")
    
    return {
        'controller': controller_type,
        'num_segs': len(results),
        'time': elapsed,
        'lataccel_cost_mean': np.mean(lataccel_costs),
        'lataccel_cost_std': np.std(lataccel_costs),
        'jerk_cost_mean': np.mean(jerk_costs),
        'jerk_cost_std': np.std(jerk_costs),
        'total_cost_mean': np.mean(total_costs),
        'total_cost_std': np.std(total_costs),
    }


def main():
    parser = argparse.ArgumentParser(description='Fast parallel benchmark')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--controller', type=str, default='pid', 
                        choices=['pid', 'pid_ff', 'td3bc', 'neural', 'ppo_debug'],
                        help='Controller to benchmark')
    parser.add_argument('--model_path', type=str, default='models/tinyphysics.onnx', 
                        help='Path to physics model')
    parser.add_argument('--neural_model', type=str, default=None,
                        help='Path to neural steer model (for neural controller)')
    parser.add_argument('--num_segs', type=int, default=64, help='Number of segments to run')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()
    
    benchmark_parallel(
        args.data_path, 
        args.controller, 
        args.model_path,
        num_segs=args.num_segs,
        workers=args.workers,
        neural_model=args.neural_model
    )


if __name__ == '__main__':
    main()

