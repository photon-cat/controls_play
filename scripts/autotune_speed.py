#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import importlib
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from tqdm import tqdm
import argparse
import re

# Configuration
MODEL_PATH = "models/tinyphysics.onnx"
CONTROLLER_MODULE = "controllers.pid_ff_scheduled_tune"
CONTROLLER_PATH = "controllers/pid_ff_scheduled_tune.py"

class TunerState:
    def __init__(self, speed_idx):
        self.speed_idx = speed_idx
        self.best_cost = float('inf')
        self.best_params = None

    def update_if_best(self, params, cost):
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params
            print(f"\nNEW BEST FOUND! Cost: {cost:.4f}. Saving to controller...")
            update_controller_file(self.speed_idx, params)
            return True
        return False

def update_controller_file(speed_idx, params):
    """Updates the source code of the controller with new params."""
    with open(CONTROLLER_PATH, 'r') as f:
        content = f.read()
    
    names = ['p_points', 'i_points', 'd_points', 'k_ff_points', 'preview_points']
    
    for i, name in enumerate(names):
        pattern = rf'(self\.{name}\s*=\s*np\.array\(\[)(.*?)(\]\))'
        
        def replace_func(match):
            prefix = match.group(1)
            values_str = match.group(2)
            suffix = match.group(3)
            values = [v.strip() for v in values_str.split(',')]
            if speed_idx < len(values):
                values[speed_idx] = f"{params[i]:.4f}"
            return f"{prefix}{', '.join(values)}{suffix}"

        lines = []
        for line in content.splitlines():
            if line.strip().startswith(f"self.{name}"):
                line = re.sub(pattern, replace_func, line)
            lines.append(line)
        content = "\n".join(lines) + "\n"
                
    with open(CONTROLLER_PATH, 'w') as f:
        f.write(content)

def run_sims(params, csv_files):
    kp, ki, kd, kf, preview = params
    module = importlib.import_module(CONTROLLER_MODULE)
    tinyphysicsmodel = TinyPhysicsModel(MODEL_PATH, debug=False)
    
    costs = []
    for csv_path in csv_files:
        controller = module.Controller()
        # The indices in the controller instance are still used for simulation
        # but we need to find which index corresponds to our speed_idx
        # For simplicity, we assume the controller class structure matches
        controller.p_points[tuner_state.speed_idx] = kp
        controller.i_points[tuner_state.speed_idx] = ki
        controller.d_points[tuner_state.speed_idx] = kd
        controller.k_ff_points[tuner_state.speed_idx] = kf
        controller.preview_points[tuner_state.speed_idx] = preview
        
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(csv_path), controller=controller, debug=False)
        c = sim.rollout()
        costs.append(c['total_cost'])
        
    return np.mean(costs)

def objective(params, csv_files, iteration_info):
    # Penalize negative gains (except KD)
    if params[0] < 0 or params[1] < 0 or params[3] < 0 or params[4] < 0:
        return 1e9
    
    cost = run_sims(params, csv_files)
    iteration_info['count'] += 1
    
    is_best = tuner_state.update_if_best(params, cost)
    star = "*" if is_best else " "
    print(f"Iter {iteration_info['count']:03d}{star} | Cost: {cost:8.2f} | P:{params[0]:.3f} I:{params[1]:.3f} D:{params[2]:.3f} FF:{params[3]:.3f} Prev:{params[4]:.2f}")
    
    return cost

def main():
    global tuner_state
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed_group", type=str, default="speed_36_40")
    parser.add_argument("--speed_idx", type=int, default=None, help="Manually specify the index in the gain arrays to tune (0-10)")
    parser.add_argument("--max_total_iter", type=int, default=100, help="Total number of simulations allowed per group")
    parser.add_argument("--num_files", type=int, default=3, help="Number of CSV files to use")
    parser.add_argument("--method", type=str, default="Hybrid", choices=["Nelder-Mead", "DE", "Hybrid"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed for file selection")
    args = parser.parse_args()

    tune_dir = Path("tune") / args.speed_group
    all_files = sorted(list(tune_dir.glob("*.csv")))
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Select N random files from the pool
    if len(all_files) > args.num_files:
        csv_files = np.random.choice(all_files, args.num_files, replace=False).tolist()
    else:
        csv_files = all_files
        
    print(f"Selected {len(csv_files)} random files for this tuning session:")
    for f in csv_files:
        print(f"  - {Path(f).name}")

    speed_map = {
        "speed_0_4": 1, "speed_4_9": 2, "speed_8_12": 3, "speed_12_16": 4,
        "speed_16_20": 5, "speed_20_24": 6, "speed_24_28": 7, "speed_28_32": 8,
        "speed_32_36": 9, "speed_36_40": 10
    }
    
    if args.speed_idx is not None:
        speed_idx = args.speed_idx
    else:
        speed_idx = speed_map.get(args.speed_group, 10)
        
    tuner_state = TunerState(speed_idx)

    module = importlib.import_module(CONTROLLER_MODULE)
    c = module.Controller()
    initial_guess = [c.p_points[speed_idx], c.i_points[speed_idx], c.d_points[speed_idx], c.k_ff_points[speed_idx], c.preview_points[speed_idx]]
    
    iteration_info = {'count': 0, 'max': args.max_total_iter}
    print(f"Starting {args.method} tuning for {args.speed_group} (Index {speed_idx}) using {len(csv_files)} files...")
    print(f"Limit: {args.max_total_iter} total simulations.")

    bounds = [(0.0, 0.5), (0.0, 0.2), (-0.15, 0.0), (0.0, 0.2), (0.0, 3.0)]

    def stop_callback(xk, convergence=None):
        if iteration_info['count'] >= iteration_info['max']:
            return True
        return False

    if args.method in ["DE", "Hybrid"]:
        # Each DE generation is roughly popsize * len(bounds) evaluations.
        # With popsize=5 and 5 params, each generation is ~25 evals.
        # We aim for ~70% of total budget in Global phase.
        de_gens = max(1, int((args.max_total_iter * 0.7) / 25))
        print(f"\n--- Phase 1: Global Search (DE, {de_gens} generations) ---")
        res = differential_evolution(
            objective, bounds, args=(csv_files, iteration_info),
            maxiter=de_gens, popsize=5, disp=False, polish=False,
            callback=stop_callback
        )
        initial_guess = res.x

    if args.method in ["Nelder-Mead", "Hybrid"] and iteration_info['count'] < iteration_info['max']:
        remaining = iteration_info['max'] - iteration_info['count']
        print(f"\n--- Phase 2: Local Refinement (Nelder-Mead, {remaining} iters max) ---")
        minimize(
            objective, initial_guess, args=(csv_files, iteration_info),
            method='Nelder-Mead', options={'maxiter': remaining},
            callback=stop_callback
        )

    print(f"\nTuning Complete. Best Cost: {tuner_state.best_cost:.4f}")

if __name__ == "__main__":
    main()


