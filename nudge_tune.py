#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
import importlib
import re
import random
import argparse
import time
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from tqdm.contrib.concurrent import process_map
from functools import partial

# Configuration
MODEL_PATH = "models/tinyphysics.onnx"
CONTROLLER_MODULE = "controllers.pid_ff_scheduled_tune"
CONTROLLER_PATH = "controllers/pid_ff_scheduled_tune.py"
DATA_DIR = Path("data")

def run_single_segment(csv_path, controller_gains):
    """Worker function for multiprocessing."""
    try:
        model = TinyPhysicsModel(MODEL_PATH, debug=False)
        module = importlib.import_module(CONTROLLER_MODULE)
        importlib.reload(module)
        controller = module.Controller()
        for name, values in controller_gains.items():
            setattr(controller, name, np.array(values))
        sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
        costs = sim.rollout()
        return costs['total_cost']
    except Exception:
        return 5000.0 # High penalty for failure

class MiniBatchTuner:
    def __init__(self, batch_size=5, roughness_weight=50.0):
        self.load_gains()
        self.all_files = sorted(list(DATA_DIR.glob("*.csv")))
        self.v_points = np.arange(0, 41, 4)
        self.batch_size = batch_size
        self.roughness_weight = roughness_weight
        self.last_file_mtime = os.path.getmtime(CONTROLLER_PATH)

    def load_gains(self):
        with open(CONTROLLER_PATH, 'r') as f:
            content = f.read()
        self.gains = {}
        names = ['p_points', 'i_points', 'd_points', 'k_ff_points', 'preview_points']
        for name in names:
            pattern = rf'self\.{name}\s*=\s*np\.array\(\[(.*?)\]\)'
            match = re.search(pattern, content)
            if match:
                self.gains[name] = np.array([float(v.strip()) for v in match.group(1).split(',')])
        self.last_file_mtime = os.path.getmtime(CONTROLLER_PATH)

    def check_for_external_changes(self):
        mtime = os.path.getmtime(CONTROLLER_PATH)
        if mtime > self.last_file_mtime:
            print("\n!!! External change detected in controller file. Reloading gains...")
            self.load_gains()

    def save_gains(self):
        with open(CONTROLLER_PATH, 'r') as f:
            content = f.read()
        for name, values in self.gains.items():
            pattern = rf'(self\.{name}\s*=\s*np\.array\(\[)(.*?)(\]\))'
            new_values_str = ", ".join([f"{v:.4f}" for v in values])
            def replace_func(m): return f"{m.group(1)}{new_values_str}{m.group(3)}"
            content = re.sub(pattern, replace_func, content)
        with open(CONTROLLER_PATH, 'w') as f:
            f.write(content)
        self.last_file_mtime = os.path.getmtime(CONTROLLER_PATH)

    def calculate_roughness(self, gains_dict):
        roughness = 0
        for name, values in gains_dict.items():
            diffs = np.diff(values)
            roughness += np.sum(diffs**2)
        return roughness

    def step(self, nudge_scale):
        self.check_for_external_changes()
        
        # 1. Pick a mini-batch of random segments
        batch_files = random.sample(self.all_files, self.batch_size)
        
        # 2. Get baseline avg cost + roughness
        base_costs = process_map(partial(run_single_segment, controller_gains=self.gains), 
                                batch_files, max_workers=os.cpu_count(), chunksize=1, leave=False, desc="Baseline")
        base_avg_cost = np.mean(base_costs)
        base_roughness = self.calculate_roughness(self.gains)
        
        # 3. Create a nudge
        target_idx = random.randint(0, len(self.v_points) - 1)
        temp_gains = {name: val.copy() for name, val in self.gains.items()}
        names_to_nudge = random.sample(list(self.gains.keys()), random.randint(1, 2))
        
        for name in names_to_nudge:
            if name == 'preview_points': n = random.uniform(-0.005, 0.005) * nudge_scale
            elif name == 'd_points': n = random.uniform(-0.0005, 0.0005) * nudge_scale
            else: n = random.uniform(-0.001, 0.001) * nudge_scale
            temp_gains[name][target_idx] += n
            if name == 'd_points': temp_gains[name][target_idx] = min(0, temp_gains[name][target_idx])
            else: temp_gains[name][target_idx] = max(0, temp_gains[name][target_idx])

        # 4. Evaluate nudge on SAME mini-batch
        new_costs = process_map(partial(run_single_segment, controller_gains=temp_gains), 
                               batch_files, max_workers=os.cpu_count(), chunksize=1, leave=False, desc="Nudge")
        new_avg_cost = np.mean(new_costs)
        new_roughness = self.calculate_roughness(temp_gains)
        
        # 5. Hybrid Score (Cost + Roughness Penalty)
        score_diff = (new_avg_cost - base_avg_cost) + (new_roughness - base_roughness) * self.roughness_weight
        
        if score_diff < 0:
            imp = base_avg_cost - new_avg_cost
            print(f"ACCEPTED (Batch Size {self.batch_size}): {base_avg_cost:.4f} -> {new_avg_cost:.4f} (Imp: {imp:.4f}) | Nudged {names_to_nudge} @ Idx {target_idx}")
            self.gains = temp_gains
            self.save_gains()
            return True
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--nudge_scale", type=float, default=1.0)
    parser.add_argument("--roughness_weight", type=float, default=50.0)
    args = parser.parse_args()
    
    tuner = MiniBatchTuner(batch_size=args.batch_size, roughness_weight=args.roughness_weight)
    print(f"Starting Mini-Batch Tuner (Batch: {args.batch_size}, Scale: {args.nudge_scale})")
    
    while True:
        try:
            tuner.step(args.nudge_scale)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()


