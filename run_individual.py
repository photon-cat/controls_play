import numpy as np
import pandas as pd
from pathlib import Path
from tinyphysics import run_rollout

def main():
    model_path = "./models/tinyphysics.onnx"
    data_dir = Path("./data")
    test_controller = "pid_ff_scheduled_tune"
    baseline_controller = "pid"
    num_segs = 10

    files = sorted(data_dir.iterdir())[:num_segs]
    
    results = []
    print(f"{'Segment':<15} | {'Controller':<25} | {'Lat':<10} | {'Jerk':<10} | {'Total':<10}")
    print("-" * 80)
    
    for f in files:
        # Run test controller
        test_cost, _, _ = run_rollout(f, test_controller, model_path)
        print(f"{f.stem:<15} | {test_controller:<25} | {test_cost['lataccel_cost']:>10.4f} | {test_cost['jerk_cost']:>10.4f} | {test_cost['total_cost']:>10.4f}")
        
        # Run baseline controller
        baseline_cost, _, _ = run_rollout(f, baseline_controller, model_path)
        print(f"{'':<15} | {baseline_controller:<25} | {baseline_cost['lataccel_cost']:>10.4f} | {baseline_cost['jerk_cost']:>10.4f} | {baseline_cost['total_cost']:>10.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
