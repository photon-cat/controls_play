#!/usr/bin/env python3
"""
Autotuner for controller parameters using training data.

Uses Bayesian optimization to find optimal controller gains.
"""

import argparse
import importlib
import json
import numpy as np
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


class ControllerAutotuner:
  """
  Automatically tune controller parameters using training data.
  """
  
  def __init__(self, model_path: str, data_dir: str, num_segs: int = 50):
    self.model_path = model_path
    self.data_dir = Path(data_dir)
    self.num_segs = num_segs
    
    # Get training files
    self.train_files = sorted(self.data_dir.glob("*.csv"))[:num_segs]
    print(f"Loaded {len(self.train_files)} training segments")
    
  def evaluate_params(self, params: np.ndarray, controller_module: str, param_names: List[str]) -> float:
    """
    Evaluate controller with given parameters on training set.
    
    Args:
      params: Array of parameter values
      controller_module: Controller module name
      param_names: Names of parameters to tune
    
    Returns:
      Average total cost across training segments (capped for stability)
    """
    # Stability check: penalize obviously bad parameters
    # High integral gain + low derivative often causes instability
    param_dict = dict(zip(param_names, params))
    if 'ki' in param_dict and 'kd' in param_dict:
      if param_dict['ki'] > 0.2 and param_dict['kd'] > -0.02:
        return 10000.0  # Heavy penalty for likely unstable combination
    
    # Create controller with these parameters
    costs = []
    
    for data_path in self.train_files:
      try:
        # Load controller
        controller_class = importlib.import_module(f'controllers.{controller_module}').Controller
        controller = controller_class()
        
        # Set parameters
        for name, value in zip(param_names, params):
          setattr(controller, name, value)
        
        # Run simulation
        model = TinyPhysicsModel(self.model_path, debug=False)
        sim = TinyPhysicsSimulator(
          model,
          str(data_path),
          controller=controller,
          debug=False,
          trace_logger=None,
        )
        cost = sim.rollout()
        
        # Cap individual costs to prevent outliers from dominating
        capped_cost = min(cost['total_cost'], 1000.0)
        costs.append(capped_cost)
        
      except Exception as e:
        print(f"Error on {data_path.name}: {e}")
        costs.append(1000.0)  # Penalty for failure (reduced from 1e6)
    
    avg_cost = np.mean(costs)
    
    # Additional penalty for high variance (unstable across segments)
    cost_std = np.std(costs)
    if cost_std > 200:
      avg_cost += cost_std * 0.5  # Penalize inconsistent performance
    
    return avg_cost
  
  def tune_feedforward_pid(self) -> Dict[str, float]:
    """
    Tune feedforward PID controller parameters.
    
    Parameters to tune:
    - kp, ki, kd: PID gains
    - kff, kff_accel: Feedforward gains
    - roll_comp_gain: Roll compensation
    - derivative_alpha: Derivative filter
    - lookahead_steps: Anticipation horizon
    """
    print("\n=== Tuning Feedforward PID Controller ===\n")
    
    param_names = [
      'kp', 'ki', 'kd',
      'kff', 'kff_accel',
      'roll_comp_gain',
      'derivative_alpha',
    ]
    
    # Parameter bounds: (min, max) - tightened for stability
    bounds = [
      (0.10, 0.35),  # kp - tighter range around known good values
      (0.0, 0.15),   # ki - reduced upper bound to prevent windup
      (-0.15, -0.01), # kd - must be negative for damping
      (0.0, 0.3),    # kff - reduced to prevent overreaction
      (0.0, 0.05),   # kff_accel - reduced range
      (0.7, 1.0),    # roll_comp_gain - tighter around 0.85
      (0.2, 0.6),    # derivative_alpha - middle range for filtering
    ]
    
    # Initial guess (current values)
    x0 = [0.195, 0.100, -0.053, 0.15, 0.02, 0.85, 0.3]
    
    print(f"Optimizing {len(param_names)} parameters:")
    for name, bound in zip(param_names, bounds):
      print(f"  {name:20s}: [{bound[0]:.3f}, {bound[1]:.3f}]")
    
    # Objective function
    def objective(params):
      cost = self.evaluate_params(params, 'feedforward_pid', param_names)
      print(f"  Cost: {cost:.2f} | Params: {[f'{p:.4f}' for p in params]}")
      return cost
    
    print(f"\nEvaluating baseline...")
    baseline_cost = objective(x0)
    print(f"Baseline cost: {baseline_cost:.2f}\n")
    
    print("Running differential evolution optimization...")
    print("This will take several minutes...\n")
    
    # Use differential evolution for global optimization
    result = differential_evolution(
      objective,
      bounds,
      maxiter=15,  # Increase for better results
      popsize=10,
      workers=1,
      polish=True,
      seed=42,
      atol=1.0,
      tol=0.01,
      updating='deferred',
      disp=True,
    )
    
    optimal_params = result.x
    optimal_cost = result.fun
    
    print(f"\n=== Optimization Complete ===")
    print(f"Baseline cost: {baseline_cost:.2f}")
    print(f"Optimized cost: {optimal_cost:.2f}")
    print(f"Improvement: {(baseline_cost - optimal_cost) / baseline_cost * 100:.1f}%\n")
    
    print("Optimal parameters:")
    optimal_dict = {}
    for name, value in zip(param_names, optimal_params):
      print(f"  {name:20s} = {value:.6f}")
      optimal_dict[name] = float(value)
    
    return optimal_dict
  
  def save_tuned_controller(self, params: Dict[str, float], output_path: str):
    """Save tuned controller to file."""
    output_file = Path(output_path)
    
    # Generate controller code with tuned parameters
    controller_code = f'''from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Auto-tuned Feedforward PID Controller.
  
  Parameters optimized on training set using differential evolution.
  """
  
  def __init__(self):
    # PID gains (auto-tuned)
    self.kp = {params['kp']:.6f}
    self.ki = {params['ki']:.6f}
    self.kd = {params['kd']:.6f}
    
    # Feedforward gains (auto-tuned)
    self.kff = {params['kff']:.6f}
    self.kff_accel = {params['kff_accel']:.6f}
    
    # Roll compensation (auto-tuned)
    self.roll_comp_gain = {params['roll_comp_gain']:.6f}
    
    # Derivative filtering (auto-tuned)
    self.derivative_alpha = {params['derivative_alpha']:.6f}
    
    # State tracking
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_target = 0.0
    self.filtered_derivative = 0.0
    
    # Look-ahead for anticipation
    self.lookahead_steps = 3
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """PID control with feedforward anticipation from future plan."""
    # Roll compensation
    roll_lataccel = state.roll_lataccel
    compensated_target = target_lataccel - self.roll_comp_gain * roll_lataccel
    compensated_current = current_lataccel - self.roll_comp_gain * roll_lataccel
    
    # Error signal
    error = compensated_target - compensated_current
    
    # Integral with anti-windup
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -100, 100)
    
    # Filtered derivative
    raw_derivative = error - self.prev_error
    self.filtered_derivative = (self.derivative_alpha * raw_derivative + 
                                (1 - self.derivative_alpha) * self.filtered_derivative)
    
    # PID control
    u_pid = self.kp * error + self.ki * self.error_integral + self.kd * self.filtered_derivative
    
    # Feedforward from future plan
    u_feedforward = 0.0
    if future_plan and hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > 0:
      if self.lookahead_steps < len(future_plan.lataccel):
        future_target = future_plan.lataccel[self.lookahead_steps]
        target_rate = (future_target - target_lataccel) / (self.lookahead_steps * 0.1)
        u_feedforward += self.kff * target_rate
      
      if len(future_plan.a_ego) > self.lookahead_steps:
        future_accel = future_plan.a_ego[self.lookahead_steps]
        current_accel = state.a_ego
        accel_change = future_accel - current_accel
        u_feedforward += self.kff_accel * accel_change
    
    # Total control
    control = u_pid + u_feedforward
    
    # Update state
    self.prev_error = error
    self.prev_target = target_lataccel
    
    return control
'''
    
    output_file.write_text(controller_code)
    print(f"\nSaved tuned controller to: {output_file}")
    
    # Also save parameters as JSON
    json_file = output_file.with_suffix('.json')
    with json_file.open('w') as f:
      json.dump(params, f, indent=2)
    print(f"Saved parameters to: {json_file}")


def main():
  parser = argparse.ArgumentParser(description="Auto-tune controller parameters")
  parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
  parser.add_argument("--data_path", type=str, default="./data")
  parser.add_argument("--num_segs", type=int, default=50, help="Number of training segments")
  parser.add_argument("--output", type=str, default="./controllers/tuned_ff_pid.py")
  args = parser.parse_args()
  
  tuner = ControllerAutotuner(args.model_path, args.data_path, args.num_segs)
  optimal_params = tuner.tune_feedforward_pid()
  tuner.save_tuned_controller(optimal_params, args.output)
  
  print("\n=== Next Steps ===")
  print(f"Test tuned controller:")
  print(f"  python tinyphysics.py --model_path {args.model_path} --data_path ./data_val --num_segs 10 --controller tuned_ff_pid")


if __name__ == "__main__":
  main()

