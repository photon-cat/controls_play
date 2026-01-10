"""
Full trajectory optimization for segment 00000.csv.

Instead of greedy per-step optimization, optimize entire steer sequence jointly.
Uses CMA-ES (evolutionary strategy) which handles the non-differentiable ML model.
"""
import sys
sys.path.insert(0, '/Users/delta/comma/controls_challenge')

import numpy as np
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import BaseController
from pathlib import Path

DATA_PATH = "data/00000.csv"
MODEL_PATH = "models/tinyphysics.onnx"


class ReplayController(BaseController):
    """Controller that replays recorded actions by step index."""
    def __init__(self, actions):
        # actions is the full action_history (indexed by step)
        # actions[i] corresponds to step i
        self.actions = list(actions)
        self.call_count = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Controller is first called at step CONTEXT_LENGTH (20)
        step = CONTEXT_LENGTH + self.call_count
        self.call_count += 1
        if step < len(self.actions):
            return self.actions[step]
        return 0.0

# Constants
N_STEPS = 400  # Steps 100-499
LAG = 5        # Approximate system lag


def create_simulator(full_action_history):
    """Create simulator with full action history (indexed by step)."""
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ReplayController(full_action_history)
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)
    return sim


def evaluate_trajectory(full_action_history):
    """
    Run simulation with given full action history and return cost.
    full_action_history: array indexed by step (step i = action at index i)
    """
    sim = create_simulator(full_action_history)

    # Run full simulation
    while sim.step_idx < len(sim.data) - 1:
        sim.step()

    # Get cost
    cost_dict = sim.compute_cost()
    return cost_dict['total_cost']


def smooth_init_from_ff(gain=2.08):
    """Initialize steer sequence from simple feedforward using data directly."""
    import pandas as pd
    data = pd.read_csv(DATA_PATH)
    steer_seq = []

    # Generate FF trajectory from data
    for i in range(N_STEPS):
        idx = CONTROL_START_IDX + i
        if idx >= len(data):
            break

        target = data['targetLateralAcceleration'].iloc[idx]
        roll = data['roll'].iloc[idx]
        roll_lataccel = np.sin(roll) * 9.81

        # Simple FF
        steer = (target - roll_lataccel) / gain
        steer = np.clip(steer, -2.0, 2.0)
        steer_seq.append(steer)

    return np.array(steer_seq)


def cma_es_optimize(init_steer, n_iterations=50, population_size=20, sigma=0.1):
    """
    CMA-ES style optimization (simplified version).
    Evolves population of steer sequences toward lower cost.
    """
    dim = len(init_steer)
    mean = init_steer.copy()
    best_steer = init_steer.copy()
    best_cost = evaluate_trajectory(init_steer)

    print(f"Initial cost: {best_cost:.2f}")

    for iteration in range(n_iterations):
        # Generate population by perturbing mean
        population = []
        costs = []

        for _ in range(population_size):
            # Add smooth perturbation (low-pass filtered noise)
            noise = np.random.randn(dim) * sigma
            # Smooth the noise to avoid adding jerk
            kernel = np.ones(10) / 10
            noise = np.convolve(noise, kernel, mode='same')

            candidate = np.clip(mean + noise, -2.0, 2.0)
            population.append(candidate)

            cost = evaluate_trajectory(candidate)
            costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_steer = candidate.copy()

        # Update mean toward best candidates
        sorted_idx = np.argsort(costs)
        top_k = population_size // 4
        elite = [population[i] for i in sorted_idx[:top_k]]
        mean = np.mean(elite, axis=0)

        # Adapt sigma
        if iteration > 0 and iteration % 10 == 0:
            sigma *= 0.9  # Decay exploration

        avg_cost = np.mean(costs)
        min_cost = np.min(costs)
        print(f"Iter {iteration+1}: avg={avg_cost:.2f}, min={min_cost:.2f}, best_ever={best_cost:.2f}")

    return best_steer, best_cost


def gradient_descent_fd(init_steer, n_iterations=20, lr=0.01, eps=0.001):
    """
    Finite-difference gradient descent.
    Slower but more precise than CMA-ES.
    """
    steer = init_steer.copy()
    best_steer = steer.copy()
    best_cost = evaluate_trajectory(steer)

    print(f"Initial cost: {best_cost:.2f}")

    for iteration in range(n_iterations):
        base_cost = evaluate_trajectory(steer)

        # Compute gradient via finite differences
        # Only perturb every Nth step to reduce computation
        grad = np.zeros_like(steer)
        step_size = 10  # Perturb every 10th element

        for i in range(0, len(steer), step_size):
            steer_plus = steer.copy()
            steer_plus[i] += eps
            cost_plus = evaluate_trajectory(steer_plus)
            grad[i] = (cost_plus - base_cost) / eps

        # Smooth gradient and interpolate
        for i in range(len(grad)):
            if i % step_size != 0:
                # Interpolate from neighbors
                left = (i // step_size) * step_size
                right = min(left + step_size, len(grad) - 1)
                if right == left:
                    grad[i] = grad[left]
                else:
                    t = (i - left) / (right - left)
                    grad[i] = grad[left] * (1-t) + grad[right] * t

        # Update
        steer = steer - lr * grad
        steer = np.clip(steer, -2.0, 2.0)

        new_cost = evaluate_trajectory(steer)
        if new_cost < best_cost:
            best_cost = new_cost
            best_steer = steer.copy()

        print(f"Iter {iteration+1}: cost={new_cost:.2f}, best={best_cost:.2f}")

        # Adaptive learning rate
        if new_cost > base_cost:
            lr *= 0.5

    return best_steer, best_cost


def optimize_segments(init_steer, segment_size=50, n_passes=3):
    """
    Optimize trajectory in segments, refining each segment iteratively.
    More tractable than full trajectory optimization.
    """
    steer = init_steer.copy()
    n_segments = len(steer) // segment_size

    best_cost = evaluate_trajectory(steer)
    print(f"Initial cost: {best_cost:.2f}")

    for pass_num in range(n_passes):
        print(f"\n=== Pass {pass_num + 1} ===")

        for seg in range(n_segments):
            start = seg * segment_size
            end = min(start + segment_size, len(steer))

            # Try perturbations on this segment
            best_seg_steer = steer.copy()
            best_seg_cost = evaluate_trajectory(steer)

            for trial in range(10):
                candidate = steer.copy()
                # Smooth random perturbation on segment
                noise = np.random.randn(end - start) * 0.05 / (pass_num + 1)
                kernel = np.ones(5) / 5
                noise = np.convolve(noise, kernel, mode='same')
                candidate[start:end] += noise
                candidate = np.clip(candidate, -2.0, 2.0)

                cost = evaluate_trajectory(candidate)
                if cost < best_seg_cost:
                    best_seg_cost = cost
                    best_seg_steer = candidate.copy()

            steer = best_seg_steer

            if best_seg_cost < best_cost:
                best_cost = best_seg_cost

        print(f"Pass {pass_num + 1} complete: cost={best_cost:.2f}")

    return steer, best_cost


def get_baseline_from_controller():
    """Run actual feedforward controller and extract its full action history."""
    from controllers.feedforward import Controller
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = Controller()
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)

    # Run simulation
    while sim.step_idx < len(sim.data) - 1:
        sim.step()

    cost = sim.compute_cost()
    # Return full action history (indexed by step)
    return np.array(sim.action_history), cost['total_cost']


def optimize_controlled_region(init_actions, segment_size=50, n_passes=3):
    """
    Optimize only the controlled region (steps 100-499) of the action history.
    """
    actions = init_actions.copy()

    best_cost = evaluate_trajectory(actions)
    print(f"Initial cost: {best_cost:.2f}")

    # Only optimize the controlled region
    start_idx = CONTROL_START_IDX
    end_idx = 500  # COST_END_IDX
    n_controlled = end_idx - start_idx
    n_segments = n_controlled // segment_size

    for pass_num in range(n_passes):
        print(f"\n=== Pass {pass_num + 1} ===")

        for seg in range(n_segments):
            seg_start = start_idx + seg * segment_size
            seg_end = min(seg_start + segment_size, end_idx)

            # Try perturbations on this segment
            best_seg_actions = actions.copy()
            best_seg_cost = evaluate_trajectory(actions)

            for trial in range(15):
                candidate = actions.copy()
                # Smooth random perturbation on segment
                noise = np.random.randn(seg_end - seg_start) * 0.03 / (pass_num + 1)
                kernel = np.ones(5) / 5
                noise = np.convolve(noise, kernel, mode='same')
                candidate[seg_start:seg_end] += noise
                candidate = np.clip(candidate, -2.0, 2.0)

                cost = evaluate_trajectory(candidate)
                if cost < best_seg_cost:
                    best_seg_cost = cost
                    best_seg_actions = candidate.copy()

            actions = best_seg_actions

            if best_seg_cost < best_cost:
                best_cost = best_seg_cost

        print(f"Pass {pass_num + 1} complete: cost={best_cost:.2f}")

    return actions, best_cost


if __name__ == "__main__":
    print("=== Trajectory Optimization for 00000.csv ===\n")

    # Get baseline from actual controller
    print("Getting baseline from feedforward controller...")
    baseline_actions, baseline_cost = get_baseline_from_controller()
    print(f"Baseline controller cost: {baseline_cost:.2f}")
    print(f"Full action history length: {len(baseline_actions)}")

    # Verify we can replay the baseline
    replay_cost = evaluate_trajectory(baseline_actions)
    print(f"Replay of baseline: {replay_cost:.2f}")

    # Try segment-based optimization on controlled region
    print("\n--- Optimizing controlled region (steps 100-499) ---")
    opt_actions, opt_cost = optimize_controlled_region(baseline_actions, segment_size=40, n_passes=3)

    print(f"\n=== Final Results ===")
    print(f"Baseline cost: {baseline_cost:.2f}")
    print(f"Optimized cost: {opt_cost:.2f}")
    print(f"Improvement: {baseline_cost - opt_cost:.2f}")

    # Save optimized trajectory
    np.save("scripts/optimized_actions_00000.npy", opt_actions)
    print("\nSaved optimized trajectory to scripts/optimized_actions_00000.npy")
