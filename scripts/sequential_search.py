"""
Sequential search for optimal steer commands.

For each timestep:
1. Start with previous timestep's optimal steer as initial guess
2. Nudge up/down by delta, see which direction improves cost
3. Keep nudging in that direction until cost gets worse
4. Halve delta and repeat until resolution threshold
5. Lock in optimal value, move to next timestep
"""
import sys
sys.path.insert(0, '/Users/delta/comma/controls_challenge')

import numpy as np
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import BaseController

DATA_PATH = "data/00000.csv"
MODEL_PATH = "models/tinyphysics.onnx"

# Search parameters
INITIAL_DELTA = 0.05      # Initial nudge size
MIN_DELTA = 0.005         # Stop refining below this
LOOKAHEAD = 5             # Steps to simulate ahead for cost evaluation


class ReplayController(BaseController):
    """Controller that replays actions by step index."""
    def __init__(self, actions):
        self.actions = list(actions)
        self.call_count = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        step = CONTEXT_LENGTH + self.call_count
        self.call_count += 1
        if step < len(self.actions):
            return self.actions[step]
        return 0.0


def get_baseline_actions():
    """Get baseline from feedforward controller."""
    from controllers.feedforward import Controller
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = Controller()
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)

    while sim.step_idx < len(sim.data) - 1:
        sim.step()

    cost = sim.compute_cost()
    return np.array(sim.action_history), cost['total_cost']


def evaluate_actions(actions):
    """Run simulation with given actions and return cost."""
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ReplayController(actions)
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)

    while sim.step_idx < len(sim.data) - 1:
        sim.step()

    return sim.compute_cost()['total_cost']


def evaluate_partial(actions, up_to_step):
    """
    Evaluate cost contribution from steps 100 to up_to_step.
    This is faster than full simulation for early steps.
    """
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ReplayController(actions)
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)

    # Run until up_to_step + LOOKAHEAD to see effect of current action
    end_step = min(up_to_step + LOOKAHEAD, len(sim.data) - 1)
    while sim.step_idx < end_step:
        sim.step()

    # Compute partial cost (lataccel error + jerk so far)
    start = CONTROL_START_IDX
    end = min(up_to_step + LOOKAHEAD, len(sim.current_lataccel_history))

    if end <= start:
        return float('inf')

    target = np.array(sim.target_lataccel_history)[start:end]
    pred = np.array(sim.current_lataccel_history)[start:end]

    lataccel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / 0.1)**2) * 100 if len(pred) > 1 else 0

    return lataccel_cost * 5 + jerk_cost


def search_optimal_steer(actions, step_idx, prev_optimal):
    """
    Binary search for optimal steer at step_idx.
    Returns (optimal_steer, final_cost).
    """
    current = prev_optimal
    delta = INITIAL_DELTA

    # Get baseline cost with current guess
    test_actions = actions.copy()
    test_actions[step_idx] = current
    best_cost = evaluate_partial(test_actions, step_idx)
    best_steer = current

    while delta >= MIN_DELTA:
        improved = False

        # Try nudging up
        test_actions[step_idx] = np.clip(current + delta, -2.0, 2.0)
        cost_up = evaluate_partial(test_actions, step_idx)

        # Try nudging down
        test_actions[step_idx] = np.clip(current - delta, -2.0, 2.0)
        cost_down = evaluate_partial(test_actions, step_idx)

        # Pick best direction
        if cost_up < best_cost and cost_up <= cost_down:
            # Go up
            current = np.clip(current + delta, -2.0, 2.0)
            best_cost = cost_up
            best_steer = current
            improved = True
        elif cost_down < best_cost:
            # Go down
            current = np.clip(current - delta, -2.0, 2.0)
            best_cost = cost_down
            best_steer = current
            improved = True

        if not improved:
            # Halve delta and try finer search
            delta /= 2

    return best_steer, best_cost


def main():
    print("=== Sequential Steer Search for 00000.csv ===\n")

    # Get baseline
    print("Getting baseline from feedforward controller...")
    actions, baseline_cost = get_baseline_actions()
    print(f"Baseline cost: {baseline_cost:.2f}")
    print(f"Actions length: {len(actions)}")

    # Verify replay
    replay_cost = evaluate_actions(actions)
    print(f"Replay cost: {replay_cost:.2f}")

    # Search optimal steer for each controlled step
    print(f"\n--- Searching steps {CONTROL_START_IDX} to 499 ---")

    optimized = actions.copy()
    prev_optimal = actions[CONTROL_START_IDX]  # Start with baseline

    for step in range(CONTROL_START_IDX, 500):
        # Search for optimal steer at this step
        optimal_steer, step_cost = search_optimal_steer(optimized, step, prev_optimal)

        # Lock it in
        optimized[step] = optimal_steer
        prev_optimal = optimal_steer

        # Progress update every 20 steps
        if (step - CONTROL_START_IDX) % 20 == 0:
            full_cost = evaluate_actions(optimized)
            print(f"Step {step}: steer={optimal_steer:.4f}, full_cost={full_cost:.2f}")

    # Final evaluation
    final_cost = evaluate_actions(optimized)
    print(f"\n=== Results ===")
    print(f"Baseline cost: {baseline_cost:.2f}")
    print(f"Optimized cost: {final_cost:.2f}")
    print(f"Improvement: {baseline_cost - final_cost:.2f}")

    # Save
    np.save("scripts/sequential_search_actions.npy", optimized)
    print("\nSaved to scripts/sequential_search_actions.npy")


if __name__ == "__main__":
    main()
