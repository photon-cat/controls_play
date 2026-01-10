"""
Simple sequential steer search.

For each timestep:
1. Start with previous optimal as guess
2. Nudge by delta (0.01), check if better
3. If nudge too big (worse both ways), halve delta
4. Repeat until delta < threshold
5. Move to next timestep
"""
import sys
sys.path.insert(0, '/Users/delta/comma/controls_challenge')

import numpy as np
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import BaseController

DATA_PATH = "data/00000.csv"
MODEL_PATH = "models/tinyphysics.onnx"

# Search params
DELTA_START = 0.01
DELTA_MIN = 0.001
LOOKAHEAD = 6  # How many steps ahead to evaluate


class ManualController(BaseController):
    def __init__(self):
        self.action = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.action


def run_and_get_cost(sim, actions_to_apply, n_steps):
    """
    Apply actions and return cost over those steps.
    sim should be at step CONTROL_START_IDX.
    """
    start_la_hist_len = len(sim.current_lataccel_history)

    for i, action in enumerate(actions_to_apply):
        if sim.step_idx >= len(sim.data) - 1:
            break
        sim.controller.action = action
        sim.step()
        if i >= n_steps - 1:
            break

    # Compute cost over these steps
    end = len(sim.current_lataccel_history)
    if end <= CONTROL_START_IDX:
        return float('inf')

    target = np.array(sim.target_lataccel_history[CONTROL_START_IDX:end])
    pred = np.array(sim.current_lataccel_history[CONTROL_START_IDX:end])

    lataccel_err = np.mean((target - pred)**2) * 100
    jerk = np.mean((np.diff(pred) / 0.1)**2) * 100 if len(pred) > 1 else 0

    return lataccel_err * 5 + jerk


def create_fresh_sim():
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ManualController()
    return TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)


def run_warmup(sim):
    """Run to control start."""
    while sim.step_idx < CONTROL_START_IDX:
        sim.step()


def search_one_step(locked_actions, test_steer, lookahead):
    """
    Evaluate cost when we use locked_actions then test_steer.
    """
    sim = create_fresh_sim()
    run_warmup(sim)

    # Apply locked actions
    for a in locked_actions:
        if sim.step_idx >= len(sim.data) - 1:
            break
        sim.controller.action = a
        sim.step()

    # Apply test steer and continue for lookahead
    test_actions = [test_steer] * lookahead

    for a in test_actions:
        if sim.step_idx >= len(sim.data) - 1:
            break
        sim.controller.action = a
        sim.step()

    # Cost
    end = len(sim.current_lataccel_history)
    if end <= CONTROL_START_IDX:
        return float('inf')

    target = np.array(sim.target_lataccel_history[CONTROL_START_IDX:end])
    pred = np.array(sim.current_lataccel_history[CONTROL_START_IDX:end])

    lataccel_err = np.mean((target - pred)**2) * 100
    jerk = np.mean((np.diff(pred) / 0.1)**2) * 100 if len(pred) > 1 else 0

    return lataccel_err * 5 + jerk


def binary_search_steer(locked_actions, initial_guess):
    """
    Binary search for optimal steer at this step.
    """
    current = initial_guess
    delta = DELTA_START

    best = current
    best_cost = search_one_step(locked_actions, current, LOOKAHEAD)

    while delta >= DELTA_MIN:
        # Try up
        up = np.clip(current + delta, -2.0, 2.0)
        cost_up = search_one_step(locked_actions, up, LOOKAHEAD)

        # Try down
        down = np.clip(current - delta, -2.0, 2.0)
        cost_down = search_one_step(locked_actions, down, LOOKAHEAD)

        if cost_up < best_cost and cost_up <= cost_down:
            current = up
            best_cost = cost_up
            best = current
        elif cost_down < best_cost:
            current = down
            best_cost = cost_down
            best = current
        else:
            # Neither better, shrink delta
            delta /= 2

    return best


def full_cost(actions):
    """Full simulation cost."""
    sim = create_fresh_sim()
    run_warmup(sim)

    for a in actions:
        if sim.step_idx >= len(sim.data) - 1:
            break
        sim.controller.action = a
        sim.step()

    # Finish remaining steps with last action
    last = actions[-1] if actions else 0
    while sim.step_idx < len(sim.data) - 1:
        sim.controller.action = last
        sim.step()

    return sim.compute_cost()['total_cost']


def main():
    print("=== Simple Sequential Search ===\n", flush=True)

    # Get FF baseline for initial guesses
    from controllers.feedforward import Controller as FFController
    ff_sim = create_fresh_sim()
    ff_sim.controller = FFController()
    while ff_sim.step_idx < len(ff_sim.data) - 1:
        ff_sim.step()

    ff_cost = ff_sim.compute_cost()['total_cost']
    ff_actions = ff_sim.action_history[CONTROL_START_IDX:500]  # Steps 100-499
    print(f"FF baseline cost: {ff_cost:.2f}", flush=True)

    # Sequential search
    locked = []
    prev_optimal = ff_actions[0]

    print("\nSearching...", flush=True)
    for i in range(400):  # 400 controlled steps
        optimal = binary_search_steer(locked, prev_optimal)
        locked.append(optimal)
        prev_optimal = optimal

        if i % 50 == 0:
            curr_cost = full_cost(locked)
            print(f"Step {CONTROL_START_IDX + i}: steer={optimal:.4f}, cost={curr_cost:.2f}", flush=True)

    final_cost = full_cost(locked)
    print(f"\n=== Results ===")
    print(f"FF baseline: {ff_cost:.2f}")
    print(f"Optimized: {final_cost:.2f}")
    print(f"Improvement: {ff_cost - final_cost:.2f}")

    np.save("scripts/simple_search_result.npy", np.array(locked))
    print("Saved to scripts/simple_search_result.npy")


if __name__ == "__main__":
    main()
