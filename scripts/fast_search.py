"""
Fast sequential search using cached state.
"""
import sys
sys.path.insert(0, '/Users/delta/comma/controls_challenge')

import numpy as np
import copy
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers import BaseController

DATA_PATH = "data/00000.csv"
MODEL_PATH = "models/tinyphysics.onnx"

# Search parameters
INITIAL_DELTA = 0.02
MIN_DELTA = 0.002
LOOKAHEAD = 8


class ManualController(BaseController):
    """Controller where we set the action manually each step."""
    def __init__(self):
        self.next_action = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.next_action


def create_sim():
    """Create fresh simulator."""
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = ManualController()
    sim = TinyPhysicsSimulator(model, DATA_PATH, controller=controller, debug=False)
    return sim


def copy_sim_state(sim):
    """Copy simulator state for rollback."""
    return {
        'step_idx': sim.step_idx,
        'current_lataccel': sim.current_lataccel,
        'state_history': list(sim.state_history),
        'action_history': list(sim.action_history),
        'current_lataccel_history': list(sim.current_lataccel_history),
        'target_lataccel_history': list(sim.target_lataccel_history),
    }


def restore_sim_state(sim, state):
    """Restore simulator state."""
    sim.step_idx = state['step_idx']
    sim.current_lataccel = state['current_lataccel']
    sim.state_history = list(state['state_history'])
    sim.action_history = list(state['action_history'])
    sim.current_lataccel_history = list(state['current_lataccel_history'])
    sim.target_lataccel_history = list(state['target_lataccel_history'])


def run_warmup(sim):
    """Run simulator through warmup phase."""
    while sim.step_idx < CONTROL_START_IDX:
        sim.step()
    return copy_sim_state(sim)


def evaluate_lookahead(sim, warmup_state, actions_so_far, test_action, steps_ahead):
    """
    Evaluate cost of test_action by simulating steps_ahead into the future.
    Returns partial cost (lataccel + jerk).
    """
    restore_sim_state(sim, warmup_state)

    # Replay actions so far
    for action in actions_so_far:
        sim.controller.next_action = action
        sim.step()

    # Apply test action
    sim.controller.next_action = test_action
    sim.step()

    # Simulate ahead
    for _ in range(steps_ahead - 1):
        if sim.step_idx >= len(sim.data) - 1:
            break
        # Use test_action for remaining steps (simple continuation)
        sim.controller.next_action = test_action
        sim.step()

    # Compute cost over the steps we care about
    start = CONTROL_START_IDX
    end = len(sim.current_lataccel_history)

    if end <= start:
        return float('inf')

    target = np.array(sim.target_lataccel_history[start:end])
    pred = np.array(sim.current_lataccel_history[start:end])

    lataccel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / 0.1)**2) * 100 if len(pred) > 1 else 0

    return lataccel_cost * 5 + jerk_cost


def search_step(sim, warmup_state, actions_so_far, prev_optimal):
    """
    Search for optimal action at current step.
    """
    current = prev_optimal
    delta = INITIAL_DELTA

    # Baseline
    best_cost = evaluate_lookahead(sim, warmup_state, actions_so_far, current, LOOKAHEAD)
    best_action = current

    iterations = 0
    while delta >= MIN_DELTA and iterations < 20:
        iterations += 1

        # Try up
        test_up = np.clip(current + delta, -2.0, 2.0)
        cost_up = evaluate_lookahead(sim, warmup_state, actions_so_far, test_up, LOOKAHEAD)

        # Try down
        test_down = np.clip(current - delta, -2.0, 2.0)
        cost_down = evaluate_lookahead(sim, warmup_state, actions_so_far, test_down, LOOKAHEAD)

        if cost_up < best_cost and cost_up <= cost_down:
            current = test_up
            best_cost = cost_up
            best_action = current
        elif cost_down < best_cost:
            current = test_down
            best_cost = cost_down
            best_action = current
        else:
            delta /= 2

    return best_action, best_cost


def full_eval(sim, actions):
    """Full evaluation with given actions."""
    sim2 = create_sim()
    for i, action in enumerate(actions):
        if sim2.step_idx >= len(sim2.data) - 1:
            break
        sim2.controller.next_action = action if sim2.step_idx >= CONTROL_START_IDX else 0
        sim2.step()

    # Run remaining steps
    while sim2.step_idx < len(sim2.data) - 1:
        sim2.controller.next_action = actions[-1] if actions else 0
        sim2.step()

    return sim2.compute_cost()['total_cost']


def main():
    print("=== Fast Sequential Search ===\n", flush=True)

    # Create simulator
    sim = create_sim()

    # Run warmup and cache state
    print("Running warmup...", flush=True)
    warmup_state = run_warmup(sim)
    print(f"Warmup complete at step {warmup_state['step_idx']}", flush=True)

    # Get baseline actions from FF controller
    print("Getting FF baseline...", flush=True)
    from controllers.feedforward import Controller as FFController
    ff_sim = create_sim()
    ff_sim.controller = FFController()
    while ff_sim.step_idx < len(ff_sim.data) - 1:
        ff_sim.step()
    baseline_cost = ff_sim.compute_cost()['total_cost']
    ff_actions = ff_sim.action_history[CONTROL_START_IDX:]
    print(f"FF baseline cost: {baseline_cost:.2f}", flush=True)

    # Sequential search
    print("\n--- Searching ---", flush=True)
    optimized_actions = []
    prev_optimal = ff_actions[0] if ff_actions else 0.0

    for step in range(400):  # 400 controlled steps (100-499)
        optimal, cost = search_step(sim, warmup_state, optimized_actions, prev_optimal)
        optimized_actions.append(optimal)
        prev_optimal = optimal

        if step % 50 == 0:
            print(f"Step {CONTROL_START_IDX + step}: action={optimal:.4f}", flush=True)

    # Full evaluation
    print("\nEvaluating full trajectory...", flush=True)

    # Build full action list
    full_actions = list(warmup_state['action_history']) + optimized_actions
    final_cost = full_eval(sim, full_actions)

    print(f"\n=== Results ===")
    print(f"FF baseline: {baseline_cost:.2f}")
    print(f"Optimized: {final_cost:.2f}")
    print(f"Improvement: {baseline_cost - final_cost:.2f}")

    np.save("scripts/fast_search_actions.npy", np.array(optimized_actions))


if __name__ == "__main__":
    main()
