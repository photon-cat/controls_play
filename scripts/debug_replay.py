"""Debug why replaying controller actions gives different cost."""
import sys
sys.path.insert(0, '/Users/delta/comma/controls_challenge')

import numpy as np
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, CONTEXT_LENGTH
from controllers.feedforward import Controller as FFController
from controllers import BaseController

DATA_PATH = "data/00000.csv"
MODEL_PATH = "models/tinyphysics.onnx"


class ReplayController(BaseController):
    """Controller that replays recorded actions by step index."""
    def __init__(self, actions):
        # actions is the full action_history from original sim
        # actions[i] corresponds to step i
        self.actions = list(actions)
        self.call_count = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Controller is first called at step CONTEXT_LENGTH (20)
        # So call 0 -> step 20, call 1 -> step 21, etc.
        step = CONTEXT_LENGTH + self.call_count
        self.call_count += 1
        if step < len(self.actions):
            return self.actions[step]
        return 0.0


# Run original controller
print("Running feedforward controller...")
model1 = TinyPhysicsModel(MODEL_PATH, debug=False)
controller1 = FFController()
sim1 = TinyPhysicsSimulator(model1, DATA_PATH, controller=controller1, debug=False)

while sim1.step_idx < len(sim1.data) - 1:
    sim1.step()

cost1 = sim1.compute_cost()
print(f"FF controller cost: {cost1}")

# Extract actions
actions = sim1.action_history.copy()
print(f"Total actions: {len(actions)}")
print(f"Actions from 100-105: {actions[100:105]}")

# Replay those actions
print("\nReplaying actions...")
model2 = TinyPhysicsModel(MODEL_PATH, debug=False)
controller2 = ReplayController(actions)
sim2 = TinyPhysicsSimulator(model2, DATA_PATH, controller=controller2, debug=False)

while sim2.step_idx < len(sim2.data) - 1:
    sim2.step()

cost2 = sim2.compute_cost()
print(f"Replay cost: {cost2}")

# Compare action histories
print(f"\nActions recorded in replay: {len(sim2.action_history)}")
print(f"Replay actions 100-105: {sim2.action_history[100:105]}")
print(f"Replay controller idx at end: {controller2.idx}")
print(f"Actions given to replay controller [100:105]: {actions[100:105]}")

# Compare lataccel histories
la1 = np.array(sim1.current_lataccel_history)
la2 = np.array(sim2.current_lataccel_history)
print(f"\nLataccel diff (first 10): {la1[:10] - la2[:10]}")
print(f"Lataccel diff (100-110): {la1[100:110] - la2[100:110]}")
print(f"Max lataccel diff: {np.max(np.abs(la1 - la2))}")
