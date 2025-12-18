# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the comma Controls Challenge v2 - a competition to build optimal lateral control systems for autonomous vehicles. The challenge uses a ML-based physics simulator (TinyPhysics) trained on real openpilot data to evaluate controllers on their ability to track desired lateral acceleration trajectories while minimizing jerk.

## Commands

### Setup
```bash
pip install -r requirements.txt
```
Recommended Python version: 3.11

### Testing Controllers

Test a single route with debug visualization:
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid
```

Batch evaluation on multiple segments:
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid
```

Generate comparison report (creates `report.html`):
```bash
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero
```

Full submission evaluation (5000 segments):
```bash
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <controller_name> --baseline_controller pid
```

## Architecture

### Core Components

**TinyPhysicsModel** (tinyphysics.py:62-96): ONNX-based ML model that simulates vehicle dynamics. It's an autoregressive transformer that predicts lateral acceleration given:
- Current state: velocity (`v_ego`), forward acceleration (`a_ego`), road roll lateral accel (`roll_lataccel`)
- Control input: steering command (`steer_action`)
- Context: 20 timesteps of history (CONTEXT_LENGTH)
- Uses tokenization of lateral accelerations into 1024 bins for prediction

**TinyPhysicsSimulator** (tinyphysics.py:98-211): Main simulation loop that:
1. Maintains vehicle state history and future plan
2. Calls controller's `update()` method to get steering commands
3. Feeds steering commands to TinyPhysicsModel to predict resulting lateral acceleration
4. Control starts at step 100 (CONTROL_START_IDX); before that, ground truth data is used
5. Limits acceleration changes to MAX_ACC_DELTA (0.5 m/s²) per step to prevent instability

**BaseController** (controllers/__init__.py): Abstract base class. All controllers must implement:
```python
def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Returns: steering command in range [-2, 2]
```

Parameters available to controllers:
- `target_lataccel`: Desired lateral acceleration at current timestep
- `current_lataccel`: Actual current lateral acceleration
- `state`: Named tuple with `roll_lataccel`, `v_ego`, `a_ego`
- `future_plan`: Named tuple with 5-second lookahead (50 steps at 10 Hz) containing arrays of `lataccel`, `roll_lataccel`, `v_ego`, `a_ego`

### Data Flow

1. CSV data files contain logged driving data: vEgo, aEgo, roll, targetLateralAcceleration, steerCommand
2. Data is preprocessed: roll converted to lateral accel, steer commands negated (log uses left-positive, simulator uses right-positive)
3. Simulator runs at 10 Hz (FPS=10, DEL_T=0.1s)
4. Each timestep:
   - Controller receives target + current state + 5s future plan
   - Controller outputs steering command (clipped to [-2, 2])
   - TinyPhysicsModel predicts next lateral acceleration
   - Acceleration delta clamped to ±0.5 m/s²

### Evaluation Metrics

Costs are computed from steps 100-500 (CONTROL_START_IDX to COST_END_IDX):

- **lataccel_cost**: Mean squared error between target and actual lateral acceleration × 100
- **jerk_cost**: Mean squared jerk (rate of change of lateral acceleration) × 100
- **total_cost**: (lataccel_cost × 50) + jerk_cost

Competitive threshold: total_cost < 100

### Controller Implementation

To create a new controller:

1. Create `controllers/your_controller_name.py`
2. Inherit from `BaseController`
3. Implement `update()` method
4. Controller is automatically discovered by `get_available_controllers()`

Example controller structure (see controllers/pid.py):
```python
from . import BaseController

class Controller(BaseController):
    def __init__(self):
        # Initialize controller state
        pass

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute steering command
        return steering_command  # float in [-2, 2]
```

Important constants (all in tinyphysics.py):
- FPS = 10 (simulation frequency)
- CONTROL_START_IDX = 100 (when controller takes over)
- COST_END_IDX = 500 (when cost evaluation ends)
- CONTEXT_LENGTH = 20 (history window for ML model)
- LATACCEL_RANGE = [-5, 5] (valid lateral acceleration range)
- STEER_RANGE = [-2, 2] (valid steering command range)
- MAX_ACC_DELTA = 0.5 (maximum acceleration change per step)
- LAT_ACCEL_COST_MULTIPLIER = 50.0 (weight for tracking vs smoothness)

### Dataset

Data automatically downloads from HuggingFace (commaai/commaSteeringControl) if not present. Each CSV contains synthetic driving data with real-world characteristics. The data/ directory should contain 20,000 segments.

## Notes

- The simulator uses a deterministic seed based on the data file path (MD5 hash), ensuring reproducible results
- The ML model has some stochasticity (temperature=0.8 in prediction) to match real-world variability
- Steering commands are inverted from the logged convention (logged: left-positive, simulator: right-positive)
- Debug mode (--debug flag) shows real-time plots of lateral acceleration, steering commands, road roll, and velocity
