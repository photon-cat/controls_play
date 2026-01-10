# Codex Agent Experiments for Controls Challenge

## Background

This is the [comma.ai controls challenge](https://github.com/commaai/controls_challenge). The goal is to control a simulated vehicle's lateral acceleration to match a target trajectory.

### Cost Function
```
total_cost = lataccel_cost * 5 + jerk_cost
lataccel_cost = mean((target - actual)^2) * 100
jerk_cost = mean((d(actual)/dt)^2) * 100
```

### Theoretical Limits
- **Theoretical minimum cost: ~8.5** (this is the jerk inherent in the target trajectory itself - even perfect tracking has this jerk)
- **Baseline PID controller: ~115**
- **Current best feedforward: ~110**
- **Target: <80 total cost**

### Key Findings So Far
1. `steer = (target_lataccel - roll_lataccel) / gain` - roll compensation is critical
2. Gain varies with velocity (1.0-2.5 range), we have a LUT in `controllers/feedforward.py`
3. ML model has ~4-5 step lag (~0.4-0.5s), lag scales with steer magnitude: `lag = 3 + 2.5 * |steer|`
4. The ML model adds jerk regardless of smooth steer input (model dynamics issue)
5. Large steer commands are "lossy" like hard braking on an EV

### Data
- Use `data_mini/` for quick tests (subset of full dataset)
- Each segment is 600 timesteps at 10Hz
- Control starts at step 100, cost evaluated on steps 100-500

---

## Experiment 1: Optimal PID Tuning

**Prompt:**
```
In the controls_challenge repo, tune the PID controller in controllers/pid.py to minimize total_cost on data_mini/00000.csv.

Grid search over:
- kp: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
- ki: [0.0, 0.05, 0.1, 0.15]
- kd: [-0.1, -0.05, 0, 0.05]

Run: python tinyphysics.py --model_path models/tinyphysics.onnx --data_path data_mini/00000.csv --controller pid

Report the best (kp, ki, kd) and its total_cost. Target: <100 cost.
```

**Deliverable:** Best PID gains and cost value

---

## Experiment 2: Velocity-Dependent Gain Calibration

**Prompt:**
```
The feedforward controller uses steer = (target - roll) / gain where gain varies with velocity.

Using the labeled data (first 100 steps of each segment where steer_command is ground truth), fit the optimal gain at each velocity bin (0-5, 5-10, ..., 35-40 m/s).

For each velocity bin:
1. Collect samples where |steer_command| > 0.1
2. Compute: gain = (target_lataccel - roll_lataccel) / steer_command
3. Report median gain and sample count

Update controllers/feedforward.py GAIN_LUT with your findings. Test on data_mini/00000.csv.
Target: <100 cost.
```

**Deliverable:** Updated GAIN_LUT values and resulting cost

---

## Experiment 3: Adaptive Lookahead Controller

**Prompt:**
```
The ML model has lag that scales with steer magnitude: lag_steps = 3 + 2.5 * |steer|

Create controllers/adaptive_lookahead.py that:
1. Estimates expected steer magnitude from current target
2. Computes adaptive lookahead: lookahead = int(3 + 2.5 * |estimated_steer|)
3. Uses future_plan.lataccel[lookahead] instead of current target
4. Applies feedforward: steer = (future_target - future_roll) / gain

Test on data_mini/00000.csv. Report cost breakdown (lataccel_cost, jerk_cost, total_cost).
Target: <90 cost.
```

**Deliverable:** New controller file and cost results

---

## Experiment 4: Model Predictive Control (MPC)

**Prompt:**
```
Implement a simple MPC controller in controllers/mpc_simple.py:

1. Horizon: 10 steps
2. At each step, try 5 candidate steers: [current-0.1, current-0.05, current, current+0.05, current+0.1]
3. For each candidate, simulate forward using the ML model (run tinyphysics internally)
4. Pick the steer that minimizes predicted cost over horizon
5. Apply only the first action, repeat

Note: You can access the ML model via TinyPhysicsModel class.
Test on data_mini/00000.csv. Target: <85 cost.
```

**Deliverable:** MPC controller and cost results

---

## Experiment 5: Learning from Labeled Data

**Prompt:**
```
The first 100 timesteps of each segment have ground-truth steer_command labels.

Create controllers/imitation.py that:
1. During steps 0-99: observe the labeled steer_command and corresponding (target, roll, v_ego)
2. Fit a simple model: steer = a * (target - roll) + b * v_ego + c
3. During steps 100+: use fitted model for control

Test on 10 segments from data_mini/. Report average cost.
Target: <95 average cost.
```

**Deliverable:** Imitation controller and average cost across segments

---

## Experiment 6: Jerk Minimization via Smoothing

**Prompt:**
```
The jerk_cost dominates our total cost. Implement a controller that explicitly trades lataccel accuracy for smoothness.

In controllers/smooth_ff.py:
1. Compute raw feedforward: u_raw = (target - roll) / gain
2. Apply exponential smoothing: u_smooth = alpha * u_raw + (1-alpha) * u_prev
3. Search for optimal alpha in [0.1, 0.2, 0.3, 0.4, 0.5]

Report cost breakdown for each alpha. Find the alpha that minimizes total_cost.
Test on data_mini/00000.csv. Target: <95 cost.
```

**Deliverable:** Optimal alpha value and cost breakdown

---

## Experiment 7: Two-Stage Controller

**Prompt:**
```
Create controllers/two_stage.py with different strategies for different phases:

Stage 1 (steps 100-200): Aggressive tracking
- Use lookahead=3, full feedforward

Stage 2 (steps 200-500): Smooth tracking
- Use lookahead=6, add smoothing alpha=0.3

The idea: settle quickly then maintain smoothly.

Test on data_mini/00000.csv and data_mini/00001.csv.
Target: <90 cost on both.
```

**Deliverable:** Two-stage controller and costs on both segments

---

## Experiment 8: Error Feedback Integration

**Prompt:**
```
Pure feedforward ignores actual vs target error. Create controllers/ff_pid.py that combines:

1. Feedforward: u_ff = (target - roll) / gain
2. PID on error: u_pid = kp*e + ki*integral(e) + kd*de/dt where e = target - actual
3. Combine: u = u_ff + weight * u_pid

Search over:
- weight: [0.1, 0.2, 0.3]
- kp: [0.1, 0.15, 0.2]
- ki: [0.05, 0.1]
- kd: [-0.05, 0]

Find best combination. Test on data_mini/00000.csv. Target: <85 cost.
```

**Deliverable:** Best parameters and cost

---

## Experiment 9: Segment-Specific Gain Learning

**Prompt:**
```
Different driving segments may have different optimal gains.

Create controllers/learn_gain.py that:
1. During steps 100-200: try gain=1.5, measure error
2. During steps 200-300: try gain=2.0, measure error
3. During steps 300-400: try gain=2.5, measure error
4. During steps 400-500: use the gain that had lowest error

Alternatively: use online gradient descent to adapt gain.

Test on 5 segments. Report per-segment costs.
Target: <95 average cost.
```

**Deliverable:** Adaptive gain controller and per-segment results

---

## Experiment 10: Ensemble Controller

**Prompt:**
```
Create controllers/ensemble.py that runs multiple strategies and picks the best:

1. At each step, compute:
   - u1 = feedforward with lookahead=3
   - u2 = feedforward with lookahead=5
   - u3 = feedforward with lookahead=7
   - u4 = smoothed feedforward (alpha=0.3)

2. Use weighted average: u = w1*u1 + w2*u2 + w3*u3 + w4*u4

Search for optimal weights. Test on data_mini/00000.csv.
Target: <50 cost.
```

**Deliverable:** Optimal weights and cost

---

## Running Tests

```bash
# Single segment
python tinyphysics.py --model_path models/tinyphysics.onnx --data_path data_mini/00000.csv --controller YOUR_CONTROLLER

# Multiple segments
python tinyphysics.py --model_path models/tinyphysics.onnx --data_path data_mini --num_segs 10 --controller YOUR_CONTROLLER

# With debug visualization
python tinyphysics.py --model_path models/tinyphysics.onnx --data_path data_mini/00000.csv --controller YOUR_CONTROLLER --debug
```

## Evaluation Criteria

1. **Cost < 80**: Excellent
2. **Cost 80-100**: Good
3. **Cost 100-115**: Baseline-level
4. **Cost > 115**: Needs improvement

Report both `lataccel_cost` and `jerk_cost` separately - they reveal different issues:
- High lataccel_cost = poor tracking (wrong gain, lag compensation)
- High jerk_cost = too aggressive/jerky (needs smoothing or better feedforward)
