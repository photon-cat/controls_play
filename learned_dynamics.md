# Learned Dynamics Model

## Objective
Learn dynamics model for system that model-based controllers can use (MPC, planning).

## Training Data
- Source: `/logging_data/pid` - 500 segments of PID controlling vehicle
- Each segment: 600 timesteps at 10Hz

## Problem Formulation

```
model = f(state_action_history, action_plan) = latAccelPlan
```

### Inputs
**state_action_history** (t-10 to t-1, 10 steps):
- vEgo (m/s)
- aEgo (m/s²)
- rollAccel (m/s²) - lateral accel from road bank
- currentLataccel (m/s²)
- steerCommand (action taken)
- Features per step: 5
- Total: 50 features

**action_plan** (t+1 to t+10, 10 steps):
- steerCommand only
- Features per step: 1
- Total: 10 features

### Output
**latAccelPlan** (t+1 to t+10, 10 steps):
- Predicted currentLataccel
- Total: 10 values

## Architecture Option 1: Simple MLP

Fast, good baseline for CPU training.

```
Input: [state_action_history (50), action_plan (10)] = 60 features
    ↓
Linear(60, 128) + ReLU
    ↓
Linear(128, 128) + ReLU
    ↓
Linear(128, 64) + ReLU
    ↓
Linear(64, 10)
    ↓
Output: latAccelPlan (10)
```

| Param | Value |
|-------|-------|
| Input dim | 60 |
| Hidden | [128, 128, 64] |
| Output dim | 10 |
| Params | ~25K |
| Activation | ReLU |

### MLP Pros/Cons
- (+) Very fast training on CPU
- (+) Simple to implement and debug
- (+) Good baseline
- (-) No explicit temporal structure
- (-) Must flatten sequence info

## Architecture Option 2: Causal Transformer

Better temporal modeling, still small enough for CPU.

```
Input Sequence (20 tokens):
┌─────────────────────────────────┬────────────────────────────────┐
│  History (10 tokens)            │  Plan (10 tokens)              │
│  [vEgo,aEgo,roll,lat,steer]     │  [steer, 0, 0, 0, 0] padded    │
└─────────────────────────────────┴────────────────────────────────┘
                    ↓
        Token Embedding (5 → d_model)
                    ↓
        Positional Encoding (sinusoidal)
                    ↓
        2x Transformer Decoder Blocks
        (causal mask: each token sees only past)
                    ↓
        Output Head (d_model → 1) on plan tokens only
                    ↓
        latAccelPlan (10 values)
```

### Transformer Config

| Param | Value | Notes |
|-------|-------|-------|
| d_model | 64 | Token embedding dim |
| n_heads | 4 | Multi-head attention |
| n_layers | 2 | Transformer blocks |
| d_ff | 128 | Feedforward hidden |
| context | 20 | 10 history + 10 plan |
| dropout | 0.1 | Regularization |
| params | ~50K | CPU-friendly |

### Attention Pattern
```
Causal mask - plan tokens attend to all history + prior plan:

        h0 h1 h2 ... h9 p0 p1 p2 ... p9
    h0  ✓
    h1  ✓  ✓
    h2  ✓  ✓  ✓
    ...
    h9  ✓  ✓  ✓  ... ✓
    p0  ✓  ✓  ✓  ... ✓  ✓
    p1  ✓  ✓  ✓  ... ✓  ✓  ✓
    ...
    p9  ✓  ✓  ✓  ... ✓  ✓  ✓  ✓  ... ✓
```

### Transformer Pros/Cons
- (+) Explicit temporal modeling
- (+) Attention can learn which history steps matter
- (+) Scales better if we increase context
- (-) Slower than MLP
- (-) More complex implementation

## Training Details

### Loss
MSE on predicted vs actual latAccel:
```
loss = mean((latAccelPlan_pred - latAccelPlan_actual)²)
```

### Data Preprocessing
1. Load all CSVs from `/logging_data/pid/`
2. For each segment, create sliding windows:
   - Start from t=20 (need 10 steps history)
   - End at t=590 (need 10 steps future)
   - ~570 samples per segment
3. Normalize inputs (standardize each feature)
4. Train/val split: 80/20 by segment

### Training Config
| Param | Value |
|-------|-------|
| Batch size | 64 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Epochs | 50-100 |
| Early stopping | patience=10 |

## Inference Usage

For MPC/planning controller:
```python
# Given current state and candidate action sequence
history = get_last_10_steps()  # from sim
action_plan = candidate_actions  # being evaluated

# Predict outcome
predicted_lataccel = model(history, action_plan)

# Use for cost computation
cost = compute_cost(predicted_lataccel, target_lataccel)
```

## Next Steps
1. Implement data loader for `/logging_data/pid/`
2. Implement MLP baseline
3. Train and evaluate MLP
4. Implement Transformer
5. Compare performance
6. Integrate with MPC controller
