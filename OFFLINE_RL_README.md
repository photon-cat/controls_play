# Offline RL with TD3+BC - Summary

## Why Offline RL?

**Online PPO failed** because:
- Exploration destroys good BC-initialized policy
- Takes 1000s of episodes to converge
- Hard to balance exploration vs exploitation

**Offline RL advantages**:
- Learn from large PID dataset (1000+ trajectories)
- No environment interaction needed
- Can leverage expert demonstrations
- More stable training

## Approach: TD3+BC

**TD3 (Twin Delayed DDPG)**:
- Twin Q-networks for stability
- Delayed policy updates
- Target policy smoothing

**+BC (Behavior Cloning)**:
- Regularization term: stay close to dataset actions
- Key innovation: `loss = -λ * Q(s,a) + MSE(π(s), a_dataset)`
- Prevents policy from diverging from data distribution

**Key parameter: α (BC weight)**:
- α=0: Pure TD3 (often fails offline)
- α=2.5: Default (works well)
- α=10: Very conservative (close to BC)

## Dataset

**From `pid_ff_scheduled_tune`:**
- 1000 segments
- ~400k transitions
- Avg cost: ~50-100 (good baseline)

**State**: 104 dims
- Past context (10×6): velocity, accel, roll, target, measured, steer
- Current (4): velocity, accel, roll, measured
- Future context (10×4): velocity, accel, roll, target

**Action**: steering command ∈ [-2, 2]

**Reward**: -(lat_error*50 + jerk + action_change*2) / 50

## Training

```bash
# 1. Collect dataset (1000 segments, ~2 min)
python3 collect_offline_dataset.py \\
    --controller pid_ff \\
    --num_segs 1000 \\
    --workers 16 \\
    --output datasets/pid_ff_1k.npz

# 2. Train TD3+BC (100k steps, ~10 min)
python3 train_td3bc.py \\
    --dataset datasets/pid_ff_1k.npz \\
    --steps 100000 \\
    --batch_size 256 \\
    --alpha 2.5 \\
    --hidden 256 \\
    --normalize

# 3. Evaluate (create controller)
# TODO: Create controllers/td3bc.py to load trained model
```

## Expected Results

**Baseline (PID)**: ~50-100 cost

**Best case**: Beat PID by 10-20%
- TD3+BC learns from PID data
- Smooths out PID's roughness
- Better long-term planning via Q-learning

**Realistic**: Match or slightly beat PID
- Offline RL is conservative
- Limited by dataset quality
- But stable and predictable!

## Next Steps

1. ✅ Collect dataset
2. ⏳ Train TD3+BC
3. Create controller wrapper
4. Benchmark vs PID
5. If needed: collect more data or tune α

## Key Advantages

1. **No exploration disasters** (unlike PPO)
2. **Leverages PID expertise** (smart initialization)
3. **Fast training** (100k steps vs 1000s of episodes)
4. **Stable** (learns offline, evaluates online)
5. **Scalable** (can add more data easily)

## Hyperparameter Guide

- `alpha`: BC weight (2.5 is good, tune if needed)
- `batch_size`: 256 (larger = more stable)
- `hidden`: 256 (can try 128 or 512)
- `steps`: 100k (can do 50k for quick test)
- `normalize`: Usually helps (zero-mean, unit-variance states)

## References

- Fujimoto & Gu (2021): "A Minimalist Approach to Offline RL"
- TD3+BC paper: https://arxiv.org/abs/2106.06860

