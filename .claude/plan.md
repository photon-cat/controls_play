# Physics-Based Feedforward Controller Plan

## Problem Analysis

From data analysis of the TinyPhysicsModel, I've identified:

1. **Steady-state gain**: lataccel ≈ 1.8-2.0 × steer_command (velocity-dependent)
2. **Rate-limited dynamics**: dLatAccel/dt ≈ 5.0 m/s³ maximum (from overshoot analysis)
3. **Velocity dependence**: Bicycle model suggests gain ∝ v²/wheelbase for small angles
4. **Overshoot problem**: At high target lataccel (>3.5 m/s²), steering saturates at 2.0, but rate limit causes overshoot to 5.0 m/s²

## Root Cause of Spikes

Steps 173-180 show:
- Steering saturates at 2.0
- Lataccel ramps at max rate: 5.0 m/s³
- Controller tries to back off (steer: 2.0 → 1.4 → 1.2 → 0.9)
- But lataccel has inertia, overshoots to 5.0 m/s²

This is a **rate-limited first-order system** with state-dependent gain.

## Proposed Physics-Based Feedforward Approach

### Model Structure

Simplified bicycle model approximation:
```
lataccel(t+dt) = lataccel(t) + rate_limited(K(v) * steer - lataccel(t), max_rate=5.0) * dt
```

Where:
- `K(v)` = velocity-dependent gain ≈ v² / wheelbase_effective
- Rate limiting: max |dLatAccel/dt| = 5.0 m/s³
- First-order lag with rate constraint

### Implementation Plan

Create `controllers/physics_feedforward.py` with:

1. **System Identification Component**
   - Online gain estimation: K(v) from observed steer → lataccel
   - Use exponential moving average to track gain at current velocity
   - Initialize with K₀ ≈ 1.8 (from data analysis)

2. **Physics-Based Feedforward**
   ```python
   # Predict required steering for target lataccel
   K_est = estimate_gain(v_ego, recent_observations)
   steer_feedforward = target_lataccel / K_est

   # Predictive rate limiting: if we're approaching target fast
   lataccel_rate = (current_lataccel - prev_lataccel) / dt
   predicted_overshoot = current_lataccel + lataccel_rate * lookahead_time - target_lataccel

   if predicted_overshoot > threshold:
       # Back off preemptively based on physics
       brake_command = -predicted_overshoot / K_est
       steer_feedforward += brake_command
   ```

3. **PID Feedback Loop**
   - Keep auto-tuned PID gains (kp=0.0893, ki=0.1540, kd=0.0130)
   - PID corrects for model errors
   - Physics feedforward handles nominal trajectory

4. **Adaptive Rate Limiting**
   - Keep existing adaptive rate limiting
   - But allow faster response when physics model says we won't overshoot

## Key Innovations

1. **Velocity-adaptive gain**: K(v) estimation prevents over/under steering at different speeds
2. **Predictive braking**: Use physics to predict overshoot before it happens
3. **Rate-aware control**: Account for 5.0 m/s³ rate limit in predictions
4. **Hybrid approach**: Physics feedforward + PID feedback = robustness

## Expected Improvements

- Reduce overshoot spikes (currently hits 5.0 m/s² saturation)
- Better tracking at high lataccel targets (3.5+ m/s²)
- Smoother control (less reactive braking)
- Lower jerk cost (anticipate instead of react)

## Implementation Steps

1. Create `PhysicsGainEstimator` class
   - Track (steer, lataccel, v_ego) history
   - Estimate K(v) using weighted least squares
   - Provide K_est for current velocity

2. Create `PhysicsPredictor` class
   - Predict lataccel evolution given current state + action
   - Account for rate limiting (5.0 m/s³)
   - Compute required steer for target trajectory

3. Create `PhysicsFFController` main controller
   - Combine physics feedforward + PID feedback
   - Predictive overshoot prevention
   - Adaptive rate limiting

4. Test and tune on challenging scenarios
   - Start with tuning_scenario_01 (peak 3.78 m/s²)
   - Compare to best autotuned result (cost 2,160.6)

## Alternative Approaches Considered

1. ❌ **Model-free gain scheduling**: Already tried, doesn't prevent overshoot
2. ❌ **Reactive braking**: Tried, increases jerk
3. ✅ **Physics-based prediction**: Anticipate dynamics, prevent overshoot proactively

## Files to Create

- `controllers/physics_feedforward.py` - Main controller
- Test with: `tinyphysics_logging.py --controller physics_feedforward`

## Success Criteria

- Total cost < 2,160 on tuning_scenario_01
- Max |lataccel| < 4.8 m/s² (avoid saturation)
- Smoother response (lower jerk than current best)
