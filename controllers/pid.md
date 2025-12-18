# PID Controller Equations

## Overview
This document describes the mathematical equations implemented in the basic PID controller (`controllers/pid.py`).

## Continuous-Time PID Equation

The standard continuous-time PID controller has the form:

```
u(t) = Kₚe(t) + Kᵢ∫e(t)dt + K_d(de(t)/dt)
```

Where:
- `u(t)` = control signal output
- `e(t)` = error = target - measurement
- `Kₚ` = proportional gain
- `Kᵢ` = integral gain
- `K_d` = derivative gain

## Discrete-Time Implementation

In discrete time (sampled at regular intervals), the PID equation becomes:

```
u[k] = Kₚe[k] + Kᵢ∑ᵢ₌₀ᵏe[i] + K_d(e[k] - e[k-1])
```

Where:
- `k` = current time step
- `u[k]` = control output at step k
- `e[k]` = error at step k
- `∑ᵢ₌₀ᵏe[i]` = accumulated error (integral)

## Code Implementation

### Error Calculation
```python
error = (target_lataccel - current_lataccel)
```

### Integral Term (Accumulation)
```python
self.error_integral += error
```
This implements: `∫e(t)dt ≈ ∑e[k]`

**Note**: This implementation has no anti-windup protection. The integral term grows indefinitely and never decreases, which can cause instability.

### Derivative Term (Rate of Change)
```python
error_diff = error - self.prev_error
```
This implements: `de(t)/dt ≈ (e[k] - e[k-1]) / Δt`

### Complete Control Law
```python
return self.p * error + self.i * self.error_integral + self.d * error_diff
```

Which is: `u[k] = p·e[k] + i·∫e[k] + d·(e[k] - e[k-1])`

## Controller Gains (Tuned Values)

```
Proportional (Kp): 0.195
Integral (Ki): 0.100
Derivative (Kd): -0.053
```

## Issues and Limitations

### Integral Windup
The current implementation suffers from **integral windup** because:
- The integral term (`self.error_integral`) only increases
- No limits or reset mechanisms
- Can cause instability when error persists

### Solutions in Other Controllers
More advanced controllers in this codebase implement anti-windup:
- Integral clamping: `np.clip(integral, -limit, +limit)`
- Conditional integration
- Integral reset on setpoint changes

## Mathematical Interpretation

Each term provides different control characteristics:

1. **Proportional (P)**: Responds to current error magnitude
   - Fast response but may have steady-state error

2. **Integral (I)**: Responds to accumulated error over time
   - Eliminates steady-state error but may cause overshoot

3. **Derivative (D)**: Responds to rate of error change
   - Improves stability and reduces overshoot
   - Negative gain (-0.053) suggests tuned for damping


first adjust to control yawratedeg vs target_lataccel
gain scheduling = adjust the gain of the controller based on the state of the vehicle
"""

"""
whats my state, and what do I know right now?
state = what the system knows about the vehicle
desired behavior = target_lataccel, measured behavior = current_lataccel
in this ex PID chooses to project all that knowledge down to one scalar error: current_lataccel - target_lataccel
system knowledge = vEgo, aEgo, roll (not used)
"""
