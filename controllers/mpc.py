from . import BaseController
import numpy as np

DT = 0.1
STEER_LIMIT = 2.0

GAIN_LUT = {
  0.5: 1.32,
  1.5: 0.96,
  2.5: 1.09,
  3.5: 1.27,
  4.5: 1.53,
  5.5: 1.67,
  6.5: 1.78,
  7.5: 1.88,
  8.5: 1.72,
  9.5: 1.33,
  10.5: 1.29,
  11.5: 1.48,
  12.5: 1.67,
  13.5: 1.79,
  14.5: 1.40,
  15.5: 1.06,
  16.5: 1.16,
  17.5: 1.60,
  18.5: 1.69,
  19.5: 1.48,
  20.5: 1.00,
  21.5: 1.18,
  22.5: 1.35,
  23.5: 1.30,
  24.5: 1.12,
  25.5: 1.59,
  26.5: 1.60,
  27.5: 1.70,
  28.5: 1.61,
  29.5: 1.57,
  30.5: 1.59,
  31.5: 1.15,
  32.5: 1.65,
  33.5: 1.90,
  34.5: 1.73,
  35.5: 1.95,
  36.5: 1.35,
  37.5: 2.45,
  38.5: 1.69,
  39.5: 2.43,
  40.5: 2.57,
}

GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])


class SimpleLataccelModel:
  def __init__(self, tau=0.35, max_delta=0.5):
    self.tau = tau
    self.max_delta = max_delta

  def gain(self, v_ego):
    v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
    return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

  def predict(self, current_lataccel, steer_cmd, v_ego, roll_lataccel):
    gain = self.gain(v_ego)
    target_lataccel = roll_lataccel + gain * steer_cmd
    alpha = DT / max(self.tau, DT)
    next_lataccel = current_lataccel + alpha * (target_lataccel - current_lataccel)
    next_lataccel = np.clip(
      next_lataccel,
      current_lataccel - self.max_delta,
      current_lataccel + self.max_delta,
    )
    return next_lataccel, gain


class Controller(BaseController):
  def __init__(self):
    self.model = SimpleLataccelModel()

    # MPC params
    self.H = 8
    self.num_samples = 64
    self.num_iters = 3
    self.elite_frac = 0.2

    self.w_tracking = 1.0
    self.w_effort = 0.08
    self.w_rate = 0.3
    self.w_jerk = 0.2

    self.prev_u = 0.0
    self.prev_solution = None
    self.rng = np.random.default_rng(0)

    self._log = {}

  def get_log(self):
    return self._log

  def _reference_sequence(self, target_lataccel, future_plan):
    if hasattr(future_plan, 'lataccel') and future_plan.lataccel:
      refs = future_plan.lataccel[:self.H]
    else:
      refs = [target_lataccel] * self.H
    return np.asarray(refs, dtype=float)

  def _sequence_from_gain(self, refs, roll_seq, v_seq):
    gains = np.array([self.model.gain(v) for v in v_seq])
    turning = refs - roll_seq
    u_ff = np.divide(turning, gains, out=np.zeros_like(turning), where=gains != 0.0)
    return np.clip(u_ff, -STEER_LIMIT, STEER_LIMIT)

  def _simulate_cost(self, u_seq, current_lataccel, refs, roll_seq, v_seq):
    cost = 0.0
    lataccel = current_lataccel
    prev_u = self.prev_u
    prev_lataccel = current_lataccel

    for i in range(self.H):
      lataccel, _ = self.model.predict(lataccel, u_seq[i], v_seq[i], roll_seq[i])
      error = lataccel - refs[i]
      du = u_seq[i] - prev_u
      jerk = lataccel - prev_lataccel
      cost += self.w_tracking * error * error
      cost += self.w_effort * u_seq[i] * u_seq[i]
      cost += self.w_rate * du * du
      cost += self.w_jerk * jerk * jerk
      prev_u = u_seq[i]
      prev_lataccel = lataccel
    return cost

  def _optimize(self, current_lataccel, refs, roll_seq, v_seq, u_seed):
    mean = np.copy(u_seed)
    std = np.full(self.H, 0.35)
    num_elite = max(1, int(self.num_samples * self.elite_frac))

    for _ in range(self.num_iters):
      samples = mean + std * self.rng.standard_normal((self.num_samples, self.H))
      samples = np.clip(samples, -STEER_LIMIT, STEER_LIMIT)

      costs = np.zeros(self.num_samples)
      for i in range(self.num_samples):
        costs[i] = self._simulate_cost(samples[i], current_lataccel, refs, roll_seq, v_seq)

      elite_idx = np.argsort(costs)[:num_elite]
      elite = samples[elite_idx]
      mean = np.mean(elite, axis=0)
      std = np.std(elite, axis=0) + 1e-3

    return mean, float(np.min(costs))

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    refs = self._reference_sequence(target_lataccel, future_plan)
    v_seq = np.array([state.v_ego] * self.H, dtype=float)
    roll_seq = np.array([state.roll_lataccel] * self.H, dtype=float)

    if hasattr(future_plan, 'v_ego') and future_plan.v_ego:
      v_seq[:min(self.H, len(future_plan.v_ego))] = np.array(future_plan.v_ego[:self.H])
    if hasattr(future_plan, 'roll_lataccel') and future_plan.roll_lataccel:
      roll_seq[:min(self.H, len(future_plan.roll_lataccel))] = np.array(future_plan.roll_lataccel[:self.H])

    u_ff = self._sequence_from_gain(refs, roll_seq, v_seq)
    if self.prev_solution is None:
      u_seed = u_ff
    else:
      u_seed = np.concatenate([self.prev_solution[1:], [self.prev_solution[-1]]])
      u_seed = 0.7 * u_seed + 0.3 * u_ff

    best_seq, best_cost = self._optimize(current_lataccel, refs, roll_seq, v_seq, u_seed)

    u_cmd = 0.6 * self.prev_u + 0.4 * best_seq[0]
    u_cmd = np.clip(u_cmd, -STEER_LIMIT, STEER_LIMIT)

    self.prev_solution = best_seq
    self.prev_u = u_cmd

    self._log = {
      'mpc_u_cmd': u_cmd,
      'mpc_u_ff': float(u_ff[0]),
      'mpc_best_cost': best_cost,
      'mpc_ref0': float(refs[0]),
      'mpc_gain': float(self.model.gain(state.v_ego)),
      'mpc_mode': 'simple_cem',
    }

    return u_cmd
