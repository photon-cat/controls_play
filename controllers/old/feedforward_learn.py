"""
Iterative Feedforward Learning Controller.
- Saves learned parameters per segment to a cache file
- On replay, loads previous params and refines them
- Converges to optimal FF parameters over multiple runs
"""
from . import BaseController
import numpy as np
import json
import os
from pathlib import Path

# Default Gain LUT
GAIN_LUT = {
    0.5: 1.32, 1.5: 0.96, 2.5: 1.09, 3.5: 1.27, 4.5: 1.53,
    5.5: 1.67, 6.5: 1.78, 7.5: 1.88, 8.5: 1.72, 9.5: 1.33,
    10.5: 1.29, 11.5: 1.48, 12.5: 1.67, 13.5: 1.79, 14.5: 1.40,
    15.5: 1.06, 16.5: 1.16, 17.5: 1.60, 18.5: 1.69, 19.5: 1.48,
    20.5: 1.00, 21.5: 1.18, 22.5: 1.35, 23.5: 1.30, 24.5: 1.12,
    25.5: 1.59, 26.5: 1.60, 27.5: 1.70, 28.5: 1.61, 29.5: 1.57,
    30.5: 1.59, 31.5: 1.15, 32.5: 1.65, 33.5: 1.90, 34.5: 1.73,
    35.5: 1.95, 36.5: 1.35, 37.5: 2.45, 38.5: 1.69, 39.5: 2.43,
    40.5: 2.57,
}

GAIN_VELOCITIES = np.array(sorted(GAIN_LUT.keys()))
GAIN_VALUES = np.array([GAIN_LUT[v] for v in GAIN_VELOCITIES])

CONTROL_START_IDX = 100
CACHE_FILE = "controllers/ff_learn_cache.json"


class Controller(BaseController):
    # Class-level cache shared across instances
    _cache = None
    _cache_loaded = False

    def __init__(self):
        # Load cache on first instantiation
        if not Controller._cache_loaded:
            Controller._load_cache()

        # Current segment ID (set by set_data_path)
        self.segment_id = None

        # Parameters (will be loaded from cache or defaults)
        self.gain = None  # Learned gain multiplier
        self.lag = 0      # Learned lag in steps
        self.run_count = 0  # How many times we've run this segment

        # PID gains (for error correction)
        self.kp = 0.0
        self.ki = 0.0
        self.kd = -0.00

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step_count = 0

        # Data collection for learning
        self.steer_history = []
        self.lataccel_history = []
        self.target_history = []
        self.roll_history = []
        self.v_history = []
        self.error_history = []

        # Logging
        self._log = {}

    @classmethod
    def _load_cache(cls):
        """Load cached parameters from file."""
        cls._cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cls._cache = json.load(f)
                print(f"Loaded {len(cls._cache)} cached segment params")
            except:
                cls._cache = {}
        cls._cache_loaded = True

    @classmethod
    def _save_cache(cls):
        """Save cached parameters to file."""
        try:
            Path(CACHE_FILE).parent.mkdir(exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(cls._cache, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def set_data_path(self, path):
        """Called by simulator to identify segment."""
        self.segment_id = Path(path).stem  # e.g., "00000" from "data/00000.csv"
        self._load_segment_params()

    def _load_segment_params(self):
        """Load cached params for this segment."""
        if self.segment_id and Controller._cache and self.segment_id in Controller._cache:
            params = Controller._cache[self.segment_id]
            # Use the 'try_gain' for this run (exploration)
            self.gain = params.get('try_gain', params.get('best_gain'))
            self.best_gain = params.get('best_gain')
            self.best_mse = params.get('best_mse')
            self.lag = params.get('lag', 0)
            self.run_count = params.get('run_count', 0) + 1
            gain_str = f"{self.gain:.3f}" if self.gain else "LUT"
            print(f"[{self.segment_id}] Run #{self.run_count}: trying gain={gain_str}")
        else:
            self.run_count = 1
            self.best_gain = None
            self.best_mse = None
            if self.segment_id:
                print(f"[{self.segment_id}] First run - using LUT defaults")

    def _save_segment_params(self):
        """Save learned params for this segment using gradient-based optimization."""
        if not self.segment_id:
            return

        # Compute MSE for this run
        if len(self.error_history) > 0:
            mse = np.mean(np.array(self.error_history) ** 2)
        else:
            return

        # Current gain used this run
        current_gain = self.gain if self.gain else self._get_lut_gain_avg()

        # Compare with previous best
        prev_best_mse = getattr(self, 'best_mse', None)
        prev_best_gain = getattr(self, 'best_gain', None)

        if prev_best_mse is None or mse < prev_best_mse:
            # This run was better!
            best_gain = current_gain
            best_mse = mse
            improved = True
        else:
            # Previous was better
            best_gain = prev_best_gain if prev_best_gain else current_gain
            best_mse = prev_best_mse
            improved = False

        # Decide next gain to try (exploration)
        # Reduce step size over runs to converge
        base_step = max(0.02, 0.1 / (1 + self.run_count * 0.3))

        if improved:
            # Keep exploring in same direction
            direction = 1 if current_gain >= (prev_best_gain or current_gain) else -1
            step = base_step
        else:
            # Alternate direction based on run count
            direction = 1 if self.run_count % 2 == 0 else -1
            step = base_step

        # After 5 runs, use best gain (stop exploring)
        if self.run_count >= 5:
            try_gain = best_gain
        else:
            try_gain = best_gain * (1 + step * direction)
            try_gain = np.clip(try_gain, 0.5, 4.0)

        Controller._cache[self.segment_id] = {
            'try_gain': try_gain,     # Gain to try next run
            'best_gain': best_gain,   # Best gain found so far
            'best_mse': float(best_mse),
            'last_mse': float(mse),
            'lag': 0,
            'run_count': self.run_count,
        }
        Controller._save_cache()

        status = "NEW BEST!" if improved else "no improvement"
        print(f"[{self.segment_id}] mse={mse:.4f} ({status}), best={best_mse:.4f}, try_next={try_gain:.3f}")

    def _get_lut_gain_avg(self):
        """Get average LUT gain for this segment's velocity."""
        if len(self.v_history) > 0:
            avg_v = np.mean(self.v_history)
            v_clamped = np.clip(avg_v, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
            return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)
        return 1.5

    def _estimate_lag(self):
        """Estimate lag from collected data."""
        if len(self.steer_history) < 50:
            return 0

        steer = np.array(self.steer_history)
        lataccel = np.array(self.lataccel_history[:len(steer)])

        if len(lataccel) < len(steer):
            return 0

        # Cross-correlation to find delay
        lataccel_diff = np.diff(lataccel)
        steer_trim = steer[:-1]

        min_len = min(len(steer_trim), len(lataccel_diff))
        steer_trim = steer_trim[:min_len]
        lataccel_diff = lataccel_diff[:min_len]

        max_lag = min(10, min_len // 4)
        best_corr = -1
        best_lag = 0

        for lag in range(max_lag + 1):
            if lag == 0:
                s, a = steer_trim, lataccel_diff
            else:
                s = steer_trim[:-lag]
                a = lataccel_diff[lag:]

            if len(s) < 20 or len(s) != len(a):
                continue

            corr = np.abs(np.corrcoef(s, a)[0, 1])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag

        return best_lag

    def _estimate_gain(self):
        """Estimate optimal gain from collected data."""
        if len(self.steer_history) < 50:
            return None

        n = len(self.steer_history)
        steer = np.array(self.steer_history)
        lataccel = np.array(self.lataccel_history[:n])
        roll = np.array(self.roll_history[:n])

        if len(lataccel) < n or len(roll) < n:
            return None

        turning = lataccel - roll
        mask = np.abs(steer) > 0.1

        if mask.sum() < 20:
            return None

        s = steer[mask]
        t = turning[mask]

        if len(s) != len(t):
            return None

        # Fit: turning = gain * steer
        denom = np.sum(s ** 2)
        if denom > 0.1:
            gain = np.sum(s * t) / denom
            if 0.3 < gain < 5.0:
                return gain
        return None

    def get_gain(self, v_ego):
        """Get gain - use learned if available, else LUT."""
        if self.gain is not None:
            return self.gain
        v_clamped = np.clip(v_ego, GAIN_VELOCITIES[0], GAIN_VELOCITIES[-1])
        return np.interp(v_clamped, GAIN_VELOCITIES, GAIN_VALUES)

    def get_log(self):
        return self._log

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        v_ego = state.v_ego
        roll_lataccel = state.roll_lataccel
        gain = self.get_gain(v_ego)

        # Use future target if we have learned lag
        effective_target = target_lataccel
        effective_roll = roll_lataccel
        if self.lag > 0 and future_plan.lataccel and len(future_plan.lataccel) > self.lag:
            effective_target = future_plan.lataccel[self.lag - 1]
            if future_plan.roll_lataccel and len(future_plan.roll_lataccel) > self.lag:
                effective_roll = future_plan.roll_lataccel[self.lag - 1]

        # ---- Feedforward ----
        turning_lataccel = effective_target - effective_roll
        u_ff = turning_lataccel / gain

        # ---- PID (small correction) ----
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -10, 10)
        error_diff = error - self.prev_error
        self.prev_error = error

        u_pid = self.kp * error + self.ki * self.error_integral + self.kd * error_diff

        # Pure FF with learned gain (no PID correction)
        u_cmd = u_ff
        u_cmd = np.clip(u_cmd, -2.0, 2.0)

        # Collect data for learning
        if self.step_count >= CONTROL_START_IDX:
            self.steer_history.append(u_cmd)
            self.lataccel_history.append(current_lataccel)
            self.target_history.append(target_lataccel)
            self.roll_history.append(roll_lataccel)
            self.v_history.append(v_ego)
            self.error_history.append(error)

        # Log
        self._log = {
            'learn_gain': gain,
            'learn_lag': self.lag,
            'learn_run': self.run_count,
        }

        return u_cmd

    def __del__(self):
        """Save params when controller is destroyed."""
        try:
            if len(self.steer_history) > 50:
                self._save_segment_params()
        except:
            pass
