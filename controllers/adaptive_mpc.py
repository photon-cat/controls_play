# controllers/adaptive_mpc.py
import numpy as np
from controllers import BaseController  # assuming this is how BaseController is exposed

class Controller(BaseController):
    def __init__(self):
        # Previous steering commands and lataccel values
        self.u_prev = 0.0           # last applied command
        self.u_prev2 = 0.0
        self.applied_samples = 0
        self.y_prev = 0.0           # compensated latacc
        self.y_prev2 = 0.0

        # Online plant gain estimate g_k (lataccel change per steering change)
        self.g_hat = 0.5    # initial guess
        self.g_min = 0.05   # clamp to avoid crazy gains
        self.g_max = 5.0

        # Base weights for error vs smoothness (tune these)
        self.w_e_base = 1.0
        self.w_du_base = 0.3

        # For numerical robustness
        self.min_du = 1e-3
        self.alpha_gain_filter = 0.1  # low-pass on g_hat
        self.initialized = False
        self.steer_limit = 2.0

        # Feedforward steering gain a_ref -> steer (adapted online)
        self.ff_gain = 3.0
        self.ff_gain_min = 0.5
        self.ff_gain_max = 6.0
        self.ff_alpha = 0.02

        # Roll compensation and shaping
        self.roll_comp_gain = 0.85
        self.max_latacc_delta = 0.5   # matches simulator clamp
        self.lataccel_limit = 5.0
        self.max_delta_u = 0.4
        self.rate_damping = 0.7
        self.preview_steps = 5

    def _apply_rate_damping(self, error, latacc_rate):
        # When latacc is already moving towards the target, reduce aggressive corrections
        if error * latacc_rate > 0:
            rate_norm = min(abs(latacc_rate) / (self.max_latacc_delta + 1e-3), 2.0)
            error *= 1.0 / (1.0 + self.rate_damping * rate_norm)
        return error

    def _update_ff_gain(self, y_comp):
        if abs(self.u_prev) < 0.1:
            return
        denom = abs(self.u_prev)
        if denom < 1e-3:
            return
        g_est = abs(y_comp) / denom
        if not np.isfinite(g_est):
            return
        g_est = np.clip(g_est, self.ff_gain_min, self.ff_gain_max)
        self.ff_gain = (1.0 - self.ff_alpha) * self.ff_gain + self.ff_alpha * g_est
    def observe_applied_action(self, applied_action: float):
        """
        Let the controller know what steering command actually hit the plant.
        This keeps the online gain estimate and control integrator aligned even
        when the simulator overrides/clamps our command.
        """
        applied = float(applied_action)
        self.u_prev2 = self.u_prev
        self.u_prev = applied
        self.applied_samples += 1

    def _update_gain_estimate(self, y_k, y_km1, u_km1, u_km2):
        dy = y_k - y_km1
        du = u_km1 - u_km2
        if abs(du) > self.min_du:
            g_new = dy / du
            # Filter & clamp
            g_new = np.clip(g_new, self.g_min, self.g_max)
            self.g_hat = (
                (1.0 - self.alpha_gain_filter) * self.g_hat
                + self.alpha_gain_filter * g_new
            )
        # If du is tiny, keep previous g_hat

    def _compute_adaptive_weights(self, future_plan):
        # Look at next 1–2 seconds to gauge curvature
        if future_plan is None or len(future_plan.lataccel) == 0:
            return self.w_e_base, self.w_du_base

        lookahead = min(20, len(future_plan.lataccel))  # 2 seconds at 10 Hz
        future_lat = np.array(future_plan.lataccel[:lookahead])
        curvature_level = float(np.mean(np.abs(future_lat)))  # m/s^2

        # Normalize curvature to [0, ~1] range-ish
        curvature_norm = np.clip(curvature_level / 3.0, 0.0, 2.0)

        # More curvature -> more tracking, less jerk penalty
        w_e = self.w_e_base * (1.0 + 1.5 * curvature_norm)
        w_du = self.w_du_base / (1.0 + 1.0 * curvature_norm)

        return w_e, w_du

    def update(self, target_lataccel, current_lataccel, state, future_plan=None):
        """
        target_lataccel: scalar current target
        current_lataccel: scalar measured lataccel
        state: State(roll_lataccel, v_ego, a_ego)
        future_plan: FuturePlan(...) or None
        returns: steering command (float)
        """
        roll = float(state.roll_lataccel)
        roll_comp = self.roll_comp_gain * roll
        y_k = float(current_lataccel) - roll_comp
        target = float(target_lataccel) - roll_comp

        if not self.initialized:
            # First call: just initialize memory
            self.y_prev2 = self.y_prev = y_k
            self.initialized = True

        # 1) Online gain estimate g_hat (once we have real action history)
        if self.applied_samples >= 2:
            self._update_gain_estimate(
                y_k=y_k,
                y_km1=self.y_prev,
                u_km1=self.u_prev,
                u_km2=self.u_prev2,
            )
        g = self.g_hat

        # 2) Choose reference using preview: look ahead a few steps
        a_ref = target
        if future_plan is not None and len(future_plan.lataccel) > 0:
            lookahead = min(self.preview_steps, len(future_plan.lataccel))
            future_targets = np.array(future_plan.lataccel[:lookahead])
            future_roll = (
                np.array(future_plan.roll_lataccel[:lookahead])
                if len(future_plan.roll_lataccel) >= lookahead
                else np.zeros(lookahead)
            )
            future_comp = future_targets - self.roll_comp_gain * future_roll
            preview_ref = float(np.mean(future_comp))
            a_ref = 0.6 * target + 0.4 * preview_ref

        # Tracking error in lataccel
        e = a_ref - y_k
        latacc_rate = y_k - self.y_prev
        e = self._apply_rate_damping(e, latacc_rate)

        # 3) Adaptive weights (dynamic tuning) based on upcoming curvature
        w_e, w_du = self._compute_adaptive_weights(future_plan)

        # 4) One-step predictive optimal Δu:
        # J(Δu) = w_e (e - g Δu)^2 + w_du (Δu)^2  -> closed-form Δu
        denom = w_e * g * g + w_du
        if denom <= 1e-6:
            delta_u = 0.0
        else:
            delta_u = (w_e * g / denom) * e
        delta_u = np.clip(delta_u, -self.max_delta_u, self.max_delta_u)

        # Feedforward steering: convert desired lataccel to steer using adaptive gain
        self._update_ff_gain(y_k)
        ff_gain = max(self.ff_gain, self.ff_gain_min)
        desired_latacc = np.clip(a_ref, -self.lataccel_limit, self.lataccel_limit)
        u_ff = np.clip(desired_latacc / ff_gain, -self.steer_limit, self.steer_limit)
        u_ff = np.clip(u_ff, -self.steer_limit, self.steer_limit)

        # 5) New steering command blends feedforward with corrective delta
        u_cmd = u_ff + delta_u
        u_cmd = np.clip(u_cmd, -self.steer_limit, self.steer_limit)

        # Update history for next step
        self.y_prev2 = self.y_prev
        self.y_prev = y_k

        return float(u_cmd)
