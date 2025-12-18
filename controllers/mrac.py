from . import BaseController
import numpy as np

class Controller(BaseController):
  def __init__(self):
    # preview + baseline feedforward (in curvature space)
    self.preview_steps = 3
    self.k_ff = 0.8

    # MRAC parameters
    self.theta = np.array([0.0, 0.0], dtype=float)   # [theta_ref, theta_meas]
    self.Gamma = np.array([0.002, 0.002], dtype=float)  # learning rates
    self.theta_min = np.array([-2.0, -2.0])
    self.theta_max = np.array([+2.0, +2.0])

    # reference model in curvature space (first-order)
    self.k_m = 0.0
    self.a_m_star = 0.2  # 1 / tau_m per step

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # -----------------------------
    # 1. Preview target lat accel
    # -----------------------------
    if (
        future_plan is not None
        and hasattr(future_plan, "lataccel")
        and len(future_plan.lataccel) > self.preview_steps
    ):
        a_ref = float(future_plan.lataccel[self.preview_steps])
    else:
        a_ref = float(target_lataccel)

    # -----------------------------
    # 2. Speed + curvature conversion
    # -----------------------------
    v = max(getattr(state, "vEgo", getattr(state, "v_ego", 0.0)), 6.0)
    v2 = v * v

    k_ref  = a_ref / v2
    k_meas = current_lataccel / v2

    # -----------------------------
    # 3. Reference model (curvature)
    # -----------------------------
    self.k_m = self.k_m + self.a_m_star * (k_ref - self.k_m)
    e_m = k_meas - self.k_m   # MRAC tracking error

    # -----------------------------
    # 4. Baseline control (REQUIRED)
    # -----------------------------
    # Feedforward: known physics
    u_ff = self.k_ff * k_ref

    # Stabilizing feedback (this was missing)
    k_p = 3.0
    u_fb = k_p * (k_ref - k_meas)

    # -----------------------------
    # 5. MRAC adaptive trim
    # -----------------------------
    # scale curvature for numerical conditioning
    k_ref_s  = k_ref  * v2
    k_meas_s = k_meas * v2

    phi = np.array([k_ref_s, -k_meas_s], dtype=float)

    denom = 1.0 + float(phi @ phi)

    self.theta = self.theta - (self.Gamma * (phi * e_m) / denom)
    self.theta = np.clip(self.theta, self.theta_min, self.theta_max)

    u_ad = float(self.theta @ phi)

    # -----------------------------
    # 6. Total steering command
    # -----------------------------
    return u_ff + u_fb + u_ad

