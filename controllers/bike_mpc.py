import numpy as np

class Bicycle():
    def __init__(self):
        # pose
        self.xc = 0.0
        self.yc = 0.0
        self.theta = 0.0

        # steering state
        self.delta = 0.0   # actual steer angle (rad)
        self.beta = 0.0    # slip angle (rad)

        # geometry
        self.L = 2.0
        self.lr = 1.2

        # steering limits
        self.delta_max = np.deg2rad(35.0)   # max physical steer
        self.u_max = 2.0                    # controller units
        self.w_max = 1.22                   # rad/s steer rate

        # tire model
        self.alpha_sat = np.deg2rad(9.0)    # slip saturation angle

        # timing
        self.sample_time = 0.01

    def reset(self):
        self.xc = 0.0
        self.yc = 0.0
        self.theta = 0.0
        self.delta = 0.0
        self.beta = 0.0

    def step(self, v, u):
        """
        v : forward velocity (m/s)
        u : steering command in [-2, 2]
        """
        dt = self.sample_time

        # --- map unitless command to physical steer ---
        delta_cmd = (u / self.u_max) * self.delta_max
        delta_cmd = np.clip(delta_cmd, -self.delta_max, self.delta_max)

        # --- steering rate limit ---
        delta_dot = np.clip(
            (delta_cmd - self.delta) / dt,
            -self.w_max,
            self.w_max
        )
        self.delta += delta_dot * dt

        # --- slip angle ---
        self.beta = np.arctan((self.lr / self.L) * np.tan(self.delta))

        # --- front tire slip ---
        alpha_f = self.delta - self.beta

        # --- tire saturation (understeer) ---
        alpha_eff = self.alpha_sat * np.tanh(alpha_f / self.alpha_sat)

        # --- effective steer after tire slip ---
        delta_eff = self.beta + alpha_eff

        # --- integrate vehicle motion ---
        self.xc += v * np.cos(self.theta + self.beta) * dt
        self.yc += v * np.sin(self.theta + self.beta) * dt

        # yaw rate
        self.theta += (v / self.L) * np.tan(delta_eff) * dt
