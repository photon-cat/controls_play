"""
Kinematic bicycle model with tire model for approximating vehicle dynamics.

Parameters to tune:
- L: wheelbase (m)
- a: distance from CG to front axle (m)
- b: distance from CG to rear axle (m)
- C_f: front cornering stiffness (N/rad)
- C_r: rear cornering stiffness (N/rad)
- mass: vehicle mass (kg)
- steering_ratio: ratio of steer_command to front wheel angle
"""

import numpy as np

ACC_G = 9.81


class BicycleModel:
    def __init__(self, L=2.7, a=1.2, b=1.5, C_f=80000, C_r=80000,
                 mass=1500, steering_ratio=15.0):
        """
        Initialize bicycle model with vehicle parameters.

        Args:
            L: wheelbase (m) - distance between front and rear axles
            a: distance from CG to front axle (m)
            b: distance from CG to rear axle (m)
            C_f: front cornering stiffness (N/rad)
            C_r: rear cornering stiffness (N/rad)
            mass: vehicle mass (kg)
            steering_ratio: ratio converting steer_command to wheel angle (deg/unit)
        """
        self.L = L
        self.a = a
        self.b = b
        self.C_f = C_f
        self.C_r = C_r
        self.mass = mass
        self.steering_ratio = steering_ratio

        # Sanity check
        assert abs(a + b - L) < 0.1, f"Wheelbase mismatch: a+b={a+b}, L={L}"

    def predict_lataccel(self, v_ego, steer_command, roll_lataccel=0.0):
        """
        Predict lateral acceleration from steer command and velocity.

        Args:
            v_ego: vehicle velocity (m/s)
            steer_command: steering command (unitless, typically -2 to 2)
            roll_lataccel: lateral acceleration contribution from road banking (m/s^2)

        Returns:
            lateral_acceleration: predicted lateral acceleration (m/s^2)
        """
        # Avoid division by zero at low speeds
        if v_ego < 0.1:
            return roll_lataccel

        # Convert steer command to front wheel angle (radians)
        # Negative sign because left-turn positive convention
        delta_f = -np.deg2rad(steer_command * self.steering_ratio)

        # Kinematic bicycle model
        # For low speeds/simplified model, use kinematic approximation:
        # beta ≈ (b/L) * tan(delta_f)  (vehicle sideslip angle)
        # But for dynamic model with tire forces:

        # Steady-state lateral acceleration (simplified)
        # From bicycle model: a_y = v^2 / R
        # where R = L / tan(delta_f) for small angles

        # For small angles: tan(delta_f) ≈ delta_f
        if abs(delta_f) < 0.01:
            # Very small steering, use linear approximation
            R = self.L / delta_f if abs(delta_f) > 1e-6 else 1e6
        else:
            R = self.L / np.tan(delta_f)

        # Lateral acceleration from turning
        turning_lataccel = (v_ego ** 2) / R if abs(R) > 0.1 else 0.0

        # Add road banking contribution
        total_lataccel = turning_lataccel + roll_lataccel

        # Limit to reasonable range
        total_lataccel = np.clip(total_lataccel, -10, 10)

        return total_lataccel

    def predict_lataccel_dynamic(self, v_ego, steer_command, roll_lataccel=0.0, yaw_rate=0.0):
        """
        Predict lateral acceleration using dynamic bicycle model with tire forces.

        This is more accurate than kinematic model at higher speeds.

        Args:
            v_ego: vehicle velocity (m/s)
            steer_command: steering command
            roll_lataccel: road banking contribution (m/s^2)
            yaw_rate: current yaw rate (rad/s) - for dynamic model

        Returns:
            lateral_acceleration: predicted lateral acceleration (m/s^2)
        """
        if v_ego < 0.1:
            return roll_lataccel

        # Convert steer command to front wheel angle
        delta_f = -np.deg2rad(steer_command * self.steering_ratio)

        # Vehicle sideslip angle at CG
        # beta = arctan(v_y / v_x) ≈ v_y / v_x for small angles
        # Assume small angles for now

        # Front and rear slip angles (simplified steady-state)
        # alpha_f = delta_f - beta - (a * yaw_rate) / v_x
        # alpha_r = -beta + (b * yaw_rate) / v_x

        # For steady-state circular motion:
        # yaw_rate = v / R
        # beta = arctan(b * tan(delta_f) / L) ≈ b * delta_f / L

        beta = (self.b * delta_f) / self.L

        # Slip angles
        alpha_f = delta_f - beta - (self.a * yaw_rate) / v_ego if v_ego > 0.1 else 0
        alpha_r = -beta + (self.b * yaw_rate) / v_ego if v_ego > 0.1 else 0

        # Lateral tire forces (linear tire model)
        F_yf = -self.C_f * alpha_f
        F_yr = -self.C_r * alpha_r

        # Total lateral force
        F_y = F_yf + F_yr

        # Lateral acceleration from tire forces
        turning_lataccel = F_y / self.mass

        # Add road banking
        total_lataccel = turning_lataccel + roll_lataccel

        # Limit to reasonable range
        total_lataccel = np.clip(total_lataccel, -10, 10)

        return total_lataccel


class SimplifiedBicycleModel(BicycleModel):
    """
    Simplified version with fewer parameters, easier to tune.
    """
    def __init__(self, steer_gain=0.7, steer_ratio=15.0):
        """
        Ultra-simplified model with just two parameters.

        Args:
            steer_gain: how much lateral accel per unit steer at 1 m/s
            steer_ratio: steering sensitivity
        """
        super().__init__()
        self.steer_gain = steer_gain
        self.steer_ratio = steer_ratio

    def predict_lataccel(self, v_ego, steer_command, roll_lataccel=0.0):
        """
        Simple linear relationship: lataccel = steer_gain * steer_command * v_ego

        This is empirically tuned to match TinyPhysics behavior.
        """
        if v_ego < 0.1:
            return roll_lataccel

        # Linear approximation
        turning_lataccel = self.steer_gain * steer_command * v_ego

        # Add road banking
        total_lataccel = turning_lataccel + roll_lataccel

        # Clip
        total_lataccel = np.clip(total_lataccel, -5, 5)

        return total_lataccel
