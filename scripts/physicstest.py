#build kinematic bike
#linearize steer input theta with speed

import numpy as np
import matplotlib.pyplot as plt
from tinyphysics import TinyPhysicsModel, State, CONTEXT_LENGTH, STEER_RANGE


class KinematicBicycle:
    """
    Kinematic bicycle model for lateral dynamics.

    Equations:
        Turn radius: R = L / tan(δ)
        Lateral accel: a_lat = v² / R = v² * tan(δ) / L

    For small angles: tan(δ) ≈ δ
        a_lat ≈ v² * δ / L

    Therefore:
        gain = a_lat / δ = v² / L
        δ = a_lat * L / v²
    """

    def __init__(self, wheelbase: float = 2.7):
        """
        Args:
            wheelbase: Distance between front and rear axle (meters)
                       Typical car: 2.5-3.0m
        """
        self.L = wheelbase

    def steer_to_lataccel(self, steer, v_ego: float):
        """
        Convert steering angle to lateral acceleration.

        Args:
            steer: Steering angle (normalized units, not radians) - scalar or array
            v_ego: Vehicle speed (m/s)

        Returns:
            Lateral acceleration (m/s²)
        """
        # For tinyphysics, steer is in normalized units
        # We need to find the effective scaling
        return (v_ego ** 2) * np.asarray(steer) / self.L

    def lataccel_to_steer(self, lataccel: float, v_ego: float) -> float:
        """
        Convert desired lateral acceleration to steering angle.

        Args:
            lataccel: Desired lateral acceleration (m/s²)
            v_ego: Vehicle speed (m/s)

        Returns:
            Required steering angle (clipped to STEER_RANGE)
        """
        if v_ego < 1.0:
            return 0.0
        steer = lataccel * self.L / (v_ego ** 2)
        return np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])

    def gain(self, v_ego: float) -> float:
        """
        Get the linearization gain at a given speed.

        gain = d(lataccel) / d(steer) = v² / L
        """
        return (v_ego ** 2) / self.L


def fit_kinematic_model(model_path: str = "models/tinyphysics.onnx"):
    """
    Fit kinematic bicycle parameters to the tinyphysics model.

    Returns the effective wheelbase that best matches the learned model.
    """
    model = TinyPhysicsModel(model_path, debug=False)

    # Test at various speeds
    speeds = np.linspace(5, 35, 13)

    # Test steer angles
    steer_angles = np.linspace(-1.5, 1.5, 21)

    results = {}
    empirical_gains = []

    print("Probing tinyphysics model at various speeds...")
    print("-" * 50)

    for v_ego in speeds:
        lataccels = []

        for steer in steer_angles:
            state = State(roll_lataccel=0.0, v_ego=v_ego, a_ego=0.0)
            sim_states = [state] * CONTEXT_LENGTH
            actions = [steer] * CONTEXT_LENGTH
            past_preds = [0.0] * CONTEXT_LENGTH

            # Average multiple predictions (model is stochastic)
            predictions = []
            for _ in range(10):
                pred = model.get_current_lataccel(sim_states, actions, past_preds)
                predictions.append(pred)

            lataccels.append(np.mean(predictions))

        # Fit linear gain: lataccel = gain * steer
        lataccels = np.array(lataccels)
        gain = np.dot(steer_angles, lataccels) / np.dot(steer_angles, steer_angles)
        empirical_gains.append(gain)

        results[v_ego] = {
            'steer': steer_angles,
            'lataccel': lataccels,
            'gain': gain
        }
        print(f"v={v_ego:5.1f} m/s: empirical gain = {gain:.4f}")

    speeds = np.array(list(results.keys()))
    empirical_gains = np.array(empirical_gains)

    # For kinematic model: gain = v²/L
    # So L = v²/gain
    # Fit: gain = k * v² where k = 1/L
    # Using least squares on gain = k * v²
    v_squared = speeds ** 2
    k = np.dot(v_squared, empirical_gains) / np.dot(v_squared, v_squared)
    effective_wheelbase = 1.0 / k

    print("-" * 50)
    print(f"Fitted effective wheelbase: L = {effective_wheelbase:.3f} m")
    print("(Typical car wheelbase: 2.5-3.0 m)")

    return results, effective_wheelbase


def plot_comparison(results, wheelbase):
    """Compare kinematic model with empirical data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    speeds = np.array(list(results.keys()))
    empirical_gains = np.array([r['gain'] for r in results.values()])

    bike = KinematicBicycle(wheelbase)

    # Plot 1: Steer vs LatAccel curves
    ax1 = axes[0, 0]
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(results)))
    for (v_ego, data), color in zip(results.items(), colors):
        ax1.plot(data['steer'], data['lataccel'], 'o', color=color,
                markersize=4, alpha=0.6, label=f'v={v_ego:.0f} (data)')
        # Kinematic model line
        steer_line = np.linspace(-1.5, 1.5, 50)
        lataccel_line = bike.steer_to_lataccel(steer_line, v_ego)
        ax1.plot(steer_line, lataccel_line, '-', color=color, alpha=0.8)
    ax1.set_xlabel('Steer Angle')
    ax1.set_ylabel('Lateral Acceleration (m/s²)')
    ax1.set_title('Steer → LatAccel (dots=data, lines=kinematic model)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.axvline(0, color='k', linewidth=0.5)

    # Plot 2: Gain vs Speed
    ax2 = axes[0, 1]
    ax2.plot(speeds, empirical_gains, 'bo-', markersize=8, label='Empirical')
    kinematic_gains = [bike.gain(v) for v in speeds]
    ax2.plot(speeds, kinematic_gains, 'r--', linewidth=2, label=f'Kinematic (L={wheelbase:.2f}m)')
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Gain (lataccel/steer)')
    ax2.set_title('Linearization Gain vs Speed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Inverse gain (feedforward coefficient)
    ax3 = axes[1, 0]
    inv_empirical = 1.0 / empirical_gains
    inv_kinematic = [1.0 / bike.gain(v) for v in speeds]
    ax3.plot(speeds, inv_empirical, 'bo-', markersize=8, label='Empirical')
    ax3.plot(speeds, inv_kinematic, 'r--', linewidth=2, label=f'Kinematic (L={wheelbase:.2f}m)')
    ax3.set_xlabel('Speed (m/s)')
    ax3.set_ylabel('Steer per unit LatAccel')
    ax3.set_title('Feedforward Coefficient (for controller use)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model fit error
    ax4 = axes[1, 1]
    error_pct = (empirical_gains - kinematic_gains) / empirical_gains * 100
    ax4.bar(speeds, error_pct, width=2, color='steelblue', alpha=0.7)
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.set_xlabel('Speed (m/s)')
    ax4.set_ylabel('Error (%)')
    ax4.set_title('Kinematic Model Error vs Empirical')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('linearization_map.png', dpi=150)
    plt.show()


def create_linear_map(wheelbase: float):
    """
    Create the linear map functions for controller use.
    """
    bike = KinematicBicycle(wheelbase)

    print("\n" + "=" * 60)
    print("LINEAR MAP FOR CONTROLLER USE")
    print("=" * 60)
    print(f"\nKinematic Bicycle Model (L = {wheelbase:.3f} m)")
    print("\nEquations:")
    print("  lataccel = (v² / L) * steer")
    print("  steer = (L / v²) * lataccel")
    print("\nGain function:")
    print(f"  gain(v) = v² / {wheelbase:.3f}")
    print("\nFeedforward steer command:")
    print(f"  steer_ff = {wheelbase:.3f} * target_lataccel / v²")

    print("\n" + "-" * 60)
    print("Lookup table (for quick reference):")
    print("-" * 60)
    print(f"{'Speed (m/s)':<15} {'Gain':<15} {'Steer/LatAccel':<15}")
    print("-" * 60)
    for v in [5, 10, 15, 20, 25, 30, 35]:
        g = bike.gain(v)
        print(f"{v:<15} {g:<15.4f} {1/g:<15.4f}")

    return bike


if __name__ == "__main__":
    print("=" * 60)
    print("KINEMATIC BICYCLE MODEL LINEARIZATION")
    print("=" * 60)

    # Fit kinematic model to tinyphysics
    results, wheelbase = fit_kinematic_model()

    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(results, wheelbase)

    # Create linear map
    bike = create_linear_map(wheelbase)

    print("\n" + "=" * 60)
    print("Example feedforward calculations:")
    print("=" * 60)
    for v in [10, 20, 30]:
        for target_lat in [1.0, 2.0]:
            steer = bike.lataccel_to_steer(target_lat, v)
            print(f"v={v:2} m/s, target_lataccel={target_lat} m/s² → steer={steer:.4f}")

    print("\nLinearization map saved to 'linearization_map.png'")
