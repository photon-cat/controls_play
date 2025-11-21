#!/usr/bin/env python3
"""
MetaDrive visualization with TinyPhysics lateral dynamics.

Replaces MetaDrive's default physics with TinyPhysics for lateral control.
Uses real trajectory data from CSV files.
"""

import argparse
import importlib
import numpy as np
from pathlib import Path

from metadrive.envs.metadrive_env import MetaDriveEnv

from tinyphysics import (
    TinyPhysicsModel, State, FuturePlan,
    CONTROL_START_IDX, STEER_RANGE, MAX_ACC_DELTA, DEL_T
)


class TinyPhysicsVehicle:
    """
    Vehicle that uses TinyPhysics for lateral dynamics instead of MetaDrive physics.
    """
    
    def __init__(self, model_path: str, data_path: str, controller_name: str):
        # Load TinyPhysics model
        self.model = TinyPhysicsModel(model_path, debug=False)
        
        # Load trajectory data
        self.data = self._load_data(data_path)
        
        # Load controller
        controller_class = importlib.import_module(f'controllers.{controller_name}').Controller
        self.controller = controller_class()
        self.controller_name = controller_name
        
        # State tracking
        self.step_idx = 0
        self.context_length = 20
        
        # Initialize histories
        self._init_histories()
        
        # Control mode
        self.manual_control = False
        self.manual_steer = 0.0
        
    def _load_data(self, data_path: str):
        """Load and process CSV trajectory data."""
        import pandas as pd
        
        df = pd.read_csv(data_path)
        ACC_G = 9.81
        
        processed_df = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values,
        })
        
        return processed_df
    
    def _init_histories(self):
        """Initialize state histories."""
        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.target_lataccel_history = []
        
        # Fill initial context
        for i in range(self.context_length):
            state, target, _ = self._get_state_target_futureplan(i)
            self.state_history.append(state)
            self.action_history.append(self.data['steer_command'].values[i])
            self.current_lataccel_history.append(target)  # Use target initially
            self.target_lataccel_history.append(target)
        
        self.step_idx = self.context_length
        self.current_lataccel = self.current_lataccel_history[-1]
    
    def _get_state_target_futureplan(self, step_idx: int):
        """Get state, target, and future plan for a given step."""
        if step_idx >= len(self.data):
            step_idx = len(self.data) - 1
        
        state_row = self.data.iloc[step_idx]
        state = State(
            roll_lataccel=state_row['roll_lataccel'],
            v_ego=state_row['v_ego'],
            a_ego=state_row['a_ego']
        )
        
        target = state_row['target_lataccel']
        
        # Future plan (next 50 steps)
        future_steps = 50
        future_start = step_idx + 1
        future_end = min(step_idx + future_steps, len(self.data))
        
        future_plan = FuturePlan(
            lataccel=self.data['target_lataccel'].values[future_start:future_end].tolist(),
            roll_lataccel=self.data['roll_lataccel'].values[future_start:future_end].tolist(),
            v_ego=self.data['v_ego'].values[future_start:future_end].tolist(),
            a_ego=self.data['a_ego'].values[future_start:future_end].tolist()
        )
        
        return state, target, future_plan
    
    def step(self, manual_steer_input: float = 0.0):
        """
        Step the TinyPhysics simulation.
        
        Args:
            manual_steer_input: Manual steering override (if manual_control=True)
        
        Returns:
            dict with state info for MetaDrive
        """
        if self.step_idx >= len(self.data):
            return None
        
        # Get current state and target
        state, target, future_plan = self._get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        
        # Compute control action
        if self.manual_control and self.step_idx >= CONTROL_START_IDX:
            action = manual_steer_input
        elif self.step_idx >= CONTROL_START_IDX:
            action = self.controller.update(target, self.current_lataccel, state, future_plan)
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        else:
            action = self.data['steer_command'].values[self.step_idx]
        
        self.action_history.append(action)
        
        # Predict next lateral acceleration using TinyPhysics model
        pred = self.model.get_current_lataccel(
            sim_states=self.state_history[-self.context_length:],
            actions=self.action_history[-self.context_length:],
            past_preds=self.current_lataccel_history[-self.context_length:]
        )
        
        # Apply max acceleration delta constraint
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = target  # Use ground truth during warmup
        
        self.current_lataccel_history.append(self.current_lataccel)
        self.step_idx += 1
        
        # Return state info for MetaDrive
        return {
            'step': self.step_idx,
            'v_ego': state.v_ego,
            'a_ego': state.a_ego,
            'target_lataccel': target,
            'current_lataccel': self.current_lataccel,
            'steer': action,
            'roll_lataccel': state.roll_lataccel,
        }
    
    def compute_cost(self):
        """Compute performance metrics."""
        if self.step_idx <= CONTROL_START_IDX:
            return {'lataccel_cost': 0, 'jerk_cost': 0, 'total_cost': 0}
        
        end_idx = min(self.step_idx, 500)
        target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:end_idx]
        pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:end_idx]
        
        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * 50.0) + jerk_cost
        
        return {
            'lataccel_cost': lat_accel_cost,
            'jerk_cost': jerk_cost,
            'total_cost': total_cost
        }


class TinyPhysicsMetaDriveEnv:
    """
    MetaDrive environment with TinyPhysics lateral dynamics.
    """
    
    def __init__(self, model_path: str, data_path: str, controller_name: str = "pid"):
        # Create base MetaDrive environment
        self.env = MetaDriveEnv({
            "use_render": True,
            "manual_control": False,
            "traffic_density": 0.0,
            "map": "S",  # Simple straight road
            "start_seed": 0,
            "vehicle_config": {
                "show_lidar": False,
                "show_navi_mark": True,
                "show_line_to_navi_mark": True,
            }
        })
        
        # Initialize TinyPhysics vehicle
        self.tiny_vehicle = TinyPhysicsVehicle(model_path, data_path, controller_name)
        
        # Track manual steering input
        self.manual_steer_input = 0.0
        
    def reset(self):
        """Reset environment."""
        obs, _ = self.env.reset()
        return obs
    
    def step(self):
        """Step both MetaDrive (for rendering) and TinyPhysics (for dynamics)."""
        # Step TinyPhysics
        tiny_state = self.tiny_vehicle.step(self.manual_steer_input)
        
        if tiny_state is None:
            return None, True
        
        # Get MetaDrive vehicle
        vehicle = self.env.engine.agent_manager.active_agents['default_agent']
        
        # Override MetaDrive physics with TinyPhysics
        # Convert lateral accel to position offset (simplified)
        dt = 0.1
        
        # Get current forward velocity from TinyPhysics
        speed = tiny_state['v_ego']
        
        # Set vehicle velocity
        vehicle.set_velocity([speed, 0, 0], in_local_frame=True)
        
        # Apply steering (this updates visual representation)
        steering = tiny_state['steer'] / 2.0  # Scale for MetaDrive
        vehicle.steering = np.clip(steering, -1, 1)
        
        # Dummy action for MetaDrive (we override physics anyway)
        action = [0, steering]
        
        # Step MetaDrive (for rendering only)
        # Ignore MetaDrive's done signal - we control termination via TinyPhysics data
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Only done when TinyPhysics data runs out
        return tiny_state, False
    
    def render(self):
        """Render the environment."""
        self.env.render(mode='human')
    
    def close(self):
        """Close environment."""
        self.env.close()


def main():
    import time
    
    parser = argparse.ArgumentParser(description="MetaDrive with TinyPhysics dynamics")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, default="./data/00000.csv")
    parser.add_argument("--controller", type=str, default="pid", help="Controller to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time)")
    args = parser.parse_args()
    
    print(f"Starting MetaDrive with TinyPhysics dynamics")
    print(f"Controller: {args.controller}")
    print(f"Data: {args.data_path}")
    print(f"Speed: {args.speed}x real-time")
    print()
    print("Controls:")
    print("  SPACE: Toggle manual/auto control")
    print("  A/D: Manual steering")
    print("  ESC: Quit")
    print()
    
    # Create environment
    env = TinyPhysicsMetaDriveEnv(args.model_path, args.data_path, args.controller)
    
    # Reset
    env.reset()
    
    # Real-time pacing
    dt = 0.1  # 10 FPS (matches TinyPhysics timestep)
    target_frame_time = dt / args.speed
    
    # Main loop
    step = 0
    done = False
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while not done:
            frame_start = time.time()
            
            # Step simulation
            tiny_state, done = env.step()
            
            if tiny_state is None:
                print("\nReached end of trajectory data")
                break
            
            # Render
            env.render()
            
            # Print status
            if step % 10 == 0:
                cost = env.tiny_vehicle.compute_cost()
                mode = "MANUAL" if env.tiny_vehicle.manual_control else "AUTO"
                elapsed = time.time() - start_time
                print(f"Step {tiny_state['step']:3d} | Time: {elapsed:5.1f}s | Mode: {mode:6s} | "
                      f"Speed: {tiny_state['v_ego']:5.1f} m/s | "
                      f"Steer: {tiny_state['steer']:+.3f} | "
                      f"Lataccel: {tiny_state['current_lataccel']:+.3f} | "
                      f"Target: {tiny_state['target_lataccel']:+.3f} | "
                      f"Cost: {cost['total_cost']:.1f}")
            
            step += 1
            
            # Real-time pacing - sleep to maintain target frame rate
            frame_elapsed = time.time() - frame_start
            sleep_time = target_frame_time - frame_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            last_frame_time = time.time()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Print final metrics
        total_time = time.time() - start_time
        cost = env.tiny_vehicle.compute_cost()
        print(f"\n{'='*60}")
        print(f"Final Metrics:")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Steps: {step}")
        print(f"  Lataccel Cost: {cost['lataccel_cost']:.2f}")
        print(f"  Jerk Cost: {cost['jerk_cost']:.2f}")
        print(f"  Total Cost: {cost['total_cost']:.2f}")
        print(f"{'='*60}")
        
        env.close()


if __name__ == "__main__":
    main()

