#!/usr/bin/env python3
"""
MetaDrive with TinyPhysics dynamics and trajectory-based road generation.

Generates a custom road path from CSV trajectory data.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time

from metadrive.envs.metadrive_env import MetaDriveEnv
import importlib

from tinyphysics import (
    TinyPhysicsModel, State, FuturePlan,
    CONTROL_START_IDX, STEER_RANGE, MAX_ACC_DELTA, DEL_T
)


class TinyPhysicsVehicle:
    """Vehicle using TinyPhysics for lateral dynamics."""
    
    def __init__(self, model_path: str, data_path: str, controller_name: str):
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data = self._load_data(data_path)
        
        controller_class = importlib.import_module(f'controllers.{controller_name}').Controller
        self.controller = controller_class()
        self.controller_name = controller_name
        
        self.step_idx = 0
        self.context_length = 20
        self._init_histories()
        
        # Cumulative position tracking
        self.cumulative_x = 0.0
        self.cumulative_y = 0.0
        
    def _load_data(self, data_path: str):
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
        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.target_lataccel_history = []
        
        for i in range(self.context_length):
            state, target, _ = self._get_state_target_futureplan(i)
            self.state_history.append(state)
            self.action_history.append(self.data['steer_command'].values[i])
            self.current_lataccel_history.append(target)
            self.target_lataccel_history.append(target)
        
        self.step_idx = self.context_length
        self.current_lataccel = self.current_lataccel_history[-1]
    
    def _get_state_target_futureplan(self, step_idx: int):
        if step_idx >= len(self.data):
            step_idx = len(self.data) - 1
        
        state_row = self.data.iloc[step_idx]
        state = State(
            roll_lataccel=state_row['roll_lataccel'],
            v_ego=state_row['v_ego'],
            a_ego=state_row['a_ego']
        )
        
        target = state_row['target_lataccel']
        
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
        if self.step_idx >= len(self.data):
            return None
        
        state, target, future_plan = self._get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        
        # Compute control
        if self.step_idx >= CONTROL_START_IDX:
            action = self.controller.update(target, self.current_lataccel, state, future_plan)
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        else:
            action = self.data['steer_command'].values[self.step_idx]
        
        self.action_history.append(action)
        
        # Predict with TinyPhysics
        pred = self.model.get_current_lataccel(
            sim_states=self.state_history[-self.context_length:],
            actions=self.action_history[-self.context_length:],
            past_preds=self.current_lataccel_history[-self.context_length:]
        )
        
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = target
        
        self.current_lataccel_history.append(self.current_lataccel)
        
        # Update cumulative position
        # Simplified: integrate velocity and lateral accel
        dt = 0.1
        self.cumulative_x += state.v_ego * dt
        # Lateral displacement from lateral accel
        self.cumulative_y += 0.5 * self.current_lataccel * dt * dt
        
        self.step_idx += 1
        
        return {
            'step': self.step_idx,
            'v_ego': state.v_ego,
            'a_ego': state.a_ego,
            'target_lataccel': target,
            'current_lataccel': self.current_lataccel,
            'steer': action,
            'roll_lataccel': state.roll_lataccel,
            'position_x': self.cumulative_x,
            'position_y': self.cumulative_y,
        }
    
    def compute_cost(self):
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


def main():
    parser = argparse.ArgumentParser(description="MetaDrive with TinyPhysics and trajectory-based road")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, default="./data/00000.csv")
    parser.add_argument("--controller", type=str, default="pid")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    args = parser.parse_args()
    
    print(f"Starting MetaDrive with TinyPhysics")
    print(f"Controller: {args.controller}")
    print(f"Data: {args.data_path}")
    print(f"Speed: {args.speed}x real-time\n")
    
    # Load trajectory for map generation
    df = pd.read_csv(args.data_path)
    
    # Create environment with very long straight road
    # Using block sequence for very long straight path
    env = MetaDriveEnv({
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.0,
        "map": "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",  # 30+ straight segments = very long road
        "start_seed": 1000,
        "vehicle_config": {
            "show_lidar": False,
            "show_navi_mark": False,
            "show_line_to_navi_mark": False,
        },
        "horizon": 10000,  # Very long episode
        "decision_repeat": 1,  # Update every frame
    })
    
    # Initialize TinyPhysics vehicle
    tiny_vehicle = TinyPhysicsVehicle(args.model_path, args.data_path, args.controller)
    
    # Reset environment
    env.reset()
    vehicle = env.engine.agent_manager.active_agents['default_agent']
    
    # Real-time pacing
    dt = 0.1
    target_frame_time = dt / args.speed
    start_time = time.time()
    step = 0
    
    try:
        while step < len(df) - 20:
            frame_start = time.time()
            
            # Step TinyPhysics
            tiny_state = tiny_vehicle.step()
            
            if tiny_state is None:
                break
            
            # Update MetaDrive vehicle position
            # Keep vehicle within reasonable bounds (loop position to stay on visible road)
            # This prevents the car from driving off into infinity
            pos_x = tiny_state['position_x'] % 2000  # Loop every 2km
            pos_y = tiny_state['position_y']
            
            vehicle.set_position([pos_x, pos_y, 0.3])
            vehicle.set_velocity([tiny_state['v_ego'], 0, 0], in_local_frame=True)
            vehicle.steering = np.clip(tiny_state['steer'] / 2.0, -1, 1)
            
            # Force physics and rendering update
            env.engine.taskMgr.step()  # Process one frame
            
            # Render - this should update the display
            env.render(mode='human', text={
                'Step': tiny_state['step'],
                'Speed': f"{tiny_state['v_ego']:.1f} m/s",
                'Steer': f"{tiny_state['steer']:+.3f}",
                'Lataccel': f"{tiny_state['current_lataccel']:+.3f}",
                'Target': f"{tiny_state['target_lataccel']:+.3f}",
            })
            
            # Print status
            if step % 10 == 0:
                cost = tiny_vehicle.compute_cost()
                elapsed = time.time() - start_time
                print(f"Step {tiny_state['step']:3d} | Time: {elapsed:5.1f}s | "
                      f"Speed: {tiny_state['v_ego']:5.1f} m/s | "
                      f"Steer: {tiny_state['steer']:+.3f} | "
                      f"Lataccel: {tiny_state['current_lataccel']:+.3f} | "
                      f"Target: {tiny_state['target_lataccel']:+.3f} | "
                      f"Cost: {cost['total_cost']:.1f}")
            
            step += 1
            
            # Real-time pacing
            frame_elapsed = time.time() - frame_start
            sleep_time = target_frame_time - frame_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        total_time = time.time() - start_time
        cost = tiny_vehicle.compute_cost()
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

