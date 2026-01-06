#!/usr/bin/env python3
"""
Gym-style environment wrapper for tinyphysics steering control.
Adapts the simulator to work with standard RL algorithms like PPO.
"""

import numpy as np
import torch
from pathlib import Path
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX
from controllers.pid import Controller as PIDController


class SteeringEnv:
    """
    Gym-like environment for steering control.
    
    Observation: (past_ctx, current, future_ctx) flattened = 104 dims
        - past_ctx: (10, 6) = [v_ego, a_ego, roll, target_lat, measured_lat, steer_cmd]
        - current: (4,) = [v_ego, a_ego, roll, measured_lat]
        - future_ctx: (10, 4) = [v_ego, a_ego, roll, target_lat]
    
    Action: steer_command in [-2, 2]
    
    Reward: -(lat_accel_cost * 50 + jerk_cost) per step
    """
    
    def __init__(self, data_path, model_path, context_len=10, lookahead_len=10):
        self.data_path = data_path
        self.model_path = model_path
        self.context_len = context_len
        self.lookahead_len = lookahead_len
        
        # Observation and action space dimensions
        self.obs_dim = context_len * 6 + 4 + lookahead_len * 4  # 104
        self.action_dim = 1
        self.action_low = -2.0
        self.action_high = 2.0
        
        # Initialize simulator
        self.physics_model = None
        self.sim = None
        self.step_count = 0
        self.prev_lataccel = 0.0
        
    def reset(self):
        """Reset environment to start of trajectory"""
        # Reinitialize physics model and simulator
        self.physics_model = TinyPhysicsModel(self.model_path, debug=False)
        
        # Dummy controller (we'll override its actions)
        class DummyController:
            def update(self, target, current, state, future_plan):
                return 0.0
        
        self.sim = TinyPhysicsSimulator(
            self.physics_model,
            str(self.data_path),
            controller=DummyController(),
            debug=False
        )
        
        self.step_count = 0
        self.prev_lataccel = 0.0
        
        # Return initial observation
        return self._get_obs()
    
    def _get_obs(self):
        """
        Extract observation from simulator state.
        Returns flattened (104,) array.
        """
        # Get history from simulator
        idx = self.sim.step_idx
        
        # Past context (10 steps): [v_ego, a_ego, roll, target_lat, measured_lat, steer_cmd]
        past_ctx = []
        for i in range(max(0, idx - self.context_len), idx):
            if i < len(self.sim.state_history):
                state = self.sim.state_history[i]
                target = self.sim.target_lataccel_history[i]
                measured = self.sim.current_lataccel_history[i]
                steer = self.sim.action_history[i]
                past_ctx.append([
                    state.v_ego, state.a_ego, state.roll_lataccel,
                    target, measured, steer
                ])
        
        # Pad if needed
        while len(past_ctx) < self.context_len:
            past_ctx.insert(0, [0, 0, 0, 0, 0, 0])
        past_ctx = np.array(past_ctx[-self.context_len:])
        
        # Current state: [v_ego, a_ego, roll, measured_lat]
        if idx < len(self.sim.state_history):
            current_state = self.sim.state_history[idx]
            current = np.array([
                current_state.v_ego,
                current_state.a_ego,
                current_state.roll_lataccel,
                self.sim.current_lataccel
            ])
        else:
            current = np.array([0, 0, 0, 0])
        
        # Future context (10 steps): [v_ego, a_ego, roll, target_lat]
        future_ctx = []
        for i in range(idx + 1, min(idx + 1 + self.lookahead_len, len(self.sim.data))):
            row = self.sim.data.iloc[i]
            future_ctx.append([
                row['v_ego'],
                row['a_ego'],
                row['roll_lataccel'],
                row['target_lataccel']
            ])
        
        # Pad if needed
        while len(future_ctx) < self.lookahead_len:
            future_ctx.append([0, 0, 0, 0])
        future_ctx = np.array(future_ctx[:self.lookahead_len])
        
        # Flatten: (10, 6) + (4,) + (10, 4) = 104
        obs = np.concatenate([
            past_ctx.flatten(),
            current,
            future_ctx.flatten()
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: float in [-2, 2]
        
        Returns:
            obs, reward, done, info
        """
        # Clip action
        action = np.clip(action, self.action_low, self.action_high)
        
        # Store previous action for smoothness penalty
        prev_action = self.sim.action_history[-1] if len(self.sim.action_history) > 0 else 0.0
        
        # Store action in simulator (override controller)
        self.sim.action_history.append(action)
        
        # Advance physics simulation
        self.sim.sim_step(self.sim.step_idx)
        
        # Get new state
        self.sim.step_idx += 1
        if self.sim.step_idx < len(self.sim.data):
            state, target, futureplan = self.sim.get_state_target_futureplan(self.sim.step_idx)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            self.sim.futureplan = futureplan
        
        # Compute reward (negative cost)
        current_lataccel = self.sim.current_lataccel
        target_lataccel = self.sim.target_lataccel_history[self.sim.step_idx - 1]
        
        # Tracking error
        lat_error = (target_lataccel - current_lataccel) ** 2
        
        # Jerk (change in lateral accel)
        jerk = ((current_lataccel - self.prev_lataccel) / 0.1) ** 2
        
        # Action smoothness (penalize rapid steering changes)
        action_change = (action - prev_action) ** 2
        
        # Combined reward (normalized scale)
        step_cost = lat_error * 50.0 + jerk + action_change * 2.0
        reward = -step_cost / 50.0  # Normalize to ~[-1, 0] range
        
        self.prev_lataccel = current_lataccel
        self.step_count += 1
        
        # Done when episode ends (after evaluation window)
        done = self.sim.step_idx >= min(COST_END_IDX, len(self.sim.data) - 1)
        
        # Info
        info = {
            'lat_error': lat_error,
            'jerk': jerk,
            'step': self.step_count
        }
        
        # Get next observation
        if not done:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        return obs, reward, done, info
    
    def get_final_cost(self):
        """Compute final cost using simulator's method"""
        if self.sim is not None:
            return self.sim.compute_cost()
        return {'lataccel_cost': 0, 'jerk_cost': 0, 'total_cost': 0}


def collect_pid_demonstrations(data_files, model_path, n_demos=10):
    """
    Collect PID demonstrations for behavior cloning warm-start.
    
    Returns:
        states: (N, 104) observations
        actions: (N,) PID actions
    """
    print(f"\nCollecting {n_demos} PID demonstrations...")
    
    all_states = []
    all_actions = []
    
    for data_idx, data_path in enumerate(data_files[:n_demos]):
        # Create environment
        env = SteeringEnv(data_path, model_path)
        
        # Run PID controller on this environment
        physics = TinyPhysicsModel(model_path, debug=False)
        pid = PIDController()
        sim = TinyPhysicsSimulator(physics, str(data_path), controller=pid, debug=False)
        
        # Run full simulation
        for step_idx in range(CONTROL_START_IDX, min(COST_END_IDX, len(sim.data))):
            # Get observation (create temp env to extract state)
            temp_env = SteeringEnv(data_path, model_path)
            temp_env.sim = sim
            obs = temp_env._get_obs()
            
            # Record PID's action for this state
            action = sim.action_history[step_idx] if step_idx < len(sim.action_history) else 0.0
            
            all_states.append(obs)
            all_actions.append(action)
            
            # Step simulator
            sim.step()
            
            if sim.step_idx >= min(COST_END_IDX, len(sim.data)):
                break
        
        cost = sim.compute_cost()
        print(f"  Demo {data_idx+1:2d}: cost={cost['total_cost']:.1f}, transitions={len(all_states)}")
    
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    
    print(f"Collected {len(states)} transitions from {n_demos} demos")
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")
    
    return states, actions


def behavior_clone_init(policy, states, actions, epochs=10, batch_size=256, lr=1e-3):
    """
    Initialize policy with behavior cloning from PID demonstrations.
    """
    print(f"\nBehavior cloning initialization ({epochs} epochs)...")
    
    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(states))
        total_loss = 0
        n_batches = 0
        
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            
            # Get policy prediction
            pred_actions, _, _ = policy(batch_states, deterministic=True)
            
            # MSE loss (fix shape warning)
            loss = torch.nn.functional.mse_loss(pred_actions.squeeze(-1), batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    print("Behavior cloning complete!")


if __name__ == "__main__":
    # Quick test
    data_path = Path("data/00000.csv")
    model_path = "models/tinyphysics.onnx"
    
    print("Testing SteeringEnv wrapper...")
    env = SteeringEnv(data_path, model_path)
    
    obs = env.reset()
    print(f"Obs shape: {obs.shape} (expected 104)")
    
    # Take a few steps
    for i in range(5):
        action = 0.1  # Small steering
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, done={done}")
        if done:
            break
    
    final_cost = env.get_final_cost()
    print(f"\nFinal cost: {final_cost}")

