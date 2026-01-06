#!/usr/bin/env python3
"""
Collect offline RL dataset from PID controller trajectories.
Runs pid_ff_scheduled_tune on many segments and saves full trajectories.

Usage:
    python3 collect_offline_dataset.py --num_segs 1000 --workers 16 --output datasets/pid_ff_1k.npz
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX


def collect_trajectory(args):
    """
    Collect a single trajectory from PID controller.
    
    Returns:
        dict with arrays: states, actions, rewards, next_states, dones
    """
    data_path, model_path, controller_type = args
    
    try:
        # Load controller
        if controller_type == 'pid_ff':
            from controllers.pid_ff_scheduled_tune import Controller
        else:
            from controllers.pid import Controller
        controller = Controller()
        
        # Load physics model
        physics = TinyPhysicsModel(model_path, debug=False)
        
        # Run simulation
        sim = TinyPhysicsSimulator(physics, str(data_path), controller=controller, debug=False)
        sim.rollout()
        
        # Extract trajectory data (evaluation window only)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for step_idx in range(CONTROL_START_IDX, min(COST_END_IDX, len(sim.data))):
            # Current state: past + current + future (104 dims)
            state = extract_state(sim, step_idx)
            
            # Action taken
            action = sim.action_history[step_idx]
            
            # Reward (negative cost - matching tinyphysics cost function)
            # Cost = lat_accel_cost * 50.0 + jerk_cost
            # where lat_accel_cost = (target - pred)^2 * 100
            # and jerk_cost = (diff(pred) / 0.1)^2 * 100
            if step_idx > CONTROL_START_IDX:
                target = sim.target_lataccel_history[step_idx]
                current = sim.current_lataccel_history[step_idx]
                prev_current = sim.current_lataccel_history[step_idx - 1]
                
                # Lataccel error cost (scaled like tinyphysics)
                lat_error_cost = ((target - current) ** 2) * 100.0
                
                # Jerk cost (scaled like tinyphysics)
                jerk_cost = (((current - prev_current) / 0.1) ** 2) * 100.0
                
                # Total cost (same weighting as tinyphysics)
                step_cost = lat_error_cost * 50.0 + jerk_cost
                
                # Reward is negative cost (normalized)
                reward = -step_cost / 100.0
            else:
                reward = 0.0
            
            # Next state
            if step_idx + 1 < min(COST_END_IDX, len(sim.data)):
                next_state = extract_state(sim, step_idx + 1)
                done = False
            else:
                next_state = state  # Terminal state
                done = True
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            if done:
                break
        
        cost = sim.compute_cost()
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
            'total_cost': cost['total_cost'],
            'segment': str(data_path)
        }
    
    except Exception as e:
        print(f"Error processing {data_path}: {e}")
        return None


def extract_state(sim, step_idx, context_len=10, lookahead_len=10):
    """
    Extract 104-dim state vector from simulator at given step.
    
    Format:
        - past_ctx: (10, 6) = [v_ego, a_ego, roll, target_lat, measured_lat, steer_cmd]
        - current: (4,) = [v_ego, a_ego, roll, measured_lat]
        - future_ctx: (10, 4) = [v_ego, a_ego, roll, target_lat]
    """
    # Past context
    past_ctx = []
    for i in range(max(0, step_idx - context_len), step_idx):
        if i < len(sim.state_history):
            state = sim.state_history[i]
            target = sim.target_lataccel_history[i]
            measured = sim.current_lataccel_history[i]
            steer = sim.action_history[i]
            past_ctx.append([
                state.v_ego, state.a_ego, state.roll_lataccel,
                target, measured, steer
            ])
    
    while len(past_ctx) < context_len:
        past_ctx.insert(0, [0, 0, 0, 0, 0, 0])
    past_ctx = np.array(past_ctx[-context_len:])
    
    # Current state
    if step_idx < len(sim.state_history):
        current_state = sim.state_history[step_idx]
        current_lataccel = sim.current_lataccel_history[step_idx] if step_idx < len(sim.current_lataccel_history) else 0.0
        current = np.array([
            current_state.v_ego,
            current_state.a_ego,
            current_state.roll_lataccel,
            current_lataccel
        ])
    else:
        current = np.array([0, 0, 0, 0])
    
    # Future context
    future_ctx = []
    for i in range(step_idx + 1, min(step_idx + 1 + lookahead_len, len(sim.data))):
        row = sim.data.iloc[i]
        future_ctx.append([
            row['v_ego'],
            row['a_ego'],
            row['roll_lataccel'],
            row['target_lataccel']
        ])
    
    while len(future_ctx) < lookahead_len:
        future_ctx.append([0, 0, 0, 0])
    future_ctx = np.array(future_ctx[:lookahead_len])
    
    # Flatten: (10, 6) + (4,) + (10, 4) = 104
    state_vec = np.concatenate([
        past_ctx.flatten(),
        current,
        future_ctx.flatten()
    ])
    
    return state_vec.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', help='Path to data directory')
    parser.add_argument('--model_path', default='models/tinyphysics.onnx', help='Physics model')
    parser.add_argument('--controller', default='pid_ff', choices=['pid', 'pid_ff'], 
                        help='Controller to use')
    parser.add_argument('--num_segs', type=int, default=1000, help='Number of segments')
    parser.add_argument('--workers', type=int, default=16, help='Parallel workers')
    parser.add_argument('--output', default='datasets/offline_dataset.npz', 
                        help='Output file path')
    args = parser.parse_args()
    
    # Get data files
    data_dir = Path(args.data_path)
    data_files = sorted(data_dir.glob('*.csv'))[:args.num_segs]
    
    print("="*60)
    print("OFFLINE DATASET COLLECTION")
    print("="*60)
    print(f"Controller: {args.controller}")
    print(f"Segments: {len(data_files)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print()
    
    # Prepare args
    args_list = [(f, args.model_path, args.controller) for f in data_files]
    
    # Collect trajectories in parallel
    trajectories = []
    total_transitions = 0
    costs = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_trajectory, arg): arg[0] for arg in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting"):
            result = future.result()
            if result is not None:
                trajectories.append(result)
                total_transitions += len(result['states'])
                costs.append(result['total_cost'])
    
    print(f"\nCollected {len(trajectories)} trajectories")
    print(f"Total transitions: {total_transitions:,}")
    print(f"Avg cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
    print(f"Min cost: {np.min(costs):.2f}, Max cost: {np.max(costs):.2f}")
    
    # Concatenate all trajectories
    all_states = np.concatenate([t['states'] for t in trajectories])
    all_actions = np.concatenate([t['actions'] for t in trajectories])
    all_rewards = np.concatenate([t['rewards'] for t in trajectories])
    all_next_states = np.concatenate([t['next_states'] for t in trajectories])
    all_dones = np.concatenate([t['dones'] for t in trajectories])
    
    print(f"\nDataset shapes:")
    print(f"  States: {all_states.shape}")
    print(f"  Actions: {all_actions.shape}")
    print(f"  Rewards: {all_rewards.shape}")
    print(f"  Next states: {all_next_states.shape}")
    print(f"  Dones: {all_dones.shape}")
    
    # Compute statistics for normalization
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0) + 1e-6
    action_mean = np.mean(all_actions)
    action_std = np.std(all_actions) + 1e-6
    
    print(f"\nStatistics:")
    print(f"  State mean: {np.mean(state_mean):.3f}, std: {np.mean(state_std):.3f}")
    print(f"  Action mean: {action_mean:.3f}, std: {action_std:.3f}")
    print(f"  Reward mean: {np.mean(all_rewards):.3f}, std: {np.std(all_rewards):.3f}")
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        states=all_states,
        actions=all_actions,
        rewards=all_rewards,
        next_states=all_next_states,
        dones=all_dones,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        costs=np.array(costs),
        num_trajectories=len(trajectories),
        num_transitions=total_transitions
    )
    
    # Also save metadata
    metadata = {
        'controller': args.controller,
        'num_segments': len(data_files),
        'num_trajectories': len(trajectories),
        'num_transitions': total_transitions,
        'avg_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)),
        'min_cost': float(np.min(costs)),
        'max_cost': float(np.max(costs)),
        'segments': [t['segment'] for t in trajectories]
    }
    
    with open(output_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Metadata saved to {output_path.with_suffix('.pkl')}")
    print(f"Total size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nReady for offline RL training!")


if __name__ == '__main__':
    main()

