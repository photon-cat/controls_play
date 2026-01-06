"""
Train neural steering model using PID as teacher (imitation learning).

Phase 1: Run PID on simulator, collect (state, action) trajectories
Phase 2: Train neural net to imitate PID (behavioral cloning)
Phase 3: Fine-tune with RL to improve beyond PID

This approach works better than training on ground truth because:
- PID actually works on the simulator (~50-130 cost)
- Neural net learns to handle real measurement errors
- RL can then fine-tune from a good starting point
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import signal
import atexit
from datetime import datetime

# Import model architecture from existing script
from train_steer_model_rl import SteerModelRL, CONTEXT_LENGTH, LOOKAHEAD_LENGTH, STEER_RANGE

ACC_G = 9.81


def collect_pid_trajectory(data_path, physics_model_path):
    """
    Run PID controller on simulator and collect (state, action) pairs.
    Returns list of (past_ctx, current, future_ctx, action) tuples.
    """
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from controllers.pid import Controller as PIDController
    
    physics = TinyPhysicsModel(physics_model_path, debug=False)
    pid = PIDController()
    
    # Collector wrapper
    class CollectorWrapper:
        def __init__(self, controller):
            self.controller = controller
            self.trajectory = []
            self.history = []  # (v_ego, a_ego, roll_lataccel, target, measured, steer)
        
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            # Get PID action
            action = self.controller.update(target_lataccel, current_lataccel, state, future_plan)
            
            # Store observation
            roll_lataccel = state.roll_lataccel
            obs = [state.v_ego, state.a_ego, roll_lataccel, target_lataccel, current_lataccel, action]
            self.history.append(obs)
            
            # Build training sample if we have enough context
            if len(self.history) >= CONTEXT_LENGTH and future_plan and len(future_plan.lataccel) >= LOOKAHEAD_LENGTH:
                # Past context (last 10 steps)
                past_ctx = np.array(self.history[-CONTEXT_LENGTH:], dtype=np.float32)
                
                # Current
                current = np.array([state.v_ego, state.a_ego, roll_lataccel, current_lataccel], dtype=np.float32)
                
                # Future lookahead
                future_ctx = np.stack([
                    np.array(future_plan.v_ego[:LOOKAHEAD_LENGTH]),
                    np.array(future_plan.a_ego[:LOOKAHEAD_LENGTH]),
                    np.array(future_plan.roll_lataccel[:LOOKAHEAD_LENGTH]),
                    np.array(future_plan.lataccel[:LOOKAHEAD_LENGTH]),
                ], axis=1).astype(np.float32)
                
                self.trajectory.append((past_ctx, current, future_ctx, action))
            
            return action
    
    wrapper = CollectorWrapper(pid)
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    try:
        sim.rollout()
        cost = sim.compute_cost()
        return wrapper.trajectory, cost['total_cost']
    except Exception as e:
        return [], float('inf')


def _collect_worker(args):
    """Worker for parallel trajectory collection"""
    data_path, physics_model_path = args
    try:
        trajectory, cost = collect_pid_trajectory(data_path, physics_model_path)
        return {
            'path': str(data_path),
            'trajectory': trajectory,
            'cost': cost,
            'success': True
        }
    except Exception as e:
        return {
            'path': str(data_path),
            'trajectory': [],
            'cost': float('inf'),
            'success': False,
            'error': str(e)
        }


def collect_all_trajectories(data_files, physics_model_path, n_workers=8):
    """Collect trajectories from all data files in parallel"""
    all_trajectories = []
    costs = []
    
    worker_args = [(str(f), physics_model_path) for f in data_files]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_collect_worker, args) for args in worker_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting PID trajectories"):
            result = future.result()
            if result['success'] and len(result['trajectory']) > 0:
                all_trajectories.extend(result['trajectory'])
                costs.append(result['cost'])
    
    return all_trajectories, costs


class TeacherDataset(torch.utils.data.Dataset):
    """Dataset for imitation learning from PID teacher"""
    def __init__(self, trajectories):
        self.trajectories = trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        past_ctx, current, future_ctx, action = self.trajectories[idx]
        return (
            torch.tensor(past_ctx, dtype=torch.float32),
            torch.tensor(current, dtype=torch.float32),
            torch.tensor(future_ctx, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
        )


def train_imitation(model, train_loader, val_loader, epochs, lr, device, ckpt_dir):
    """Phase 2: Train to imitate PID"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    training_log = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for past, cur, fut, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            past, cur, fut, tgt = past.to(device), cur.to(device), fut.to(device), tgt.to(device)
            
            mean, _, _ = model(past, cur, fut, deterministic=True)
            loss = F.mse_loss(mean, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past, cur, fut, tgt in val_loader:
                past, cur, fut, tgt = past.to(device), cur.to(device), fut.to(device), tgt.to(device)
                mean, _, _ = model(past, cur, fut, deterministic=True)
                val_loss += F.mse_loss(mean, tgt).item() * past.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }, ckpt_dir / f"imitation_epoch_{epoch+1:03d}.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, ckpt_dir / "best_imitation.pt")
            print("  → Saved best model")
    
    return model, training_log


def evaluate_on_sim(model, data_files, physics_model_path, device, n_episodes=20):
    """Evaluate model on simulator"""
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    from train_steer_model_rl import RLController
    
    model.eval()
    controller = RLController(model, device)
    
    class Wrapper:
        def __init__(self, ctrl):
            self.ctrl = ctrl
        def update(self, target, current, state, future_plan):
            action, _ = self.ctrl.get_action(state, target, current, future_plan, deterministic=True)
            return action
    
    costs = []
    for i, data_file in enumerate(data_files[:n_episodes]):
        controller.reset()
        physics = TinyPhysicsModel(physics_model_path, debug=False)
        wrapper = Wrapper(controller)
        sim = TinyPhysicsSimulator(physics, str(data_file), controller=wrapper, debug=False)
        
        try:
            sim.rollout()
            cost = sim.compute_cost()['total_cost']
            costs.append(cost)
        except:
            costs.append(float('inf'))
    
    return costs


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup checkpoint directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(args.output).parent / f"teacher_ckpts_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")
    
    # Get data files
    data_files = sorted(Path(args.data_path).glob("*.csv"))
    if args.max_files:
        data_files = data_files[:args.max_files]
    print(f"Using {len(data_files)} data files")
    
    # ========================================
    # Phase 1: Collect PID trajectories
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1: Collecting PID trajectories")
    print("="*60)
    
    trajectories, pid_costs = collect_all_trajectories(
        data_files[:args.collect_segments], 
        args.physics_model, 
        n_workers=args.n_workers
    )
    
    print(f"\nCollected {len(trajectories)} training samples")
    print(f"PID costs: mean={np.mean(pid_costs):.1f}, min={np.min(pid_costs):.1f}, max={np.max(pid_costs):.1f}")
    
    # Split data
    np.random.shuffle(trajectories)
    n_val = int(len(trajectories) * 0.1)
    train_traj = trajectories[:-n_val]
    val_traj = trajectories[-n_val:]
    
    train_ds = TeacherDataset(train_traj)
    val_ds = TeacherDataset(val_traj)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_traj)}, Val samples: {len(val_traj)}")
    
    # ========================================
    # Phase 2: Train to imitate PID
    # ========================================
    print("\n" + "="*60)
    print("PHASE 2: Training imitation model")
    print("="*60)
    
    model = SteerModelRL(d_model=args.hidden).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    model, imitation_log = train_imitation(
        model, train_loader, val_loader, 
        args.imitation_epochs, args.lr, device, ckpt_dir
    )
    
    # Evaluate imitation model
    print("\nEvaluating imitation model on simulator...")
    eval_costs = evaluate_on_sim(model, data_files, args.physics_model, device, n_episodes=20)
    print(f"Imitation model costs: mean={np.mean(eval_costs):.1f}, min={np.min(eval_costs):.1f}, max={np.max(eval_costs):.1f}")
    
    # ========================================
    # Phase 3: RL fine-tuning (optional)
    # ========================================
    if args.rl_epochs > 0:
        print("\n" + "="*60)
        print("PHASE 3: RL fine-tuning")
        print("="*60)
        
        from train_steer_model_rl import rl_rollout, RLController
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rl_lr, weight_decay=1e-4)
        cost_baseline = np.mean(eval_costs)
        
        for epoch in range(args.rl_epochs):
            print(f"\nRL Epoch {epoch+1}/{args.rl_epochs}")
            
            # Run rollouts and collect trajectories
            episode_files = list(np.random.choice(data_files, size=args.rl_episodes, replace=False))
            all_trajectories = []
            rl_costs = []
            
            model.eval()
            for data_file in tqdm(episode_files, desc="RL Episodes"):
                try:
                    result = rl_rollout(
                        model, data_file, args.physics_model, device,
                        deterministic=False, collect_trajectory=True
                    )
                    cost, _, _, _, trajectory = result
                    rl_costs.append(cost['total_cost'])
                    all_trajectories.append((trajectory, cost['total_cost']))
                    
                    # Update baseline
                    cost_baseline = 0.95 * cost_baseline + 0.05 * cost['total_cost']
                except Exception as e:
                    print(f"Rollout failed: {e}")
            
            # REINFORCE update
            if all_trajectories:
                model.train()
                pg_loss = 0.0
                n_samples = 0
                
                for trajectory, cost in all_trajectories:
                    advantage = (cost_baseline - cost) / max(cost_baseline, 100.0)
                    
                    if abs(advantage) < 0.01 or len(trajectory) == 0:
                        continue
                    
                    # Sample from trajectory
                    max_steps = min(50, len(trajectory))
                    indices = np.random.choice(len(trajectory), max_steps, replace=False)
                    
                    for idx in indices:
                        past_np, cur_np, fut_np, action_taken = trajectory[idx]
                        
                        past = torch.tensor(past_np, dtype=torch.float32).unsqueeze(0).to(device)
                        cur = torch.tensor(cur_np, dtype=torch.float32).unsqueeze(0).to(device)
                        fut = torch.tensor(fut_np, dtype=torch.float32).unsqueeze(0).to(device)
                        action = torch.tensor([[action_taken]], dtype=torch.float32).to(device)
                        
                        mean, std, _ = model(past, cur, fut, deterministic=True)
                        log_std = torch.log(std + 1e-8)
                        log_prob = -0.5 * ((action - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
                        
                        pg_loss += -advantage * log_prob.mean()
                        n_samples += 1
                
                if n_samples > 0:
                    pg_loss = pg_loss / n_samples * 0.01  # Scale down
                    
                    if abs(pg_loss.item()) < 1.0:
                        optimizer.zero_grad()
                        pg_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()
                        print(f"  PG update: loss={pg_loss.item():.6f}, samples={n_samples}")
            
            avg_cost = np.mean(rl_costs) if rl_costs else float('inf')
            print(f"  rl_cost={avg_cost:.1f}, baseline={cost_baseline:.1f}")
            
            # Save checkpoint
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'rl_cost': avg_cost,
            }, ckpt_dir / f"rl_epoch_{epoch+1:03d}.pt")
    
    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'model_type': 'teacher_rl',
        'hidden': args.hidden,
    }, args.output)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final model saved: {args.output}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Train steering model with PID teacher")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--physics_model", type=str, default="models/tinyphysics.onnx")
    parser.add_argument("--output", type=str, default="models/steer_model_teacher.pt")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    
    # Phase 1: Collection
    parser.add_argument("--collect_segments", type=int, default=500, 
                        help="Number of segments to collect PID trajectories from")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Parallel workers for trajectory collection")
    
    # Phase 2: Imitation
    parser.add_argument("--imitation_epochs", type=int, default=10,
                        help="Epochs for imitation learning")
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Phase 3: RL (optional)
    parser.add_argument("--rl_epochs", type=int, default=0,
                        help="Epochs for RL fine-tuning (0 to skip)")
    parser.add_argument("--rl_episodes", type=int, default=8,
                        help="Episodes per RL epoch")
    parser.add_argument("--rl_lr", type=float, default=1e-4)
    
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()
    
    main(args)

