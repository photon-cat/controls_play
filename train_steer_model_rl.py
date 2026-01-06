"""
RL-based training for steer model.

Architecture:
- Past context (t-10 to t-1): [vEgo, aEgo, roll, targetLat, measuredLat, steerCmd] × 10
- Current (t0): [vEgo, aEgo, roll, measuredLat]
- Future lookahead (t+1 to t+10): [vEgo, aEgo, roll, targetLat] × 10
- Output: steerCommand

Two-phase training:
1. Supervised: Train on CSV data, targetLat = measuredLat
2. RL: Run on simulator, optimize actual tracking + jerk cost
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import signal
import json
import atexit
from datetime import datetime

# Constants
ACC_G = 9.81
CONTEXT_LENGTH = 10
LOOKAHEAD_LENGTH = 10
STEER_RANGE = [-2, 2]
CONTROL_START_IDX = 100

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])


class SteerModelRL(nn.Module):
    """
    Transformer model with past context + future lookahead.
    
    Inputs:
        past_ctx: (batch, 10, 6) - past [vEgo, aEgo, roll, targetLat, measuredLat, steerCmd]
        current: (batch, 4) - current [vEgo, aEgo, roll, measuredLat]
        future_ctx: (batch, 10, 4) - future [vEgo, aEgo, roll, targetLat]
    
    Output:
        steer_mean: (batch,) - mean of steer distribution
        steer_std: (batch,) - std of steer distribution (for RL exploration)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        # Input projections
        self.past_proj = nn.Linear(6, d_model)
        self.current_proj = nn.Linear(4, d_model)
        self.future_proj = nn.Linear(4, d_model)
        
        # Position embeddings: 10 past + 1 current + 10 future = 21 positions
        self.pos_emb = nn.Parameter(torch.randn(21, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Learnable baseline for variance reduction
        self.baseline = nn.Parameter(torch.tensor(0.0))
        
        # Input normalization
        self.register_buffer('past_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('past_std', torch.tensor([15., 1., 0.5, 1., 1., 0.5]))
        self.register_buffer('current_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('current_std', torch.tensor([15., 1., 0.5, 1.]))
        self.register_buffer('future_mean', torch.tensor([20., 0., 0., 0.]))
        self.register_buffer('future_std', torch.tensor([15., 1., 0.5, 1.]))
    
    def forward(self, past_ctx, current, future_ctx, deterministic=False):
        # Normalize inputs
        past_ctx = (past_ctx - self.past_mean) / self.past_std
        current = (current - self.current_mean) / self.current_std
        future_ctx = (future_ctx - self.future_mean) / self.future_std
        
        # Project to d_model
        past_emb = self.past_proj(past_ctx)  # (batch, 10, d_model)
        current_emb = self.current_proj(current).unsqueeze(1)  # (batch, 1, d_model)
        future_emb = self.future_proj(future_ctx)  # (batch, 10, d_model)
        
        # Concatenate: [past..., current, future...]
        seq = torch.cat([past_emb, current_emb, future_emb], dim=1)  # (batch, 21, d_model)
        seq = seq + self.pos_emb
        
        # Transformer
        out = self.transformer(seq)
        
        # Use current position (index 10) for output
        current_out = out[:, 10]
        
        # Get mean and std
        mean = self.mean_head(current_out).squeeze(-1)
        log_std = self.log_std_head(current_out).squeeze(-1)
        # Std for exploration - larger for better exploration
        # exp(-2)=0.135, exp(-0.5)=0.606
        std = torch.exp(log_std.clamp(-2, -0.5))
        
        if deterministic:
            return mean, std, None
        else:
            # Sample action
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, std, log_prob
    
    def get_log_prob(self, past_ctx, current, future_ctx, action):
        """Get log probability of action (for policy gradient)"""
        mean, std, _ = self.forward(past_ctx, current, future_ctx, deterministic=True)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action)


def load_supervised_data(data_dir: Path, max_files: int = None):
    """Load data for supervised pre-training"""
    files = sorted(data_dir.glob("*.csv"))
    if max_files:
        files = files[:max_files]
    
    samples = []
    print(f"Loading {len(files)} files for supervised training...")
    
    for f in tqdm(files):
        df = pd.read_csv(f)
        
        # Only use rows with valid steerCommand
        valid_mask = ~df['steerCommand'].isna()
        df = df[valid_mask].reset_index(drop=True)
        
        if len(df) < CONTEXT_LENGTH + LOOKAHEAD_LENGTH + 1:
            continue
        
        # Extract features
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        roll_lataccel = np.sin(df['roll'].values) * ACC_G
        target_lataccel = df['targetLateralAcceleration'].values
        steer_command = -df['steerCommand'].values
        # For supervised: measured = target + noise (to simulate real errors)
        measured_lataccel = target_lataccel + np.random.randn(len(target_lataccel)) * 0.3
        
        # Build samples with past and future context
        for i in range(CONTEXT_LENGTH, len(df) - LOOKAHEAD_LENGTH):
            # Past context (t-10 to t-1)
            past_slice = slice(i - CONTEXT_LENGTH, i)
            past_ctx = np.stack([
                v_ego[past_slice],
                a_ego[past_slice],
                roll_lataccel[past_slice],
                target_lataccel[past_slice],
                measured_lataccel[past_slice],
                steer_command[past_slice],
            ], axis=1)  # (10, 6)
            
            # Current (t0)
            current = np.array([
                v_ego[i],
                a_ego[i],
                roll_lataccel[i],
                measured_lataccel[i],
            ])  # (4,)
            
            # Future lookahead (t+1 to t+10)
            future_slice = slice(i + 1, i + 1 + LOOKAHEAD_LENGTH)
            future_ctx = np.stack([
                v_ego[future_slice],
                a_ego[future_slice],
                roll_lataccel[future_slice],
                target_lataccel[future_slice],
            ], axis=1)  # (10, 4)
            
            # Target
            target_steer = steer_command[i]
            
            samples.append((past_ctx, current, future_ctx, target_steer))
    
    return samples


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        past, cur, fut, tgt = self.samples[idx]
        return (
            torch.tensor(past, dtype=torch.float32),
            torch.tensor(cur, dtype=torch.float32),
            torch.tensor(fut, dtype=torch.float32),
            torch.tensor(tgt, dtype=torch.float32),
        )


def supervised_train(model, train_loader, val_loader, epochs, lr, device):
    """Phase 1: Supervised pre-training"""
    print("\n" + "="*50)
    print("Phase 1: Supervised Pre-training")
    print("="*50)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for past, cur, fut, tgt in tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}/{epochs}"):
            past, cur, fut, tgt = past.to(device), cur.to(device), fut.to(device), tgt.to(device)
            
            # Forward (deterministic for supervised)
            mean, _, _ = model(past, cur, fut, deterministic=True)
            
            # MSE loss
            loss = F.mse_loss(mean, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * past.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/steer_model_rl_supervised.pt')
            print(f"  → Saved supervised checkpoint")
    
    return model


class RLController:
    """Controller wrapper for RL rollouts"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.past_buffer = []
        self.prev_steer = 0.0
        self.last_inputs = None  # For trajectory collection
    
    def reset(self):
        self.past_buffer = []
        self.prev_steer = 0.0
        self.last_inputs = None
    
    def get_action(self, state, target_lataccel, current_lataccel, future_plan, deterministic=False):
        """Get action from model"""
        v_ego, a_ego, roll_lataccel = state.v_ego, state.a_ego, state.roll_lataccel
        
        # Build past observation
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, self.prev_steer]
        self.past_buffer.append(obs)
        if len(self.past_buffer) > CONTEXT_LENGTH:
            self.past_buffer = self.past_buffer[-CONTEXT_LENGTH:]
        
        # Not enough context
        if len(self.past_buffer) < CONTEXT_LENGTH:
            steer = 0.3 * (target_lataccel - current_lataccel)
            self.prev_steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
            self.last_inputs = None
            return self.prev_steer, None
        
        # Build future context
        if future_plan and len(future_plan.lataccel) >= LOOKAHEAD_LENGTH:
            future_ctx = np.stack([
                future_plan.v_ego[:LOOKAHEAD_LENGTH],
                future_plan.a_ego[:LOOKAHEAD_LENGTH],
                future_plan.roll_lataccel[:LOOKAHEAD_LENGTH],
                future_plan.lataccel[:LOOKAHEAD_LENGTH],
            ], axis=1)
        else:
            # Fallback: repeat current state
            future_ctx = np.tile([v_ego, a_ego, roll_lataccel, target_lataccel], (LOOKAHEAD_LENGTH, 1))
        
        # Store numpy arrays for trajectory collection (before tensor conversion)
        past_np = np.array(self.past_buffer, dtype=np.float32)
        cur_np = np.array([v_ego, a_ego, roll_lataccel, current_lataccel], dtype=np.float32)
        fut_np = np.array(future_ctx, dtype=np.float32)
        self.last_inputs = (past_np.copy(), cur_np.copy(), fut_np.copy())
        
        # To tensors (no grad during rollout - we'll recompute for REINFORCE)
        with torch.no_grad():
            past = torch.tensor(past_np).unsqueeze(0).to(self.device)
            cur = torch.tensor(cur_np).unsqueeze(0).to(self.device)
            fut = torch.tensor(fut_np).unsqueeze(0).to(self.device)
            
            # Get action
            if deterministic:
                mean, _, _ = self.model(past, cur, fut, deterministic=True)
                steer = mean.item()
                log_prob = None
            else:
                steer, _, log_prob = self.model(past, cur, fut, deterministic=False)
                steer = steer.item()
                log_prob = log_prob.item() if log_prob is not None else None
        
        steer = np.clip(steer, STEER_RANGE[0], STEER_RANGE[1])
        self.prev_steer = steer
        return steer, log_prob


def rl_rollout(model, data_path, physics_model_path, device, deterministic=False, collect_trajectory=False):
    """
    Run one episode on simulator.
    Returns: cost, log_probs, lataccel trajectory, (optional) trajectory with per-step costs
    """
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH as SIM_CONTEXT, DEL_T, LAT_ACCEL_COST_MULTIPLIER
    
    # Setup
    controller = RLController(model, device)
    physics = TinyPhysicsModel(physics_model_path, debug=False)
    
    # Custom controller wrapper for simulator
    class ControllerWrapper:
        def __init__(self, rl_ctrl):
            self.rl_ctrl = rl_ctrl
            self.log_probs = []
            self.trajectory = []  # (past, cur, fut, action, step_cost) for RL
            self.prev_lataccel = None
        
        def update(self, target_lataccel, current_lataccel, state, future_plan):
            action, log_prob = self.rl_ctrl.get_action(
                state, target_lataccel, current_lataccel, future_plan,
                deterministic=deterministic
            )
            if log_prob is not None:
                self.log_probs.append(log_prob)
            
            # Compute per-step cost (this is what we want to minimize)
            step_cost = 0.0
            lataccel_error = (target_lataccel - current_lataccel) ** 2
            step_cost += lataccel_error * LAT_ACCEL_COST_MULTIPLIER * 100
            
            if self.prev_lataccel is not None:
                jerk = ((current_lataccel - self.prev_lataccel) / DEL_T) ** 2
                step_cost += jerk * 100
            
            self.prev_lataccel = current_lataccel
            
            # Store trajectory with per-step cost
            if collect_trajectory and hasattr(self.rl_ctrl, 'last_inputs') and self.rl_ctrl.last_inputs is not None:
                past, cur, fut = self.rl_ctrl.last_inputs
                self.trajectory.append((past, cur, fut, action, step_cost))
            
            return action
    
    wrapper = ControllerWrapper(controller)
    sim = TinyPhysicsSimulator(physics, str(data_path), controller=wrapper, debug=False)
    
    # Run rollout
    sim.rollout()
    
    # Compute cost
    cost = sim.compute_cost()
    
    if collect_trajectory:
        return cost, wrapper.log_probs, sim.current_lataccel_history, sim.target_lataccel_history, wrapper.trajectory
    return cost, wrapper.log_probs, sim.current_lataccel_history, sim.target_lataccel_history


def _rollout_worker(args):
    """Worker function for parallel rollouts - collects trajectories with per-step costs"""
    data_path, physics_model_path, model_state_dict, hidden_dim = args
    
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    
    # Recreate model in worker (CPU only for simplicity)
    device = torch.device('cpu')
    model = SteerModelRL(d_model=hidden_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    try:
        result = rl_rollout(
            model, data_path, physics_model_path, device, 
            deterministic=True, collect_trajectory=True
        )
        cost, _, pred_lat, target_lat, trajectory = result
        return {
            'path': str(data_path), 
            'cost': cost, 
            'trajectory': trajectory,  # List of (past, cur, fut, action, step_cost)
            'success': True
        }
    except Exception as e:
        return {'path': str(data_path), 'cost': None, 'trajectory': [], 'success': False, 'error': str(e)}


def parallel_rollouts(model, data_files, physics_model_path, hidden_dim, n_workers=4):
    """
    Run multiple rollouts in parallel.
    Returns list of (trajectory, cost) tuples for REINFORCE.
    """
    # Get model state dict (CPU)
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    # Prepare args for workers
    worker_args = [
        (str(f), physics_model_path, model_state, hidden_dim)
        for f in data_files
    ]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_rollout_worker, args) for args in worker_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel rollouts"):
            result = future.result()
            if result['success'] and len(result.get('trajectory', [])) > 0:
                results.append({
                    'trajectory': result['trajectory'],
                    'cost': result['cost']['total_cost']
                })
    
    return results


def rl_train(model, data_files, physics_model_path, epochs, lr, device):
    """Phase 2: RL fine-tuning on simulator"""
    print("\n" + "="*50)
    print("Phase 2: RL Fine-tuning on Simulator")
    print("="*50)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Running average of costs for baseline
    cost_baseline = None
    
    for epoch in range(epochs):
        model.train()
        epoch_costs = []
        epoch_losses = []
        
        # Sample subset of files for this epoch
        epoch_files = np.random.choice(data_files, size=min(20, len(data_files)), replace=False)
        
        for data_file in tqdm(epoch_files, desc=f"RL Epoch {epoch+1}/{epochs}"):
            # Rollout with stochastic policy
            cost, log_probs, pred_lat, target_lat = rl_rollout(
                model, data_file, physics_model_path, device, deterministic=False
            )
            
            total_cost = cost['total_cost']
            epoch_costs.append(total_cost)
            
            if len(log_probs) == 0:
                continue
            
            # Update baseline (moving average)
            if cost_baseline is None:
                cost_baseline = total_cost
            else:
                cost_baseline = 0.9 * cost_baseline + 0.1 * total_cost
            
            # Policy gradient loss
            # We want to minimize cost, so: loss = log_prob * (cost - baseline)
            log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, device=device)
            advantage = total_cost - cost_baseline
            pg_loss = (log_probs_tensor * advantage).mean()
            
            # Add entropy bonus for exploration
            entropy_bonus = -0.01 * log_probs_tensor.mean()
            
            loss = pg_loss + entropy_bonus
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_cost = np.mean(epoch_costs)
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"RL Epoch {epoch+1}: avg_cost={avg_cost:.2f}, baseline={cost_baseline:.2f}, loss={avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'cost_baseline': cost_baseline,
            'epoch': epoch,
        }, f'models/steer_model_rl_epoch{epoch+1}.pt')
    
    return model


def curriculum_train(model, train_loader, val_loader, data_files, physics_model_path, 
                     total_epochs, device, args):
    """
    Curriculum learning:
    - Phase 1: Pure supervised until loss < threshold
    - Phase 2: Mixed supervised + RL 
    - Phase 3: Pure RL (final epochs)
    """
    print("\n" + "="*60)
    print("CURRICULUM TRAINING")
    print("="*60)
    
    # Setup checkpoint directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(args.output).parent / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")
    
    # Training log
    training_log = {
        'run_name': run_name,
        'args': vars(args),
        'epochs': [],
        'best_val_loss': float('inf'),
        'best_rl_cost': float('inf'),
    }
    log_path = ckpt_dir / "training_log.json"
    
    def save_log():
        """Save training log to disk"""
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2, default=str)
        print(f"\n📝 Training log saved: {log_path}")
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n\n⚠️  Interrupted (signal {signum})! Saving log...")
        save_log()
        exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(save_log)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
    
    # Thresholds for phase transitions
    supervised_loss_threshold = args.loss_threshold  # Start RL when supervised loss < this
    rl_mix_ratio = 0.0  # Fraction of RL vs supervised (0 = pure supervised, 1 = pure RL)
    
    cost_baseline = None
    best_val_loss = float('inf')
    current_phase = getattr(args, 'start_phase', 1)
    
    if current_phase == 3:
        print("*** Starting in PURE RL mode (--pure_rl) ***")
    
    for epoch in range(total_epochs):
        # Determine current phase and RL mix ratio
        if current_phase == 1:
            rl_mix_ratio = 0.0
            phase_name = "SUPERVISED"
        elif current_phase == 2:
            # Gradually increase RL ratio
            epochs_in_phase2 = epoch - args.phase2_start_epoch
            rl_mix_ratio = min(0.7, 0.1 + epochs_in_phase2 * 0.1)  # 0.1 → 0.7
            phase_name = f"MIXED (RL={rl_mix_ratio:.0%})"
        else:  # Phase 3 - Pure RL
            rl_mix_ratio = 1.0  # 100% RL
            phase_name = "PURE RL (100%)"
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{total_epochs} | Phase {current_phase}: {phase_name}")
        print(f"{'='*60}")
        
        model.train()
        train_loss = 0
        rl_costs = []
        
        # === SUPERVISED PORTION ===
        if rl_mix_ratio < 1.0:
            n_supervised_batches = int(len(train_loader) * (1 - rl_mix_ratio))
            supervised_iter = iter(train_loader)
            
            for _ in tqdm(range(n_supervised_batches), desc="Supervised"):
                try:
                    past, cur, fut, tgt = next(supervised_iter)
                except StopIteration:
                    supervised_iter = iter(train_loader)
                    past, cur, fut, tgt = next(supervised_iter)
                
                past, cur, fut, tgt = past.to(device), cur.to(device), fut.to(device), tgt.to(device)
                
                mean, _, _ = model(past, cur, fut, deterministic=True)
                loss = F.mse_loss(mean, tgt)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= max(n_supervised_batches, 1)
        
        # === RL PORTION (Batch collection + optimization) ===
        if rl_mix_ratio > 0 and data_files:
            n_rl_episodes = max(20, int(50 * rl_mix_ratio))  # 20-50 episodes per epoch
            episode_files = list(np.random.choice(data_files, size=min(n_rl_episodes, len(data_files)), replace=False))
            
            # Collect trajectories from rollouts (parallel if n_workers > 1)
            all_trajectories = []  # List of (trajectory, cost)
            
            model.eval()  # No grad during rollout collection
            n_workers = min(args.n_workers, len(episode_files))
            
            if n_workers > 1:
                # Parallel collection
                results = parallel_rollouts(model, episode_files, physics_model_path, args.hidden, n_workers)
                for r in results:
                    total_cost = r['cost']['total_cost'] if isinstance(r['cost'], dict) else r['cost']
                    rl_costs.append(total_cost)
                    all_trajectories.append((r['trajectory'], total_cost))
            else:
                # Sequential collection
                for data_file in tqdm(episode_files, desc="RL Episodes"):
                    try:
                        result = rl_rollout(
                            model, data_file, physics_model_path, device, 
                            deterministic=True, collect_trajectory=True
                        )
                        cost, log_probs, _, _, trajectory = result
                    except Exception as e:
                        print(f"RL rollout failed: {e}")
                        continue
                    
                    total_cost = cost['total_cost']
                    rl_costs.append(total_cost)
                    
                    if len(trajectory) > 0:
                        all_trajectories.append((trajectory, total_cost))
            
            # Update baseline to mean of this batch
            if rl_costs:
                batch_mean = np.mean(rl_costs)
                if cost_baseline is None:
                    cost_baseline = batch_mean
                else:
                    cost_baseline = 0.8 * cost_baseline + 0.2 * batch_mean
            
            # === Reward-Weighted Regression (more stable than REINFORCE) ===
            if all_trajectories and len(all_trajectories) >= 3:
                model.train()
                
                # Collect ALL per-step samples with costs
                all_samples = []
                
                for trajectory, episode_cost in all_trajectories:
                    for step in trajectory:
                        if len(step) == 5:
                            all_samples.append(step)
                
                if len(all_samples) < 100:
                    print(f"  [RL] Not enough samples: {len(all_samples)}")
                else:
                    step_costs = np.array([s[4] for s in all_samples])
                    episode_costs = np.array([cost for _, cost in all_trajectories])
                    
                    # Use FIXED baseline from first epoch (critical for stability!)
                    if not hasattr(args, '_fixed_step_baseline'):
                        args._fixed_step_baseline = np.percentile(step_costs, 50)  # Median of first epoch
                        print(f"  [RL] Setting FIXED baseline: {args._fixed_step_baseline:.2f}")
                    
                    fixed_baseline = args._fixed_step_baseline
                    
                    # Find actions BETTER than fixed baseline (not current batch!)
                    good_mask = step_costs < fixed_baseline
                    n_good = good_mask.sum()
                    
                    print(f"  [RL] Episodes: cost_mean={episode_costs.mean():.1f}, cost_std={episode_costs.std():.1f}")
                    print(f"  [RL] Steps: {len(all_samples)}, FIXED_baseline={fixed_baseline:.2f}, good={n_good}/{len(all_samples)} ({100*n_good/len(all_samples):.1f}%)")
                    
                    if n_good >= 50:
                        # Collect only GOOD samples (beat the fixed baseline)
                        good_samples = []
                        for i, sample in enumerate(all_samples):
                            if good_mask[i]:
                                # Weight by improvement over baseline
                                improvement = (fixed_baseline - step_costs[i]) / fixed_baseline
                                weight = max(0.1, min(1.0, improvement * 2))
                                good_samples.append((*sample[:4], weight))
                        
                        np.random.shuffle(good_samples)
                        batch = good_samples[:min(500, len(good_samples))]
                        
                        past_batch = torch.tensor(np.stack([s[0] for s in batch]), dtype=torch.float32).to(device)
                        cur_batch = torch.tensor(np.stack([s[1] for s in batch]), dtype=torch.float32).to(device)
                        fut_batch = torch.tensor(np.stack([s[2] for s in batch]), dtype=torch.float32).to(device)
                        action_batch = torch.tensor([s[3] for s in batch], dtype=torch.float32).to(device)
                        weight_batch = torch.tensor([s[4] for s in batch], dtype=torch.float32).to(device)
                        
                        # Forward pass
                        mean, std, _ = model(past_batch, cur_batch, fut_batch, deterministic=True)
                        
                        # Weighted MSE - push mean toward good actions
                        mse = (mean - action_batch) ** 2
                        loss = (weight_batch * mse).mean()
                        
                        optimizer.zero_grad()
                        loss.backward()
                        
                        total_grad_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_grad_norm += p.grad.data.norm(2).item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        print(f"  [RL] MSE loss={loss.item():.6f}, grad_norm={total_grad_norm:.4f}, std_mean={std.mean().item():.4f}")
                    else:
                        print(f"  [RL] DEGRADED: Only {n_good} good actions - model worse than baseline!")
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past, cur, fut, tgt in val_loader:
                past, cur, fut, tgt = past.to(device), cur.to(device), fut.to(device), tgt.to(device)
                mean, _, _ = model(past, cur, fut, deterministic=True)
                val_loss += F.mse_loss(mean, tgt).item() * past.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        # Compute RL cost mean
        avg_rl_cost = np.mean(rl_costs) if rl_costs else None
        
        # Print stats
        rl_cost_str = f", rl_cost={avg_rl_cost:.1f}" if avg_rl_cost else ""
        print(f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}{rl_cost_str}")
        
        # Log epoch data
        epoch_data = {
            'epoch': epoch + 1,
            'phase': current_phase,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rl_cost': avg_rl_cost,
            'lr': optimizer.param_groups[0]['lr'],
        }
        training_log['epochs'].append(epoch_data)
        
        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save({
            'model_state': model.state_dict(),
            'model_type': 'rl',
            'hidden': args.hidden,
            'epoch': epoch,
            'phase': current_phase,
            'val_loss': val_loss,
            'rl_cost': avg_rl_cost,
        }, ckpt_path)
        
        # Save best model
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_log['best_val_loss'] = val_loss
            is_best = True
        
        if avg_rl_cost and avg_rl_cost < training_log['best_rl_cost']:
            training_log['best_rl_cost'] = avg_rl_cost
            is_best = True
        
        if is_best:
            torch.save({
                'model_state': model.state_dict(),
                'model_type': 'rl',
                'hidden': args.hidden,
                'epoch': epoch,
                'phase': current_phase,
            }, args.output)
            print(f"  → Saved best model")
        
        # Phase transitions
        forced_supervised = getattr(args, 'supervised_epochs', 0)
        if current_phase == 1:
            if forced_supervised > 0 and epoch + 1 >= forced_supervised:
                # Forced supervised epochs complete → skip to phase 3 (pure RL)
                print(f"\n*** {forced_supervised} supervised epoch(s) complete → Starting Pure RL ***")
                current_phase = 3
            elif forced_supervised == 0 and val_loss < supervised_loss_threshold:
                print(f"\n*** Loss {val_loss:.6f} < threshold {supervised_loss_threshold} → Starting Phase 2 ***")
                current_phase = 2
                args.phase2_start_epoch = epoch
        
        # Move to phase 3 in last 20% of training (only if not already in phase 3)
        if current_phase == 2 and epoch >= total_epochs * 0.8:
            print(f"\n*** Entering Phase 3: Pure RL ***")
            current_phase = 3
    
    # Final log save
    save_log()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best val_loss: {training_log['best_val_loss']:.6f}")
    if training_log['best_rl_cost'] < float('inf'):
        print(f"Best rl_cost: {training_log['best_rl_cost']:.1f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Best model: {args.output}")
    
    return model


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SteerModelRL(d_model=args.hidden).to(device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}, phase {ckpt.get('phase', '?')}")
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load supervised data
    samples = load_supervised_data(Path(args.data_path), max_files=args.max_files)
    
    n_val = int(len(samples) * 0.1)
    train_samples = samples[:-n_val]
    val_samples = samples[-n_val:]
    
    train_ds = SupervisedDataset(train_samples)
    val_ds = SupervisedDataset(val_samples)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # Get data files for RL
    data_files = sorted(Path(args.data_path).glob("*.csv"))
    if args.max_files:
        data_files = data_files[:args.max_files]
    
    # Curriculum training
    args.phase2_start_epoch = 0  # Will be set when phase 2 starts
    args.start_phase = 3 if args.pure_rl else 1  # Skip to phase 3 if --pure_rl
    model = curriculum_train(
        model, train_loader, val_loader, data_files, 
        args.physics_model, args.epochs, device, args
    )
    
    print(f"\nTraining complete! Best model saved to: {args.output}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA/MPS compatibility
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--physics_model", type=str, default="models/tinyphysics.onnx")
    parser.add_argument("--output", type=str, default="models/steer_model_rl.pt")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss_threshold", type=float, default=0.005, 
                        help="Start RL when supervised loss drops below this")
    parser.add_argument("--supervised_epochs", type=int, default=0,
                        help="Force N supervised epochs before RL (0=use loss_threshold)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Parallel workers for sim rollouts")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (path to .pt file)")
    parser.add_argument("--pure_rl", action="store_true",
                        help="Skip to pure RL phase (requires --resume)")
    args = parser.parse_args()
    
    main(args)

