"""
DAgger (Dataset Aggregation) training for the neural steering model.

The problem: Model trained on ground truth data sees different states at inference
when running on its own predictions (distribution shift).

Solution: 
1. Run the current model on the simulator
2. At each step, record the state AND what an expert would do
3. Add this "on-policy" data to training set
4. Retrain
5. Repeat

The expert can be:
- Ground truth (first 100 steps only)
- A well-tuned PID controller
- The optimal action computed from cost function
"""

import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, 
    CONTROL_START_IDX, CONTEXT_LENGTH, ACC_G, STEER_RANGE
)

NEURAL_CONTEXT = 10


class SteerTransformer(nn.Module):
    def __init__(self, context_len=NEURAL_CONTEXT, context_dim=6, current_dim=5, 
                 d_model=128, nhead=4, num_layers=2):
        super().__init__()
        
        self.register_buffer('ctx_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('ctx_std', torch.tensor([15., 1., 0.5, 1., 0.5, 1.]))
        self.register_buffer('cur_mean', torch.tensor([20., 0., 0., 0., 0.]))
        self.register_buffer('cur_std', torch.tensor([15., 1., 0.5, 1., 1.]))
        
        self.context_proj = nn.Linear(context_dim, d_model)
        self.current_proj = nn.Linear(current_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(context_len + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        
        ctx_emb = self.context_proj(context)
        cur_emb = self.current_proj(current).unsqueeze(1)
        seq = torch.cat([ctx_emb, cur_emb], dim=1)
        seq = seq + self.pos_emb
        out = self.transformer(seq)
        return self.head(out[:, -1]).squeeze(-1)


class NeuralControllerForDagger:
    """Neural controller that also records data for DAgger"""
    def __init__(self, model, expert_controller):
        self.model = model
        self.model.eval()
        self.expert = expert_controller
        
        self.context = []
        self.prev_steer = 0.0
        
        # Data collection
        self.collected_data = []
    
    @torch.no_grad()
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel
        
        # Record context before prediction
        obs = [v_ego, a_ego, roll_lataccel, target_lataccel, self.prev_steer, current_lataccel]
        self.context.append(obs)
        
        if len(self.context) > NEURAL_CONTEXT:
            self.context = self.context[-NEURAL_CONTEXT:]
        
        # Get expert's action (what we should have done)
        expert_steer = self.expert.update(target_lataccel, current_lataccel, state, future_plan)
        
        # Get neural model's prediction
        if len(self.context) >= NEURAL_CONTEXT:
            ctx = torch.tensor(self.context, dtype=torch.float32).unsqueeze(0)
            cur = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel],
                              dtype=torch.float32).unsqueeze(0)
            neural_steer = self.model(ctx, cur).item()
            neural_steer = np.clip(neural_steer, STEER_RANGE[0], STEER_RANGE[1])
            
            # Collect data: (context, current, expert_label)
            self.collected_data.append({
                'context': np.array(self.context),
                'current': np.array([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel]),
                'expert_steer': expert_steer,
                'neural_steer': neural_steer,
            })
        else:
            neural_steer = 0.3 * (target_lataccel - current_lataccel)
            neural_steer = np.clip(neural_steer, STEER_RANGE[0], STEER_RANGE[1])
        
        # Use neural model's output (we want to collect ON-POLICY data)
        self.prev_steer = neural_steer
        return neural_steer


def collect_dagger_data(model, data_files, physics_model_path, expert_name='pid_ff_scheduled_tune'):
    """Run neural model and collect on-policy data with expert labels"""
    
    # Load expert controller
    expert_module = importlib.import_module(f'controllers.{expert_name}')
    
    all_data = []
    
    for data_file in tqdm(data_files, desc="Collecting DAgger data"):
        # Fresh expert for each file
        expert = expert_module.Controller()
        
        # Create DAgger controller
        dagger_ctrl = NeuralControllerForDagger(model, expert)
        
        # Run simulation
        physics = TinyPhysicsModel(physics_model_path, debug=False)
        sim = TinyPhysicsSimulator(physics, str(data_file), controller=dagger_ctrl, debug=False)
        
        try:
            sim.rollout()
        except Exception as e:
            print(f"Error on {data_file}: {e}")
            continue
        
        all_data.extend(dagger_ctrl.collected_data)
    
    return all_data


def train_on_dagger_data(model, dagger_data, original_data, epochs=5, lr=1e-4, mix_ratio=0.5):
    """
    Train model on mix of original data and DAgger data.
    mix_ratio: fraction of each batch that comes from DAgger data
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Prepare DAgger data
    dagger_contexts = torch.tensor(np.array([d['context'] for d in dagger_data]), dtype=torch.float32)
    dagger_currents = torch.tensor(np.array([d['current'] for d in dagger_data]), dtype=torch.float32)
    dagger_labels = torch.tensor(np.array([d['expert_steer'] for d in dagger_data]), dtype=torch.float32)
    
    # Prepare original data
    orig_contexts = torch.tensor(np.array([d['context'] for d in original_data]), dtype=torch.float32)
    orig_currents = torch.tensor(np.array([d['current'] for d in original_data]), dtype=torch.float32)
    orig_labels = torch.tensor(np.array([d['label'] for d in original_data]), dtype=torch.float32)
    
    batch_size = 256
    n_dagger = len(dagger_data)
    n_orig = len(original_data)
    
    print(f"Training on {n_dagger} DAgger samples + {n_orig} original samples")
    
    for epoch in range(epochs):
        # Shuffle both datasets
        dagger_perm = torch.randperm(n_dagger)
        orig_perm = torch.randperm(n_orig)
        
        total_loss = 0
        n_batches = 0
        
        # Number of batches based on DAgger data
        n_batches_total = n_dagger // int(batch_size * mix_ratio) if mix_ratio > 0 else n_orig // batch_size
        
        for i in range(0, n_batches_total):
            # Sample from DAgger data
            n_dagger_batch = int(batch_size * mix_ratio)
            dagger_idx = dagger_perm[(i * n_dagger_batch) % n_dagger: ((i + 1) * n_dagger_batch) % n_dagger + n_dagger_batch]
            if len(dagger_idx) < n_dagger_batch:
                dagger_idx = dagger_perm[:n_dagger_batch]
            dagger_idx = dagger_idx[:n_dagger_batch]
            
            # Sample from original data
            n_orig_batch = batch_size - n_dagger_batch
            orig_idx = orig_perm[(i * n_orig_batch) % n_orig: ((i + 1) * n_orig_batch) % n_orig + n_orig_batch]
            if len(orig_idx) < n_orig_batch:
                orig_idx = orig_perm[:n_orig_batch]
            orig_idx = orig_idx[:n_orig_batch]
            
            # Combine batches
            ctx = torch.cat([dagger_contexts[dagger_idx], orig_contexts[orig_idx]]).to(device)
            cur = torch.cat([dagger_currents[dagger_idx], orig_currents[orig_idx]]).to(device)
            tgt = torch.cat([dagger_labels[dagger_idx], orig_labels[orig_idx]]).to(device)
            
            # Forward pass
            pred = model(ctx, cur)
            loss = nn.functional.mse_loss(pred, tgt)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    return model


def load_original_data(data_dir, max_files=1000):
    """Load original training data in DAgger-compatible format"""
    files = sorted(Path(data_dir).glob("*.csv"))[:max_files]
    
    samples = []
    for f in tqdm(files, desc="Loading original data"):
        df = pd.read_csv(f)
        valid_mask = ~df['steerCommand'].isna()
        df = df[valid_mask].reset_index(drop=True)
        
        if len(df) < NEURAL_CONTEXT + 1:
            continue
        
        roll_lataccel = np.sin(df['roll'].values) * ACC_G
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        target_lataccel = df['targetLateralAcceleration'].values
        steer_command = -df['steerCommand'].values
        measured_lataccel = target_lataccel.copy()
        
        for i in range(NEURAL_CONTEXT, len(df)):
            ctx_slice = slice(i - NEURAL_CONTEXT, i)
            context = np.stack([
                v_ego[ctx_slice], a_ego[ctx_slice], roll_lataccel[ctx_slice],
                target_lataccel[ctx_slice], steer_command[ctx_slice], measured_lataccel[ctx_slice]
            ], axis=1)
            
            current = np.array([v_ego[i], a_ego[i], roll_lataccel[i], target_lataccel[i], measured_lataccel[i]])
            
            samples.append({
                'context': context,
                'current': current,
                'label': steer_command[i]
            })
    
    return samples


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load current model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
    hidden = checkpoint.get('hidden', 128)
    model = SteerTransformer(d_model=hidden)
    model.load_state_dict(checkpoint['model_state'])
    
    # Load original training data
    print("Loading original training data...")
    original_data = load_original_data(args.data_path, max_files=args.max_files)
    print(f"Original data: {len(original_data)} samples")
    
    for iteration in range(args.iterations):
        print(f"\n{'='*50}")
        print(f"DAgger Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*50}")
        
        # Collect on-policy data
        data_files = sorted(Path(args.data_path).glob("*.csv"))[:args.max_files]
        dagger_data = collect_dagger_data(
            model, data_files, args.physics_model, args.expert
        )
        print(f"Collected {len(dagger_data)} on-policy samples")
        
        # Train on mixed data
        model = train_on_dagger_data(
            model, dagger_data, original_data,
            epochs=args.epochs_per_iter,
            lr=args.lr,
            mix_ratio=args.mix_ratio
        )
        
        # Save checkpoint
        output_path = args.output.replace('.pt', f'_dagger{iteration+1}.pt')
        torch.save({
            'model_state': model.state_dict(),
            'model_type': 'transformer',
            'hidden': hidden,
            'dagger_iteration': iteration + 1,
        }, output_path)
        print(f"Saved: {output_path}")
    
    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'model_type': 'transformer',
        'hidden': hidden,
        'dagger_iterations': args.iterations,
    }, args.output)
    print(f"\nFinal model saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAgger training for neural steering")
    parser.add_argument("--model_path", type=str, default="models/steer_model.pt", 
                        help="Initial model to improve")
    parser.add_argument("--output", type=str, default="models/steer_model_dagger.pt")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--physics_model", type=str, default="models/tinyphysics.onnx")
    parser.add_argument("--expert", type=str, default="pid_ff_scheduled_tune",
                        help="Expert controller to provide labels")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=3, help="DAgger iterations")
    parser.add_argument("--epochs_per_iter", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                        help="Fraction of batch from DAgger data (0.5 = 50/50 mix)")
    args = parser.parse_args()
    
    main(args)

