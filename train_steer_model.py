"""
Train a neural network to predict steer_command from context window.

Input features per timestep:
  - v_ego, a_ego, roll_lataccel, target_lataccel, steer_command, measured_lataccel

Split: 85% train, 10% val, 5% test - BY SEGMENT FILE (no data leakage)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse

# Match tinyphysics constants
ACC_G = 9.81
CONTEXT_LENGTH = 10  # sliding window size
STEER_RANGE = [-2, 2]


def load_samples_from_files(files: list) -> list:
    """Load all samples from a list of CSV files."""
    samples = []
    for f in tqdm(files, desc="Loading"):
        df = pd.read_csv(f)
        
        # Only use rows with valid steerCommand (first ~100 rows before CONTROL_START_IDX)
        valid_mask = ~df['steerCommand'].isna()
        df = df[valid_mask].reset_index(drop=True)
        
        if len(df) < CONTEXT_LENGTH + 1:
            continue
        
        # Compute roll_lataccel same as tinyphysics
        roll_lataccel = np.sin(df['roll'].values) * ACC_G
        v_ego = df['vEgo'].values
        a_ego = df['aEgo'].values
        target_lataccel = df['targetLateralAcceleration'].values
        steer_command = -df['steerCommand'].values  # flip sign like tinyphysics
        
        # For training, we use target_lataccel as the "measured" lataccel
        measured_lataccel = target_lataccel.copy()
        
        # Build context windows
        for i in range(CONTEXT_LENGTH, len(df)):
            ctx_slice = slice(i - CONTEXT_LENGTH, i)
            context = np.stack([
                v_ego[ctx_slice],
                a_ego[ctx_slice],
                roll_lataccel[ctx_slice],
                target_lataccel[ctx_slice],
                steer_command[ctx_slice],
                measured_lataccel[ctx_slice],
            ], axis=1)  # shape: (CONTEXT_LENGTH, 6)
            
            current = np.array([
                v_ego[i],
                a_ego[i],
                roll_lataccel[i],
                target_lataccel[i],
                measured_lataccel[i],
            ])
            
            target_steer = steer_command[i]
            samples.append((context, current, target_steer))
    
    return samples


class SteerDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, noise_std: float = 0.0):
        self.samples = samples
        self.noise_std = noise_std  # inject noise to simulate inference errors
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ctx, cur, tgt = self.samples[idx]
        ctx = torch.tensor(ctx, dtype=torch.float32)
        cur = torch.tensor(cur, dtype=torch.float32)
        tgt = torch.tensor(tgt, dtype=torch.float32)
        
        # Simulate closed-loop errors without running simulator
        if self.noise_std > 0:
            # Simulate what measured_lataccel would be if model made errors
            # Simple first-order dynamics: lataccel responds to steer with lag
            k_steer = 0.7  # steering effectiveness
            tau = 0.3      # time constant
            alpha = 0.1 / tau  # dt / tau
            
            # Simulate error accumulation through context
            simulated_lat = ctx[0, 5].item()  # start from first measured
            for t in range(ctx.shape[0]):
                steer = ctx[t, 4].item()
                target = ctx[t, 3].item()
                roll = ctx[t, 2].item()
                
                # Simple dynamics: lataccel moves toward steer-induced + roll
                steer_effect = k_steer * steer + roll
                simulated_lat = simulated_lat + alpha * (steer_effect - simulated_lat)
                
                # Add some noise
                simulated_lat += np.random.randn() * self.noise_std * 0.1
                
                # Replace measured with simulated (shows error vs target)
                ctx[t, 5] = simulated_lat
            
            # Current measured is also simulated
            cur[4] = simulated_lat + np.random.randn() * self.noise_std * 0.1
        
        return ctx, cur, tgt


class SteerMLP(nn.Module):
    """Simple MLP that flattens context + current and predicts steer"""
    def __init__(self, context_len=CONTEXT_LENGTH, context_dim=6, current_dim=5, hidden=256):
        super().__init__()
        input_dim = context_len * context_dim + current_dim
        
        # Input normalization
        self.register_buffer('ctx_mean', torch.tensor([20., 0., 0., 0., 0., 0.]))
        self.register_buffer('ctx_std', torch.tensor([15., 1., 0.5, 1., 0.5, 1.]))
        self.register_buffer('cur_mean', torch.tensor([20., 0., 0., 0., 0.]))
        self.register_buffer('cur_std', torch.tensor([15., 1., 0.5, 1., 1.]))
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    
    def forward(self, context, current):
        context = (context - self.ctx_mean) / self.ctx_std
        current = (current - self.cur_mean) / self.cur_std
        x = torch.cat([context.flatten(1), current], dim=1)
        return self.net(x).squeeze(-1)


class SteerTransformer(nn.Module):
    """Transformer encoder over context window"""
    def __init__(self, context_len=CONTEXT_LENGTH, context_dim=6, current_dim=5, 
                 d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # Input normalization
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


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get all files and split BY SEGMENT (85/10/5)
    all_files = sorted(Path(args.data_path).glob("*.csv"))
    if args.max_files:
        all_files = all_files[:args.max_files]
    
    n_files = len(all_files)
    n_train = int(n_files * 0.85)
    n_val = int(n_files * 0.10)
    # n_test = n_files - n_train - n_val  (remaining 5%)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print(f"Split by segment: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Load data
    print("\n--- Loading train data ---")
    train_samples = load_samples_from_files(train_files)
    print(f"Train samples: {len(train_samples)}")
    
    print("\n--- Loading val data ---")
    val_samples = load_samples_from_files(val_files)
    print(f"Val samples: {len(val_samples)}")
    
    train_ds = SteerDataset(train_samples, noise_std=args.noise)  # noise for robustness
    val_ds = SteerDataset(val_samples, noise_std=0.0)  # no noise for clean eval
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Model
    if args.model == "mlp":
        model = SteerMLP(hidden=args.hidden).to(device)
    else:
        model = SteerTransformer(d_model=args.hidden, nhead=4, num_layers=2).to(device)
    
    print(f"\nModel: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training noise: {args.noise} (0=off, higher=more robust to errors)")
    print(f"Smoothness weight: {args.smooth} (penalizes steer changes)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_smooth_loss = 0
        for ctx, cur, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            ctx, cur, tgt = ctx.to(device), cur.to(device), tgt.to(device)
            
            pred = model(ctx, cur)
            
            # MSE loss on prediction
            mse_loss = nn.functional.mse_loss(pred, tgt)
            
            # Smoothness loss: penalize change from previous steer (ctx[:, -1, 4] is prev_steer)
            prev_steer = ctx[:, -1, 4]  # last timestep, steer column
            smooth_loss = nn.functional.mse_loss(pred, prev_steer)
            
            # Combined loss
            loss = mse_loss + args.smooth * smooth_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += mse_loss.item() * ctx.size(0)
            train_smooth_loss += smooth_loss.item() * ctx.size(0)
        
        train_loss /= len(train_ds)
        train_smooth_loss /= len(train_ds)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ctx, cur, tgt in val_loader:
                ctx, cur, tgt = ctx.to(device), cur.to(device), tgt.to(device)
                pred = model(ctx, cur)
                val_loss += nn.functional.mse_loss(pred, tgt).item() * ctx.size(0)
        val_loss /= len(val_ds)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, smooth_loss={train_smooth_loss:.6f}, val_loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'model_type': args.model,
                'hidden': args.hidden,
            }, args.output)
            print(f"  → Saved best model to {args.output}")
    
    # Final test evaluation
    print("\n--- Loading test data ---")
    test_samples = load_samples_from_files(test_files)
    print(f"Test samples: {len(test_samples)}")
    
    if test_samples:
        test_ds = SteerDataset(test_samples)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        
        # Load best model
        checkpoint = torch.load(args.output, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for ctx, cur, tgt in test_loader:
                ctx, cur, tgt = ctx.to(device), cur.to(device), tgt.to(device)
                pred = model(ctx, cur)
                test_loss += nn.functional.mse_loss(pred, tgt).item() * ctx.size(0)
        test_loss /= len(test_ds)
        
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS (best val model)")
        print(f"  Val loss:  {best_val_loss:.6f}")
        print(f"  Test loss: {test_loss:.6f}")
        print(f"{'='*50}")
    else:
        print(f"\nDone! Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="models/steer_model.pt", help="Output model path")
    parser.add_argument("--model", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_files", type=int, default=None, help="Limit files for quick testing")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise std for robustness (0=off)")
    parser.add_argument("--smooth", type=float, default=0.1, help="Smoothness loss weight (penalizes steer changes)")
    args = parser.parse_args()
    
    train(args)
