"""
Train learned dynamics model for multi-step prediction.
Predicts how state evolves over the next N timesteps (default 10).

Can be used in two modes:
1. Single-step: Predict next state given current state and action
2. Multi-step: Predict next N states given current state and future actions
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import random
from tqdm import tqdm

# Constants from tinyphysics
CONTROL_START_IDX = 100
HISTORY_LEN = 10
PREDICT_LEN = 10  # Predict 10 timesteps into the future

# Features
HISTORY_FEATURES = ['vEgo', 'aEgo', 'rollLateralAcceleration', 'currentLateralAcceleration', 'steerCommand']
STATE_FEATURES = ['currentLateralAcceleration', 'vEgo', 'aEgo', 'rollLateralAcceleration']
TARGET_FEATURE = 'currentLateralAcceleration'


class MultiStepDynamicsDataset(Dataset):
    """Dataset for learning multi-step vehicle dynamics."""

    def __init__(self, segment_paths, history_len=HISTORY_LEN, predict_len=PREDICT_LEN):
        self.history_len = history_len
        self.predict_len = predict_len
        self.samples = []

        for path in tqdm(segment_paths, desc="Loading segments"):
            df = pd.read_csv(path)
            # Valid range: need history after control starts, predict_len steps into future
            start_idx = CONTROL_START_IDX + history_len
            end_idx = len(df) - predict_len

            for t in range(start_idx, end_idx):
                self.samples.append((path, t))

        # Load all data into memory for speed
        self.data_cache = {}
        for path in segment_paths:
            self.data_cache[path] = pd.read_csv(path)

        # Compute normalization stats
        self._compute_stats()
        print(f"Created {len(self.samples)} samples")

    def _compute_stats(self):
        """Compute mean/std for normalization."""
        all_history = []
        all_actions = []
        all_states = []

        # Sample subset for computing stats
        sample_indices = random.sample(range(len(self.samples)), min(10000, len(self.samples)))

        for idx in sample_indices:
            path, t = self.samples[idx]
            df = self.data_cache[path]
            hist = df[HISTORY_FEATURES].values[t-self.history_len:t]
            actions = df['steerCommand'].values[t:t+self.predict_len]
            states = df[STATE_FEATURES].values[t:t+self.predict_len]

            all_history.append(hist.flatten())
            all_actions.append(actions)
            all_states.append(states.flatten())

        all_history = np.array(all_history)
        all_actions = np.array(all_actions)
        all_states = np.array(all_states)

        self.history_mean = all_history.mean(axis=0)
        self.history_std = all_history.std(axis=0) + 1e-8
        self.action_mean = all_actions.mean()
        self.action_std = all_actions.std() + 1e-8
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8

        # Also compute per-feature stats for state
        self.state_feature_mean = np.array([all_states[:, i::len(STATE_FEATURES)].mean() for i in range(len(STATE_FEATURES))])
        self.state_feature_std = np.array([all_states[:, i::len(STATE_FEATURES)].std() + 1e-8 for i in range(len(STATE_FEATURES))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, t = self.samples[idx]
        df = self.data_cache[path]

        # History: [t-history_len, t) - history_len steps of features
        history = df[HISTORY_FEATURES].values[t-self.history_len:t].flatten()

        # Current state at time t (starting point for prediction)
        current_state = df[STATE_FEATURES].values[t]

        # Future actions: [t, t+predict_len)
        future_actions = df['steerCommand'].values[t:t+self.predict_len]

        # Future states: [t+1, t+predict_len+1) - what we want to predict
        future_states = df[STATE_FEATURES].values[t+1:t+self.predict_len+1]

        # Normalize history and actions
        history_norm = (history - self.history_mean) / self.history_std
        future_actions_norm = (future_actions - self.action_mean) / self.action_std

        # Normalize current and future states per-feature
        current_state_norm = (current_state - self.state_feature_mean) / self.state_feature_std
        future_states_norm = (future_states - self.state_feature_mean) / self.state_feature_std

        return {
            'history': torch.FloatTensor(history_norm),
            'current_state': torch.FloatTensor(current_state_norm),
            'future_actions': torch.FloatTensor(future_actions_norm),
            'future_states': torch.FloatTensor(future_states_norm),
            # Also return unnormalized for evaluation
            'future_states_raw': torch.FloatTensor(future_states),
            'current_state_raw': torch.FloatTensor(current_state),
        }


class MultiStepDynamicsModel(nn.Module):
    """
    Multi-step dynamics prediction model using LSTM.

    Architecture:
    - Encoder: processes history to get hidden state
    - Decoder: autoregressively predicts future states
    """

    def __init__(self, history_dim, state_dim, hidden_dim=128, num_layers=2, predict_len=PREDICT_LEN):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.predict_len = predict_len
        self.num_layers = num_layers

        # Encoder: history -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Recurrent decoder for autoregressive prediction
        # Input: current state + action
        self.decoder_lstm = nn.LSTM(
            input_size=state_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output: predict next state (as delta from current)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        # Initial hidden state projection
        self.init_h = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(hidden_dim, hidden_dim * num_layers)

    def forward(self, history, current_state, future_actions, future_states=None, teacher_forcing_ratio=0.5):
        """
        Args:
            history: (batch, history_dim) - flattened history features
            current_state: (batch, state_dim) - current state
            future_actions: (batch, predict_len) - future actions
            future_states: (batch, predict_len, state_dim) - ground truth for teacher forcing
            teacher_forcing_ratio: probability of using ground truth during training

        Returns:
            predictions: (batch, predict_len, state_dim) - predicted future states
        """
        batch_size = history.shape[0]

        # Encode history
        encoded = self.encoder(history)

        # Initialize LSTM hidden state from encoded history
        h = self.init_h(encoded).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = self.init_c(encoded).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Autoregressive decoding
        predictions = []
        prev_state = current_state

        for t in range(self.predict_len):
            # Input: previous state + current action
            action_t = future_actions[:, t:t+1]
            lstm_input = torch.cat([prev_state, action_t], dim=-1).unsqueeze(1)

            # LSTM step
            output, (h, c) = self.decoder_lstm(lstm_input, (h, c))

            # Predict delta (residual prediction)
            delta = self.output_head(output.squeeze(1))
            next_state = prev_state + delta

            predictions.append(next_state)

            # Teacher forcing: use ground truth with some probability
            if self.training and future_states is not None and random.random() < teacher_forcing_ratio:
                prev_state = future_states[:, t]
            else:
                prev_state = next_state

        return torch.stack(predictions, dim=1)


class DirectMultiStepModel(nn.Module):
    """
    Direct multi-step prediction without autoregression.
    Predicts all future states at once.
    """

    def __init__(self, history_dim, state_dim, hidden_dim=256, predict_len=PREDICT_LEN):
        super().__init__()
        self.state_dim = state_dim
        self.predict_len = predict_len

        # Input: history + current_state + all future actions
        input_dim = history_dim + state_dim + predict_len

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, predict_len * state_dim)
        )

    def forward(self, history, current_state, future_actions, **kwargs):
        batch_size = history.shape[0]
        x = torch.cat([history, current_state, future_actions], dim=-1)
        out = self.net(x)
        return out.view(batch_size, self.predict_len, self.state_dim)


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0
    total_lataccel_loss = 0

    for batch in loader:
        history = batch['history'].to(device)
        current_state = batch['current_state'].to(device)
        future_actions = batch['future_actions'].to(device)
        future_states = batch['future_states'].to(device)

        optimizer.zero_grad()

        if hasattr(model, 'decoder_lstm'):
            pred = model(history, current_state, future_actions, future_states, teacher_forcing_ratio)
        else:
            pred = model(history, current_state, future_actions)

        loss = criterion(pred, future_states)

        # Also track lataccel loss specifically (first feature)
        lataccel_loss = criterion(pred[:, :, 0], future_states[:, :, 0])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_lataccel_loss += lataccel_loss.item()

    return total_loss / len(loader), total_lataccel_loss / len(loader)


def eval_epoch(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    total_lataccel_loss = 0

    # Track per-timestep error in original units
    timestep_errors = np.zeros(PREDICT_LEN)
    count = 0

    with torch.no_grad():
        for batch in loader:
            history = batch['history'].to(device)
            current_state = batch['current_state'].to(device)
            future_actions = batch['future_actions'].to(device)
            future_states = batch['future_states'].to(device)
            future_states_raw = batch['future_states_raw'].to(device)

            pred = model(history, current_state, future_actions)
            loss = criterion(pred, future_states)
            lataccel_loss = criterion(pred[:, :, 0], future_states[:, :, 0])

            # Denormalize predictions for error in original units
            pred_raw = pred.cpu().numpy() * dataset.state_feature_std + dataset.state_feature_mean
            gt_raw = future_states_raw.cpu().numpy()

            # Lataccel error per timestep (first feature = currentLateralAcceleration)
            for t in range(PREDICT_LEN):
                timestep_errors[t] += np.mean((pred_raw[:, t, 0] - gt_raw[:, t, 0]) ** 2)

            total_loss += loss.item()
            total_lataccel_loss += lataccel_loss.item()
            count += 1

    return total_loss / len(loader), total_lataccel_loss / len(loader), timestep_errors / count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='logging_data/pid')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='models/dynamics_multistep.pt')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'direct'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--predict_len', type=int, default=PREDICT_LEN)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load segment paths
    segment_paths = sorted(glob(f'{args.data_path}/*.csv'))
    print(f"Found {len(segment_paths)} segments")

    # Random split by segment
    random.shuffle(segment_paths)
    split_idx = int(len(segment_paths) * (1 - args.val_split))
    train_paths = segment_paths[:split_idx]
    val_paths = segment_paths[split_idx:]
    print(f"Train segments: {len(train_paths)}, Val segments: {len(val_paths)}")

    # Datasets
    print("Loading training data...")
    train_dataset = MultiStepDynamicsDataset(train_paths, predict_len=args.predict_len)

    print("Loading validation data...")
    val_dataset = MultiStepDynamicsDataset(val_paths, predict_len=args.predict_len)
    # Use same normalization stats
    val_dataset.history_mean = train_dataset.history_mean
    val_dataset.history_std = train_dataset.history_std
    val_dataset.action_mean = train_dataset.action_mean
    val_dataset.action_std = train_dataset.action_std
    val_dataset.state_mean = train_dataset.state_mean
    val_dataset.state_std = train_dataset.state_std
    val_dataset.state_feature_mean = train_dataset.state_feature_mean
    val_dataset.state_feature_std = train_dataset.state_feature_std

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    history_dim = HISTORY_LEN * len(HISTORY_FEATURES)
    state_dim = len(STATE_FEATURES)

    if args.model_type == 'lstm':
        model = MultiStepDynamicsModel(
            history_dim=history_dim,
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            predict_len=args.predict_len
        )
    else:
        model = DirectMultiStepModel(
            history_dim=history_dim,
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            predict_len=args.predict_len
        )

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model type: {args.model_type}, Parameters: {num_params:,}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Decay teacher forcing ratio
        tf_ratio = max(0.0, 0.5 - epoch / (args.epochs * 2))

        train_loss, train_lataccel = train_epoch(model, train_loader, optimizer, criterion, device, tf_ratio)
        val_loss, val_lataccel, timestep_errors = eval_epoch(model, val_loader, criterion, device, val_dataset)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} (lat: {train_lataccel:.6f}) | Val: {val_loss:.6f} (lat: {val_lataccel:.6f}) | LR: {lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save model and normalization stats
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': args.model_type,
                'history_dim': history_dim,
                'state_dim': state_dim,
                'hidden_dim': args.hidden_dim,
                'predict_len': args.predict_len,
                'history_len': HISTORY_LEN,
                'history_features': HISTORY_FEATURES,
                'state_features': STATE_FEATURES,
                'history_mean': train_dataset.history_mean,
                'history_std': train_dataset.history_std,
                'action_mean': train_dataset.action_mean,
                'action_std': train_dataset.action_std,
                'state_feature_mean': train_dataset.state_feature_mean,
                'state_feature_std': train_dataset.state_feature_std,
            }, args.save_path)
            print(f"  -> Saved best model to {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Best Model")
    print("=" * 60)

    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, timestep_errors = eval_epoch(model, val_loader, criterion, device, val_dataset)

    print("\nPer-timestep lataccel MSE (in original units):")
    for t in range(args.predict_len):
        rmse = np.sqrt(timestep_errors[t])
        print(f"  t+{t+1:2d}: MSE={timestep_errors[t]:.6f}, RMSE={rmse:.4f} m/s²")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # Plot training curve
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(train_losses, label='Train')
        axes[0].plot(val_losses, label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Training Curve')

        axes[1].bar(range(1, args.predict_len + 1), np.sqrt(timestep_errors))
        axes[1].set_xlabel('Timestep into future')
        axes[1].set_ylabel('RMSE (m/s²)')
        axes[1].set_title('Prediction Error by Timestep')

        plt.tight_layout()
        plot_path = args.save_path.replace('.pt', '_training.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training plot to {plot_path}")
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")


if __name__ == '__main__':
    main()
