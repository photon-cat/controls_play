import argparse
import importlib
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile

from io import BytesIO
from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm.contrib.concurrent import process_map

from controllers import BaseController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent / "data"
DATASET_PATH_MINI = Path(__file__).resolve().parent / "data_mini"

class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray, List[float]]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> int:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample

  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, temperature=0.8))


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_path: str, controller: BaseController, debug: bool = False, log_path: str = None) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.controller = controller
    self.debug = debug
    self.log_path = log_path
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_history = [x[0] for x in state_target_futureplans]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_future = None
    self.current_lataccel = self.current_lataccel_history[-1]
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)
    # Logging
    self.sim_log = []

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    pred = self.sim_model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

    self.current_lataccel_history.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], future_plan=self.futureplan)
    if step_idx < CONTROL_START_IDX:
      action = self.data['steer_command'].values[step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)
    # Notify controller of actual action used (for learning from labeled data)
    if hasattr(self.controller, 'observe_action'):
      self.controller.observe_action(action, step_idx)

  def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
    state = self.data.iloc[step_idx]
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
        a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
      )
    )

  def step(self) -> None:
    state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.futureplan = futureplan
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)

    # Log timestep data
    log_entry = {
      't': self.step_idx,
      'vEgo': state.v_ego,
      'aEgo': state.a_ego,
      'roll': np.arcsin(state.roll_lataccel / ACC_G),  # Convert back to roll angle (radians)
      'rollLateralAcceleration': state.roll_lataccel,
      'targetLateralAcceleration': target,
      'currentLateralAcceleration': self.current_lataccel,
      'steerCommand': self.action_history[-1],
    }
    # Get controller log if available
    if hasattr(self.controller, 'get_log'):
      controller_log = self.controller.get_log()
      if controller_log:
        log_entry.update(controller_log)
    self.sim_log.append(log_entry)

    self.step_idx += 1

  def save_log(self, path: str = None) -> None:
    """Save simulation log to CSV"""
    save_path = path or self.log_path
    if save_path and self.sim_log:
      log_df = pd.DataFrame(self.sim_log)
      log_df.to_csv(save_path, index=False)
      print(f"Saved log to {save_path}")

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> Dict[str, float]:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def rollout(self) -> Dict[str, float]:
    if self.debug:
      plt.ion()
      fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

    for _ in range(CONTEXT_LENGTH, len(self.data)):
      self.step()
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
        plt.pause(0.01)

    if self.debug:
      plt.ioff()
      plt.show()

    # Save log if path specified
    if self.log_path:
      self.save_log()

    return self.compute_cost()


def get_available_controllers():
  return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout_with_log_folder(data_path, controller_type, model_path, log_folder=None):
  """Wrapper for multiprocessing that generates log_path from log_folder."""
  log_path = str(Path(log_folder) / f"{Path(data_path).stem}.csv") if log_folder else None
  return run_rollout(data_path, controller_type, model_path, debug=False, log_path=log_path)


def run_rollout(data_path, controller_type, model_path, debug=False, log_path=None):
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  controller = importlib.import_module(f'controllers.{controller_type}').Controller()
  # Inject physics model if controller supports it (for MPC)
  if hasattr(controller, 'set_physics_model'):
    controller.set_physics_model(tinyphysicsmodel)
  # Inject data path if controller supports it (for learning)
  if hasattr(controller, 'set_data_path'):
    controller.set_data_path(str(data_path))
  sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug, log_path=log_path)
  return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


def download_dataset():
  print("Downloading dataset (0.6G)...")
  DATASET_PATH.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(DATASET_URL) as resp:
    with zipfile.ZipFile(BytesIO(resp.read())) as z:
      for member in z.namelist():
        if not member.endswith('/'):
          with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:
            dest.write(src.read())


def get_unique_log_path(log_dir: Path, segment_name: str, controller: str) -> Path:
  """Get unique log file path, incrementing suffix if file exists."""
  base_path = log_dir / f"{segment_name}_{controller}.csv"
  if not base_path.exists():
    return base_path
  n = 2
  while True:
    path = log_dir / f"{segment_name}_{controller}_{n}.csv"
    if not path.exists():
      return path
    n += 1


def get_unique_log_folder(log_dir: Path, controller: str) -> Path:
  """Get unique log folder for multi-segment runs."""
  base_path = log_dir / controller
  if not base_path.exists():
    return base_path
  n = 2
  while True:
    path = log_dir / f"{controller}_{n}"
    if not path.exists():
      return path
    n += 1


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--num_segs_rand", type=int, default=None, help="Sample N random segments instead of first N")
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='pid', choices=available_controllers)
  parser.add_argument("--log_path", type=str, default=None, help="Folder to save simulation logs")
  args = parser.parse_args()

  # Use data_mini as fallback if data doesn't exist
  if not DATASET_PATH.exists():
    if DATASET_PATH_MINI.exists():
      print("Using data_mini (data folder not found)")
      DATASET_PATH = DATASET_PATH_MINI
    else:
      download_dataset()

  data_path = Path(args.data_path)
  # Also handle if user passes "data" but only data_mini exists
  if not data_path.exists() and str(data_path).endswith('data'):
    mini_path = Path(str(data_path) + '_mini')
    if mini_path.exists():
      print(f"Using {mini_path} (data folder not found)")
      data_path = mini_path
  if data_path.is_file():
    log_file = None
    if args.log_path:
      log_dir = Path(args.log_path)
      log_dir.mkdir(parents=True, exist_ok=True)
      segment_name = data_path.stem
      log_file = str(get_unique_log_path(log_dir, segment_name, args.controller))
    cost, _, _ = run_rollout(data_path, args.controller, args.model_path, debug=args.debug, log_path=log_file)
    print(f"\nAverage lataccel_cost: {cost['lataccel_cost']:>6.4}, average jerk_cost: {cost['jerk_cost']:>6.4}, average total_cost: {cost['total_cost']:>6.4}")
  elif data_path.is_dir():
    import random
    all_files = sorted(data_path.iterdir())
    if args.num_segs_rand is not None:
      files = random.sample(all_files, min(args.num_segs_rand, len(all_files)))
    else:
      files = all_files[:args.num_segs]

    log_folder = None
    if args.log_path:
      log_dir = Path(args.log_path)
      log_dir.mkdir(parents=True, exist_ok=True)
      log_folder = get_unique_log_folder(log_dir, args.controller)
      log_folder.mkdir(parents=True, exist_ok=True)
      print(f"Saving logs to {log_folder}")

    run_rollout_partial = partial(run_rollout_with_log_folder, controller_type=args.controller, model_path=args.model_path, log_folder=str(log_folder) if log_folder else None)
    results = process_map(run_rollout_partial, files, max_workers=16, chunksize=10)
    costs = [result[0] for result in results]
    costs_df = pd.DataFrame(costs)
    print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
    for cost in costs_df.columns:
      plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)
    plt.xlabel('costs')
    plt.ylabel('Frequency')
    plt.title('costs Distribution')
    plt.legend()
    plt.show()
