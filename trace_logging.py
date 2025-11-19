"""Minimal trace schema + writer for TinyPhysicsSimulator.

Rows are appended to a CSV with the following columns:
- step: integer simulator step index
- target_lataccel/current_lataccel: m/s^2
- steer_command: steering command issued to the simulator
- roll_lataccel, v_ego, a_ego: state inputs
- controller: controller identifier
- future_plan_ref: inline/sidecar marker for the serialized future plan
- future_plan_json: inline JSON payload when a sidecar is not used
- control_start_idx: constant reference for playback highlighting
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

FUTURE_PLAN_KEYS = ["lataccel", "roll_lataccel", "v_ego", "a_ego"]

TRACE_COLUMNS = [
  "step",
  "target_lataccel",
  "current_lataccel",
  "steer_command",
  "roll_lataccel",
  "v_ego",
  "a_ego",
  "controller",
  "future_plan_ref",
  "future_plan_json",
  "control_start_idx",
]


@dataclass
class TraceRow:
  step: int
  target_lataccel: float
  current_lataccel: float
  steer_command: float
  roll_lataccel: float
  v_ego: float
  a_ego: float
  controller: str
  future_plan_ref: str
  future_plan_json: str
  control_start_idx: int


class TraceLogger:
  """Streamed logger for TinyPhysicsSimulator traces.

  Parameters
  ----------
  trace_path: Path to the CSV file that will hold the per-step table.
  controller_name: Human readable controller identifier written to every row.
  control_start_idx: Index where control actions start (used for highlighting later).
  log_future_steps: Number of future-plan steps to capture for every entry.
  future_sidecar: Optional JSONL sidecar file. When provided, future-plan
    snapshots are written there and referenced from the CSV.
  """

  def __init__(
    self,
    trace_path: Path,
    controller_name: str,
    control_start_idx: int,
    log_future_steps: int = 0,
    future_sidecar: Optional[Path] = None,
  ) -> None:
    self.trace_path = Path(trace_path)
    self.controller_name = controller_name
    self.log_future_steps = max(0, int(log_future_steps))
    self.future_sidecar = Path(future_sidecar) if future_sidecar else None
    self.control_start_idx = control_start_idx

    self.trace_path.parent.mkdir(parents=True, exist_ok=True)
    self._csv_file = self.trace_path.open("w", newline="")
    self._writer = csv.DictWriter(self._csv_file, fieldnames=TRACE_COLUMNS)
    self._writer.writeheader()

    self._future_file = None
    if self.future_sidecar:
      self.future_sidecar.parent.mkdir(parents=True, exist_ok=True)
      self._future_file = self.future_sidecar.open("w")

  def _serialize_future_plan(self, future_plan: Dict[str, Iterable[float]], step: int) -> str:
    plan_snapshot = {
      key: list(future_plan.get(key, []))[: self.log_future_steps] for key in FUTURE_PLAN_KEYS
    }
    if self._future_file:
      record = {"step": step, "future_plan": plan_snapshot}
      self._future_file.write(json.dumps(record) + "\n")
      return f"sidecar:{self.future_sidecar.name}:{step}"
    return json.dumps(plan_snapshot)

  def log_step(
    self,
    step: int,
    target_lataccel: float,
    current_lataccel: float,
    steer_command: float,
    roll_lataccel: float,
    v_ego: float,
    a_ego: float,
    future_plan: Optional[Dict[str, Iterable[float]]] = None,
  ) -> None:
    future_ref = ""
    future_json = ""
    if future_plan and self.log_future_steps:
      serialized = self._serialize_future_plan(future_plan, step)
      if serialized.startswith("sidecar:"):
        future_ref = serialized
      else:
        future_json = serialized
        future_ref = "inline"
    row = TraceRow(
      step=step,
      target_lataccel=target_lataccel,
      current_lataccel=current_lataccel,
      steer_command=steer_command,
      roll_lataccel=roll_lataccel,
      v_ego=v_ego,
      a_ego=a_ego,
      controller=self.controller_name,
      future_plan_ref=future_ref,
      future_plan_json=future_json,
      control_start_idx=self.control_start_idx,
    )
    self._writer.writerow(row.__dict__)

  def close(self) -> None:
    if not self._csv_file.closed:
      self._csv_file.flush()
      self._csv_file.close()
    if self._future_file and not self._future_file.closed:
      self._future_file.flush()
      self._future_file.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
