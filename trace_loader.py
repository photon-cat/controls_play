"""Schema-aware loader for TinyPhysics trace logs.

The CSV schema matches the TraceLogger output in ``tinyphysics.py``:
  step (int): simulation step index
  target_lataccel (float): target lateral acceleration (m/s^2)
  current_lataccel (float): simulated lateral acceleration (m/s^2)
  steer_command (float): controller steering command
  roll_lataccel (float): lateral acceleration from road roll (m/s^2)
  v_ego (float): ego speed (m/s)
  a_ego (float): ego longitudinal acceleration (m/s^2)
  controller (str): controller name
  future_ref (str): "inline" when the future_plan column is populated, otherwise a step key used in the sidecar
  future_plan (str): JSON blob with future plan arrays (lataccel, roll_lataccel, v_ego, a_ego, window) when inline

Sidecar storage uses JSONL with objects of the form::
  {"step": <int>, "plan": {"lataccel": [...], "roll_lataccel": [...], "v_ego": [...], "a_ego": [...], "window": <int>}}

This loader validates the schema, reconstructs the future plans (inline or sidecar),
performs basic type coercion, and provides convenient accessors for visualization tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TRACE_COLUMNS = [
  "step",
  "target_lataccel",
  "current_lataccel",
  "steer_command",
  "roll_lataccel",
  "v_ego",
  "a_ego",
  "controller",
  "future_ref",
  "future_plan",
]


@dataclass
class FuturePlanSlice:
  lataccel: List[float]
  roll_lataccel: List[float]
  v_ego: List[float]
  a_ego: List[float]
  window: int


@dataclass
class TraceStep:
  step: int
  target_lataccel: float
  current_lataccel: float
  steer_command: float
  roll_lataccel: float
  v_ego: float
  a_ego: float
  controller: str
  future_ref: str
  future_plan: Optional[FuturePlanSlice]


@dataclass
class TraceDataset:
  path: Path
  dataframe: pd.DataFrame
  steps: List[TraceStep]
  future_sidecar: Optional[Path]

  def as_arrays(self) -> Dict[str, List[float]]:
    """Return core columns as primitive lists for plotting convenience."""
    return {col: self.dataframe[col].tolist() for col in self.dataframe.columns if col != "future_plan"}


class TraceFormatError(ValueError):
  pass


def _validate_columns(df: pd.DataFrame) -> None:
  missing = [c for c in TRACE_COLUMNS if c not in df.columns]
  if missing:
    raise TraceFormatError(f"Trace is missing columns: {missing}")


def _load_sidecar(future_path: Path) -> Dict[int, FuturePlanSlice]:
  plans: Dict[int, FuturePlanSlice] = {}
  if not future_path.exists():
    return plans
  for line in future_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    record = json.loads(line)
    plan = record.get("plan", {})
    plans[int(record["step"])] = FuturePlanSlice(
      lataccel=list(map(float, plan.get("lataccel", []))),
      roll_lataccel=list(map(float, plan.get("roll_lataccel", []))),
      v_ego=list(map(float, plan.get("v_ego", []))),
      a_ego=list(map(float, plan.get("a_ego", []))),
      window=int(plan.get("window", len(plan.get("lataccel", [])))),
    )
  return plans


def _coerce_future(blob: str, step: int) -> FuturePlanSlice:
  try:
    parsed = json.loads(blob)
  except json.JSONDecodeError as exc:
    raise TraceFormatError(f"Invalid future_plan JSON at step {step}") from exc
  return FuturePlanSlice(
    lataccel=list(map(float, parsed.get("lataccel", []))),
    roll_lataccel=list(map(float, parsed.get("roll_lataccel", []))),
    v_ego=list(map(float, parsed.get("v_ego", []))),
    a_ego=list(map(float, parsed.get("a_ego", []))),
    window=int(parsed.get("window", len(parsed.get("lataccel", [])))),
  )


def load_trace(trace_path: Path | str, future_path: Optional[Path | str] = None, expected_window: Optional[int] = None) -> TraceDataset:
  trace_path = Path(trace_path)
  df = pd.read_csv(trace_path)
  df.fillna("", inplace=True)
  _validate_columns(df)

  sidecar_path = Path(future_path) if future_path else trace_path.with_suffix(trace_path.suffix + ".future.jsonl")
  sidecar_plans = _load_sidecar(sidecar_path)

  steps: List[TraceStep] = []
  for _, row in df.iterrows():
    step = int(row["step"])
    future_blob = str(row.get("future_plan", ""))
    future_ref = str(row.get("future_ref", ""))
    future: Optional[FuturePlanSlice] = None
    if future_blob:
      future = _coerce_future(future_blob, step)
    elif future_ref:
      if not sidecar_plans:
        raise TraceFormatError(f"future_ref provided for step {step} but no sidecar was found")
      if int(float(future_ref)) not in sidecar_plans:
        raise TraceFormatError(f"future_ref {future_ref} missing from sidecar at step {step}")
      future = sidecar_plans[int(float(future_ref))]

    if expected_window is not None and future is not None and future.window != expected_window:
      raise TraceFormatError(f"Expected window {expected_window} but got {future.window} at step {step}")

    steps.append(TraceStep(
      step=step,
      target_lataccel=float(row["target_lataccel"]),
      current_lataccel=float(row["current_lataccel"]),
      steer_command=float(row["steer_command"]),
      roll_lataccel=float(row["roll_lataccel"]),
      v_ego=float(row["v_ego"]),
      a_ego=float(row["a_ego"]),
      controller=str(row["controller"]),
      future_ref=future_ref,
      future_plan=future,
    ))

  df["step"] = df["step"].astype(int)
  df.sort_values("step", inplace=True)
  return TraceDataset(path=trace_path, dataframe=df, steps=steps, future_sidecar=sidecar_path if sidecar_plans else None)
