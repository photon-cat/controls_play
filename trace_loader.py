import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from trace_logging import TRACE_COLUMNS

DEFAULT_DEL_T = 0.1
DEFAULT_STEER_RANGE = (-2, 2)
DEFAULT_MAX_ACC_DELTA = 0.5


class TraceValidationError(RuntimeError):
  pass


@dataclass
class FuturePlanSnapshot:
  lataccel: List[float]
  roll_lataccel: List[float]
  v_ego: List[float]
  a_ego: List[float]


@dataclass
class TraceRecord:
  name: str
  table: pd.DataFrame
  future_plans: Dict[int, FuturePlanSnapshot]
  del_t: float = DEFAULT_DEL_T
  steer_range: Tuple[float, float] = DEFAULT_STEER_RANGE
  max_acc_delta: float = DEFAULT_MAX_ACC_DELTA

  def get_future_plan(self, step: int) -> Optional[FuturePlanSnapshot]:
    return self.future_plans.get(int(step))

  def compute_jerk(self) -> np.ndarray:
    return np.gradient(self.table["current_lataccel"].values, self.del_t)


def _normalize_units(table: pd.DataFrame) -> pd.DataFrame:
  normalized = table.copy()
  if "v_ego_mph" in normalized.columns and "v_ego" not in normalized.columns:
    normalized["v_ego"] = normalized["v_ego_mph"] * 0.44704
  if "roll_lataccel_g" in normalized.columns and "roll_lataccel" not in normalized.columns:
    normalized["roll_lataccel"] = normalized["roll_lataccel_g"] * 9.81
  return normalized


def _validate_columns(table: pd.DataFrame) -> None:
  missing = [col for col in TRACE_COLUMNS if col not in table.columns]
  if missing:
    raise TraceValidationError(f"Trace is missing required columns: {missing}")


def _load_inline_future_plans(table: pd.DataFrame) -> Dict[int, FuturePlanSnapshot]:
  plans: Dict[int, FuturePlanSnapshot] = {}
  for _, row in table.iterrows():
    future_json = row.get("future_plan_json")
    if not future_json or not isinstance(future_json, str):
      continue
    payload = json.loads(future_json)
    plans[int(row["step"])] = FuturePlanSnapshot(
      lataccel=list(payload.get("lataccel", [])),
      roll_lataccel=list(payload.get("roll_lataccel", [])),
      v_ego=list(payload.get("v_ego", [])),
      a_ego=list(payload.get("a_ego", [])),
    )
  return plans


def _load_sidecar_future_plans(sidecar_path: Path) -> Dict[int, FuturePlanSnapshot]:
  plans: Dict[int, FuturePlanSnapshot] = {}
  if not sidecar_path.exists():
    return plans
  with sidecar_path.open() as fh:
    for line in fh:
      record = json.loads(line)
      step = int(record["step"])
      payload = record.get("future_plan", {})
      plans[step] = FuturePlanSnapshot(
        lataccel=list(payload.get("lataccel", [])),
        roll_lataccel=list(payload.get("roll_lataccel", [])),
        v_ego=list(payload.get("v_ego", [])),
        a_ego=list(payload.get("a_ego", [])),
      )
  return plans


def load_trace(
  trace_path: Path,
  future_sidecar: Optional[Path] = None,
  name: Optional[str] = None,
  del_t: float = DEFAULT_DEL_T,
  steer_range: Tuple[float, float] = DEFAULT_STEER_RANGE,
  max_acc_delta: float = DEFAULT_MAX_ACC_DELTA,
) -> TraceRecord:
  table = pd.read_csv(trace_path)
  table = _normalize_units(table)
  _validate_columns(table)

  plans = _load_inline_future_plans(table)
  sidecar_path = future_sidecar
  if future_sidecar is None:
    candidate = trace_path.with_suffix(trace_path.suffix + ".future.jsonl")
    if candidate.exists():
      sidecar_path = candidate
  if sidecar_path:
    plans.update(_load_sidecar_future_plans(Path(sidecar_path)))

  return TraceRecord(
    name=name or trace_path.stem,
    table=table,
    future_plans=plans,
    del_t=del_t,
    steer_range=steer_range,
    max_acc_delta=max_acc_delta,
  )


def load_traces(paths: List[Path], future_sidecars: Optional[List[Optional[Path]]] = None) -> List[TraceRecord]:
  records: List[TraceRecord] = []
  future_sidecars = future_sidecars or [None] * len(paths)
  for path, future in zip(paths, future_sidecars):
    records.append(load_trace(path, future_sidecar=future))
  return records
