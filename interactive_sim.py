#!/usr/bin/env python3
"""
TinyPhysics interactive simulator (PyVista edition).

We treat the ONNX TinyPhysics model as a lateral-acceleration engine, wrap it in
simple kinematics (latacc -> yaw rate -> heading -> x/y), and render the road,
future plan, and controller telemetry with PyVista. Controllers can be swapped
on the fly, manual steering is only a keypress away, and a handful of runtime
disturbances make it easy to torture-test robustness.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pyvista as pv

from tinyphysics import (
  TinyPhysicsModel,
  ACC_G,
  CONTEXT_LENGTH,
  DEL_T,
  FUTURE_PLAN_STEPS,
  MAX_ACC_DELTA,
  State,
  FuturePlan,
  STEER_RANGE,
)

SCREEN_W, SCREEN_H = 1400, 900
SCALE = 8.0
LANE_WIDTH = 3.6
HISTORY_LEN = 200
FPS = int(1.0 / DEL_T)

WHITE = (245, 245, 245)
LIGHT_GRAY = (180, 180, 180)
GRAY = (90, 90, 90)
ORANGE = (250, 158, 65)
BLUE = (78, 149, 245)


@dataclass
class RoadSegment:
  length: float
  curvature: float
  bank: float
  label: str = ""


@dataclass
class Scenario:
  name: str
  segments: Sequence[RoadSegment]
  nominal_speed: float
  description: str
  color: Tuple[int, int, int] = (114, 221, 247)


@dataclass
class RoadSample:
  s: float
  x: float
  y: float
  heading: float
  curvature: float
  bank: float


def build_scenarios() -> List[Scenario]:
  scenarios: List[Scenario] = []
  scenarios.append(
    Scenario(
      name="Straight Crown",
      segments=[RoadSegment(length=400.0, curvature=0.0, bank=math.radians(4.5), label="crown")],
      nominal_speed=18.0,
      description="Long straight with aggressive crown for windup tests.",
      color=(114, 221, 247),
    )
  )
  scenarios.append(
    Scenario(
      name="S Curve",
      segments=[
        RoadSegment(60.0, 0.0, math.radians(1.0)),
        RoadSegment(80.0, 0.007, math.radians(2.0)),
        RoadSegment(60.0, -0.009, math.radians(-1.5)),
        RoadSegment(100.0, 0.0, 0.0),
      ],
      nominal_speed=20.0,
      description="Left-right transition for transient response tuning.",
      color=(250, 158, 65),
    )
  )
  scenarios.append(
    Scenario(
      name="Highway Sweep",
      segments=[
        RoadSegment(120.0, 0.0, math.radians(1.0)),
        RoadSegment(250.0, 0.0045, math.radians(5.0)),
        RoadSegment(80.0, 0.0, math.radians(1.0)),
      ],
      nominal_speed=32.0,
      description="High-speed curve for feed-forward validation.",
      color=(68, 214, 44),
    )
  )

  rng = random.Random(7)
  random_segments: List[RoadSegment] = []
  curv = 0.0
  for _ in range(18):
    curv += rng.uniform(-0.004, 0.004)
    curv = float(np.clip(curv, -0.01, 0.01))
    bank = math.radians(rng.uniform(-5.0, 5.0))
    random_segments.append(RoadSegment(length=rng.uniform(30.0, 70.0), curvature=curv, bank=bank, label="quirk"))
  scenarios.append(
    Scenario(
      name="Adversarial",
      segments=random_segments,
      nominal_speed=22.0,
      description="Randomized curvature/bank noise to torture controllers.",
      color=(180, 108, 255),
    )
  )
  return scenarios


class RoadGeometry:
  def __init__(self, scenario: Scenario, step: float = 2.0):
    self.scenario = scenario
    self.step = step
    self.samples: List[RoadSample] = []
    x = y = heading = 0.0
    s = 0.0
    for segment in scenario.segments:
      remaining = segment.length
      while remaining > 1e-6:
        ds = min(step, remaining)
        heading += segment.curvature * ds
        x += math.cos(heading) * ds
        y += math.sin(heading) * ds
        self.samples.append(RoadSample(s, x, y, heading, segment.curvature, segment.bank))
        s += ds
        remaining -= ds
    self.total_length = s
    if self.total_length <= 0:
      raise ValueError("Scenario must have positive length.")

  def sample_at_s(self, s: float) -> RoadSample:
    wrapped = (s % self.total_length + self.total_length) % self.total_length
    idx = int(wrapped / self.step) % len(self.samples)
    return self.samples[idx]

  def polyline(self) -> np.ndarray:
    return np.array([[p.x, p.y, 0.0] for p in self.samples], dtype=float)


class FaultToggles:
  def __init__(self, dt: float):
    self.dt = dt
    self.tire_failure = False
    self.lat_noise = False
    self.steer_delay_steps = 0
    self.steer_clip = False
    self.delay_buffer = deque([0.0], maxlen=1)

  def set_delay(self, seconds: float):
    steps = int(round(seconds / self.dt))
    self.steer_delay_steps = steps
    maxlen = max(1, steps + 1)
    self.delay_buffer = deque([0.0] * maxlen, maxlen=maxlen)

  def apply_to_command(self, command: float) -> float:
    if self.steer_clip:
      command = float(np.clip(command, -0.35, 0.35))
    if self.steer_delay_steps > 0:
      self.delay_buffer.append(command)
      command = self.delay_buffer.popleft()
    return command

  def apply_to_latacc(self, latacc: float) -> float:
    value = latacc
    if self.tire_failure:
      value = float(np.clip(value, -0.3 * ACC_G, 0.3 * ACC_G))
    if self.lat_noise:
      value += np.random.normal(0.0, 0.15)
    return value

  def description(self) -> str:
    active = []
    if self.tire_failure:
      active.append("TIRE±0.3g")
    if self.lat_noise:
      active.append("LAT_NOISE")
    if self.steer_delay_steps > 0:
      active.append(f"DELAY {self.steer_delay_steps * self.dt:.2f}s")
    if self.steer_clip:
      active.append("STEER_CLIP")
    return ", ".join(active) if active else "None"


class LatAccelEngine:
  """Thin wrapper around TinyPhysicsModel that maintains the rolling context."""

  def __init__(self, model: TinyPhysicsModel, init_state: State):
    self.model = model
    self.state_hist = deque([init_state] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.action_hist = deque([0.0] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.latacc_hist = deque([0.0] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.current_latacc = 0.0

  def reset(self, init_state: State):
    self.state_hist = deque([init_state] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.action_hist = deque([0.0] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.latacc_hist = deque([0.0] * CONTEXT_LENGTH, maxlen=CONTEXT_LENGTH)
    self.current_latacc = 0.0

  def step(self, commanded_steer: float, road_roll: float, v_ego: float, a_ego: float, faults: FaultToggles) -> Tuple[float, float]:
    applied = faults.apply_to_command(commanded_steer)
    state = State(roll_lataccel=road_roll, v_ego=v_ego, a_ego=a_ego)
    self.state_hist.append(state)
    self.action_hist.append(applied)
    latacc = self.model.get_current_lataccel(
      list(self.state_hist),
      list(self.action_hist),
      list(self.latacc_hist),
    )
    latacc = float(np.clip(latacc, self.current_latacc - MAX_ACC_DELTA, self.current_latacc + MAX_ACC_DELTA))
    latacc = faults.apply_to_latacc(latacc)
    self.latacc_hist.append(latacc)
    self.current_latacc = latacc
    return latacc, applied


@dataclass
class VehiclePose:
  x: float = 0.0
  y: float = 0.0
  heading: float = 0.0
  yaw_rate: float = 0.0


class VehicleKinematics:
  def __init__(self):
    self.pose = VehiclePose()

  def integrate(self, latacc: float, v_ego: float, dt: float):
    yaw_rate = 0.0
    if abs(v_ego) > 1e-3:
      yaw_rate = latacc / max(v_ego, 0.1)
    self.pose.yaw_rate = yaw_rate
    self.pose.heading += yaw_rate * dt
    self.pose.x += v_ego * math.cos(self.pose.heading) * dt
    self.pose.y += v_ego * math.sin(self.pose.heading) * dt


class SimulationLogger:
  def __init__(self):
    self.file = None
    self.writer = None

  def toggle(self, scenario: Scenario, controller_name: str) -> bool:
    if self.file:
      self.close()
      return False
    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / f"sim_{scenario.name.replace(' ', '_')}_{controller_name}.csv"
    self.file = open(path, "w", newline="")
    self.writer = csv.writer(self.file)
    self.writer.writerow([
      "time",
      "scenario",
      "controller",
      "step",
      "target_lataccel",
      "current_lataccel",
      "road_roll",
      "v_ego",
      "yaw_rate",
      "commanded_steer",
      "applied_steer",
      "x",
      "y",
    ])
    return True

  def log(
    self,
    scenario: Scenario,
    controller_name: str,
    sim_time: float,
    step: int,
    target_lataccel: float,
    current_lataccel: float,
    road_roll: float,
    v_ego: float,
    yaw_rate: float,
    commanded: float,
    applied: float,
    pose: VehiclePose,
  ):
    if not self.writer:
      return
    self.writer.writerow([
      f"{sim_time:.2f}",
      scenario.name,
      controller_name,
      step,
      f"{target_lataccel:.4f}",
      f"{current_lataccel:.4f}",
      f"{road_roll:.4f}",
      f"{v_ego:.3f}",
      f"{yaw_rate:.4f}",
      f"{commanded:.4f}",
      f"{applied:.4f}",
      f"{pose.x:.3f}",
      f"{pose.y:.3f}",
    ])

  def close(self):
    if self.file:
      self.file.close()
      self.file = None
      self.writer = None


class InteractiveSimulator:
  def __init__(self, model_path: str, controllers: Sequence[str]):
    self.controllers = list(controllers)
    if not self.controllers:
      raise SystemExit("No controllers discovered in controllers/ directory.")

    self.scenarios = build_scenarios()
    self.controller_idx = 0
    self.scenario_idx = 0

    self.speed_mps = self.current_scenario.nominal_speed
    self.manual_mode = False
    self.manual_command = 0.0
    self.show_track = True
    self.show_future = True
    self.show_overlays = True
    self.show_graphs = True

    self.sim_time = 0.0
    self.step_count = 0
    self.car_s = 0.0
    self.future_plan_cache: List[Tuple[float, float]] = []

    self.target_history = deque(maxlen=HISTORY_LEN)
    self.current_history = deque(maxlen=HISTORY_LEN)
    self.steer_history = deque(maxlen=HISTORY_LEN)
    self.roll_history = deque(maxlen=HISTORY_LEN)
    self.yaw_history = deque(maxlen=HISTORY_LEN)

    self.plotter = pv.Plotter(window_size=(SCREEN_W, SCREEN_H))
    self.plotter.set_background("black")
    self.plotter.enable_parallel_projection()

    self.track_actors: List[pv.Actor] = []
    self.future_actor: pv.Actor | None = None
    self.car_actor: pv.Actor | None = None

    self.status_actor = self.plotter.add_text("", position="upper_left", font_size=12, color="white")
    self.status_corner_idx = 2  # upper_left in vtkCornerAnnotation
    self.help_actor = self.plotter.add_text("", position="lower_left", font_size=10, color="white")
    self.help_corner_idx = 0  # lower_left
    self._init_help_text()
    self._init_charts()

    self.model = TinyPhysicsModel(model_path, debug=False, num_threads=2)
    init_state = State(roll_lataccel=0.0, v_ego=self.speed_mps, a_ego=0.0)
    self.lat_engine = LatAccelEngine(self.model, init_state)
    self.vehicle = VehicleKinematics()
    self.faults = FaultToggles(DEL_T)
    self.logger = SimulationLogger()

    self._load_controller()
    self._build_road()
    self._register_key_events()

    self.plotter.add_callback(self._on_timer, interval=int(DEL_T * 1000))

  @property
  def current_scenario(self) -> Scenario:
    return self.scenarios[self.scenario_idx]

  @property
  def controller_name(self) -> str:
    return self.controllers[self.controller_idx]

  def _init_help_text(self):
    lines = [
      "SPACE: toggle manual | A/D steer | W decay | S zero",
      "-/= speed | 1-9 controller | F1-F4 scenario | R reset",
      "T track | F future plan | G graphs | TAB overlays",
      "L log | F5 tire | F6 noise | F7 delay | F8 steer clip",
    ]
    self._set_text(self.help_actor, "\n".join(lines), self.help_corner_idx)

  def _init_charts(self):
    self.chart_latacc = pv.Chart2D(size=(0.34, 0.25), loc=(0.02, 0.70))
    self.lat_line_target = self.chart_latacc.plot([0], [0])
    self._configure_plot(self.lat_line_target, color="lime", label="target")
    self.lat_line_current = self.chart_latacc.plot([0], [0])
    self._configure_plot(self.lat_line_current, color="cyan", label="current")
    self._enable_legend(self.chart_latacc)
    self.chart_latacc.x_axis.label = "time (s)"
    self.chart_latacc.y_axis.label = "latacc (m/s²)"
    self.plotter.add_chart(self.chart_latacc)

    self.chart_steer = pv.Chart2D(size=(0.34, 0.22), loc=(0.02, 0.40))
    self.steer_line = self.chart_steer.plot([0], [0])
    self._configure_plot(self.steer_line, color="orange", label="steer")
    self.chart_steer.y_axis.label = "steer (rad)"
    self.plotter.add_chart(self.chart_steer)

  def _load_controller(self):
    module = importlib.import_module(f"controllers.{self.controller_name}")
    self.controller = module.Controller()

  def _build_road(self):
    self.road = RoadGeometry(self.current_scenario)
    self.car_s = 0.0
    self.vehicle = VehicleKinematics()
    init_state = State(roll_lataccel=0.0, v_ego=self.speed_mps, a_ego=0.0)
    self.lat_engine.reset(init_state)
    self.sim_time = 0.0
    self.step_count = 0
    self.target_history.clear()
    self.current_history.clear()
    self.steer_history.clear()
    self.roll_history.clear()
    self.yaw_history.clear()
    self.future_plan_cache.clear()
    self._refresh_track_meshes()
    self._update_text()
    self._update_car_actor()
    self._update_future_actor()
    self._update_camera()

  def _refresh_track_meshes(self):
    for actor in self.track_actors:
      self.plotter.remove_actor(actor, render=False)
    self.track_actors.clear()
    center_pts = self.road.polyline()
    if len(center_pts) < 2:
      return
    center_mesh = pv.lines_from_points(center_pts, close=False)
    center_actor = self.plotter.add_mesh(center_mesh, color=LIGHT_GRAY, line_width=2)
    normals = np.array([[-math.sin(p.heading), math.cos(p.heading), 0.0] for p in self.road.samples], dtype=float)
    offsets = normals * (LANE_WIDTH / 2.0)
    left_mesh = pv.lines_from_points(center_pts + offsets, close=False)
    right_mesh = pv.lines_from_points(center_pts - offsets, close=False)
    left_actor = self.plotter.add_mesh(left_mesh, color=GRAY, line_width=1)
    right_actor = self.plotter.add_mesh(right_mesh, color=GRAY, line_width=1)
    self.track_actors.extend([center_actor, left_actor, right_actor])
    for actor in self.track_actors:
      actor.SetVisibility(self.show_track)

  def _register_key_events(self):
    self.plotter.add_key_event("space", self._toggle_manual_mode)
    self.plotter.add_key_event("a", lambda: self._nudge_manual(-0.05))
    self.plotter.add_key_event("d", lambda: self._nudge_manual(0.05))
    self.plotter.add_key_event("w", lambda: self._decay_manual(0.97))
    self.plotter.add_key_event("s", lambda: self._set_manual(0.0))
    self.plotter.add_key_event("minus", lambda: self._adjust_speed(-1.0))
    self.plotter.add_key_event("equal", lambda: self._adjust_speed(1.0))
    self.plotter.add_key_event("r", self._build_road)
    self.plotter.add_key_event("t", self._toggle_track)
    self.plotter.add_key_event("f", self._toggle_future)
    self.plotter.add_key_event("g", self._toggle_graphs)
    self.plotter.add_key_event("Tab", self._toggle_overlays)
    self.plotter.add_key_event("l", self._toggle_logging)
    for idx in range(min(9, len(self.controllers))):
      self.plotter.add_key_event(str(idx + 1), lambda idx=idx: self._switch_controller(idx))
    for scen_idx in range(min(4, len(self.scenarios))):
      self.plotter.add_key_event(f"F{scen_idx + 1}", lambda scen_idx=scen_idx: self._switch_scenario(scen_idx))
    self.plotter.add_key_event("F5", self._toggle_tire_failure)
    self.plotter.add_key_event("F6", self._toggle_noise)
    self.plotter.add_key_event("F7", self._toggle_delay)
    self.plotter.add_key_event("F8", self._toggle_clip)

  def _switch_controller(self, idx: int):
    if idx == self.controller_idx or idx >= len(self.controllers):
      return
    self.controller_idx = idx
    self._load_controller()
    self._build_road()

  def _switch_scenario(self, idx: int):
    if idx == self.scenario_idx or idx >= len(self.scenarios):
      return
    self.scenario_idx = idx
    self.speed_mps = self.current_scenario.nominal_speed
    self._build_road()

  def _toggle_manual_mode(self):
    self.manual_mode = not self.manual_mode
    if not self.manual_mode:
      self.manual_command = 0.0
    self._update_car_actor()
    self._update_text()

  def _nudge_manual(self, delta: float):
    self.manual_command = float(np.clip(self.manual_command + delta, STEER_RANGE[0], STEER_RANGE[1]))

  def _decay_manual(self, factor: float):
    self.manual_command *= factor

  def _set_manual(self, value: float):
    self.manual_command = float(np.clip(value, STEER_RANGE[0], STEER_RANGE[1]))

  def _adjust_speed(self, delta: float):
    self.speed_mps = float(np.clip(self.speed_mps + delta, 5.0, 45.0))
    self._update_text()

  def _toggle_track(self):
    self.show_track = not self.show_track
    for actor in self.track_actors:
      actor.SetVisibility(self.show_track)

  def _toggle_future(self):
    self.show_future = not self.show_future
    if self.future_actor:
      self.future_actor.SetVisibility(self.show_future)

  def _toggle_graphs(self):
    self.show_graphs = not self.show_graphs
    self.chart_latacc.visible = self.show_graphs
    self.chart_steer.visible = self.show_graphs

  def _toggle_overlays(self):
    self.show_overlays = not self.show_overlays
    self.status_actor.SetVisibility(self.show_overlays)
    self.help_actor.SetVisibility(self.show_overlays)

  def _toggle_logging(self):
    active = self.logger.toggle(self.current_scenario, self.controller_name)
    print(f"Logging {'enabled' if active else 'disabled'} -> logs/")

  def _toggle_tire_failure(self):
    self.faults.tire_failure = not self.faults.tire_failure

  def _toggle_noise(self):
    self.faults.lat_noise = not self.faults.lat_noise

  def _toggle_delay(self):
    if self.faults.steer_delay_steps > 0:
      self.faults.set_delay(0.0)
    else:
      self.faults.set_delay(0.1)

  def _toggle_clip(self):
    self.faults.steer_clip = not self.faults.steer_clip

  def _build_future_plan(self, s: float, v_ego: float) -> FuturePlan:
    latacc = []
    roll = []
    v_list = []
    a_list = []
    ds = max(v_ego, 1e-3) * DEL_T
    polyline = []
    for i in range(FUTURE_PLAN_STEPS):
      sample = self.road.sample_at_s(s + i * ds)
      latacc.append(v_ego * v_ego * sample.curvature)
      roll.append(ACC_G * math.sin(sample.bank))
      v_list.append(v_ego)
      a_list.append(0.0)
      polyline.append((sample.x, sample.y))
    self.future_plan_cache = polyline
    return FuturePlan(lataccel=latacc, roll_lataccel=roll, v_ego=v_list, a_ego=a_list)

  def _on_timer(self):
    self.update()

  def update(self):
    sample = self.road.sample_at_s(self.car_s)
    road_roll = ACC_G * math.sin(sample.bank)
    future_plan = self._build_future_plan(self.car_s, self.speed_mps)
    target_latacc = future_plan.lataccel[0] if future_plan.lataccel else 0.0
    current_latacc = self.lat_engine.current_latacc

    if self.manual_mode:
      steer_cmd = self.manual_command
    else:
      state = State(roll_lataccel=road_roll, v_ego=self.speed_mps, a_ego=0.0)
      steer_cmd = self.controller.update(target_latacc, current_latacc, state, future_plan)
    steer_cmd = float(np.clip(steer_cmd, STEER_RANGE[0], STEER_RANGE[1]))

    latacc, applied = self.lat_engine.step(steer_cmd, road_roll, self.speed_mps, 0.0, self.faults)
    if hasattr(self.controller, "observe_applied_action"):
      self.controller.observe_applied_action(applied)
    self.vehicle.integrate(latacc + road_roll, self.speed_mps, DEL_T)
    self.car_s += self.speed_mps * DEL_T
    self.sim_time += DEL_T
    self.step_count += 1

    self.target_history.append(target_latacc)
    self.current_history.append(latacc)
    self.steer_history.append(applied)
    self.roll_history.append(road_roll)
    self.yaw_history.append(self.vehicle.pose.yaw_rate)

    self.logger.log(
      self.current_scenario,
      self.controller_name,
      self.sim_time,
      self.step_count,
      target_latacc,
      latacc,
      road_roll,
      self.speed_mps,
      self.vehicle.pose.yaw_rate,
      steer_cmd,
      applied,
      self.vehicle.pose,
    )

    self._update_car_actor()
    self._update_future_actor()
    self._update_camera()
    self._update_text()
    if self.show_graphs:
      self._update_charts()

  def _update_car_actor(self):
    points = self._car_polygon_points()
    poly = pv.PolyData(points).delaunay_2d()
    color = ORANGE if self.manual_mode else BLUE
    if self.car_actor:
      self.plotter.remove_actor(self.car_actor, render=False)
    self.car_actor = self.plotter.add_mesh(poly, color=color, ambient=0.5, specular=0.1)

  def _car_polygon_points(self) -> np.ndarray:
    length = 4.5
    width = 1.9
    base = np.array([
      [length / 2.0, 0.0, 0.4],
      [-length / 2.0, width / 2.0, 0.4],
      [-length / 2.0, -width / 2.0, 0.4],
    ])
    heading = self.vehicle.pose.heading
    rot = np.array([
      [math.cos(heading), -math.sin(heading), 0.0],
      [math.sin(heading), math.cos(heading), 0.0],
      [0.0, 0.0, 1.0],
    ])
    points = (rot @ base.T).T
    points[:, 0] += self.vehicle.pose.x
    points[:, 1] += self.vehicle.pose.y
    return points

  def _update_future_actor(self):
    if self.future_actor:
      self.plotter.remove_actor(self.future_actor, render=False)
      self.future_actor = None
    if not self.future_plan_cache or not self.show_future:
      return
    pts = np.array([[x, y, 0.1] for (x, y) in self.future_plan_cache], dtype=float)
    self.future_actor = self.plotter.add_points(
      pts,
      color=self.current_scenario.color,
      render_points_as_spheres=True,
      point_size=12,
    )

  def _update_camera(self):
    pos = (self.vehicle.pose.x, self.vehicle.pose.y, 120.0)
    focal = (self.vehicle.pose.x, self.vehicle.pose.y, 0.0)
    self.plotter.camera_position = [pos, focal, (0.0, 1.0, 0.0)]

  def _update_text(self):
    if not self.show_overlays:
      return
    latacc = self.current_history[-1] if self.current_history else 0.0
    target = self.target_history[-1] if self.target_history else 0.0
    roll = self.roll_history[-1] if self.roll_history else 0.0
    yaw = self.yaw_history[-1] if self.yaw_history else 0.0
    steer = self.steer_history[-1] if self.steer_history else 0.0
    lines = [
      f"Scenario: {self.current_scenario.name}",
      f"Controller: {self.controller_name}",
      f"Mode: {'MANUAL' if self.manual_mode else 'AUTO'}   Speed: {self.speed_mps:.1f} m/s",
      f"LatAcc target/current: {target:+.3f} / {latacc:+.3f} m/s²",
      f"Road roll: {roll:+.3f} m/s² | Yaw rate: {yaw:+.3f} rad/s",
      f"Steer applied: {steer:+.3f} rad",
      f"Faults: {self.faults.description()}",
      f"Logging: {'ON' if self.logger.writer else 'OFF'}",
    ]
    self._set_text(self.status_actor, "\n".join(lines), self.status_corner_idx)

  @staticmethod
  def _set_text(actor, text: str, corner_idx: int):
    if hasattr(actor, "SetInput"):
      actor.SetInput(text)
    else:
      actor.SetText(corner_idx, text)

  @staticmethod
  def _configure_plot(line, color: str | None, label: str | None):
    if color is not None:
      try:
        line.color = color
      except AttributeError:
        pen = getattr(line, "pen", None)
        if pen is not None and hasattr(pen, "color"):
          pen.color = color
          line.pen = pen
    if label is not None and hasattr(line, "label"):
      line.label = label

  @staticmethod
  def _enable_legend(chart):
    legend = getattr(chart, "legend", None)
    if legend is not None and hasattr(legend, "visible"):
      legend.visible = True
      return
    show = getattr(chart, "show_legend", None)
    if callable(show):
      try:
        show(True)
      except Exception:
        pass

  def _update_charts(self):
    if len(self.target_history) >= 2:
      n = len(self.target_history)
      times = np.linspace(-(n - 1) * DEL_T, 0.0, n)
      self.lat_line_target.update(times, list(self.target_history))
      self.lat_line_current.update(times, list(self.current_history))
    if len(self.steer_history) >= 2:
      n = len(self.steer_history)
      times = np.linspace(-(n - 1) * DEL_T, 0.0, n)
      self.steer_line.update(times, list(self.steer_history))

  def run(self):
    self.plotter.show(auto_close=False)
    self.logger.close()


def get_available_controllers() -> List[str]:
  return sorted(
    f.stem
    for f in Path("controllers").iterdir()
    if f.is_file() and f.suffix == ".py" and f.stem != "__init__"
  )


def main():
  parser = argparse.ArgumentParser(description="TinyPhysics interactive simulator (PyVista)")
  parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx")
  args = parser.parse_args()

  controllers = get_available_controllers()
  print(f"Controllers: {controllers}")
  sim = InteractiveSimulator(args.model_path, controllers)
  sim.run()


if __name__ == "__main__":
  main()
