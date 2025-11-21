import math
import unittest

import numpy as np

from interactive_sim import (
  LatAccelEngine,
  FaultToggles,
  RoadGeometry,
  RoadSegment,
  Scenario,
  State,
  build_scenarios,
)
from interactive_sim import InteractiveSimulator  # noqa: E402
from tinyphysics import ACC_G, CONTEXT_LENGTH


class DummyModel:
  """Minimal TinyPhysicsModel stub returning the last applied steer."""

  def get_current_lataccel(self, sim_states, actions, past_preds):
    assert len(sim_states) == CONTEXT_LENGTH
    assert len(actions) == CONTEXT_LENGTH
    assert len(past_preds) == CONTEXT_LENGTH
    return actions[-1]


class ScenarioTests(unittest.TestCase):
  def test_scenarios_have_positive_length(self):
    scenarios = build_scenarios()
    self.assertGreaterEqual(len(scenarios), 3)
    for scenario in scenarios:
      total_length = sum(seg.length for seg in scenario.segments)
      self.assertGreater(total_length, 0.0, scenario.name)

  def test_road_geometry_sampling_wraps(self):
    scenario = Scenario(
      name="mini",
      segments=[
        RoadSegment(50.0, 0.0, math.radians(2.0)),
        RoadSegment(30.0, 0.01, math.radians(-1.0)),
      ],
      nominal_speed=15.0,
      description="unit test",
    )
    road = RoadGeometry(scenario, step=5.0)
    sample0 = road.sample_at_s(0.0)
    sample_wrap = road.sample_at_s(road.total_length * 2.0)
    self.assertAlmostEqual(sample0.x, sample_wrap.x, places=5)
    self.assertAlmostEqual(sample0.y, sample_wrap.y, places=5)


class FaultToggleTests(unittest.TestCase):
  def test_delay_pipeline(self):
    faults = FaultToggles(dt=0.1)
    faults.set_delay(0.2)  # two steps
    outputs = []
    for cmd in [0.1, 0.3, 0.5]:
      outputs.append(faults.apply_to_command(cmd))
    self.assertEqual(outputs[0], 0.0)
    self.assertEqual(outputs[1], 0.0)
    self.assertEqual(outputs[2], 0.1)

  def test_tire_clip(self):
    faults = FaultToggles(dt=0.1)
    faults.tire_failure = True
    clipped = faults.apply_to_latacc(ACC_G)
    self.assertAlmostEqual(clipped, 0.3 * ACC_G, places=5)


class LatAccelEngineTests(unittest.TestCase):
  def setUp(self):
    init_state = State(roll_lataccel=0.0, v_ego=10.0, a_ego=0.0)
    self.engine = LatAccelEngine(DummyModel(), init_state)
    self.faults = FaultToggles(dt=0.1)

  def test_step_clamps_delta(self):
    latacc, applied = self.engine.step(0.5, 0.0, 10.0, 0.0, self.faults)
    self.assertAlmostEqual(latacc, 0.5)
    self.assertAlmostEqual(applied, 0.5)

    latacc2, applied2 = self.engine.step(2.0, 0.0, 10.0, 0.0, self.faults)
    self.assertAlmostEqual(latacc2, 1.0)  # previous 0.5 + MAX_ACC_DELTA (0.5)
    self.assertAlmostEqual(applied2, 2.0)


class HelperTests(unittest.TestCase):
  def test_set_text_prefers_setinput(self):
    class WithInput:
      def __init__(self):
        self.value = ""

      def SetInput(self, text):
        self.value = text

    actor = WithInput()
    InteractiveSimulator._set_text(actor, "hello", 2)
    self.assertEqual(actor.value, "hello")

  def test_set_text_falls_back_to_corner(self):
    class Corner:
      def __init__(self):
        self.slots = {}

      def SetText(self, idx, text):
        self.slots[idx] = text

    actor = Corner()
    InteractiveSimulator._set_text(actor, "world", 1)
    self.assertEqual(actor.slots[1], "world")

  def test_configure_plot_sets_color_and_label(self):
    class Line:
      def __init__(self):
        self.color = None
        self.label = None

    line = Line()
    InteractiveSimulator._configure_plot(line, color="lime", label="target")
    self.assertEqual(line.color, "lime")
    self.assertEqual(line.label, "target")

  def test_enable_legend_handles_old_versions(self):
    class Chart:
      def show_legend(self, flag):
        self.flag = flag

    chart = Chart()
    InteractiveSimulator._enable_legend(chart)
    self.assertTrue(getattr(chart, "flag", True))



if __name__ == "__main__":
  unittest.main()
