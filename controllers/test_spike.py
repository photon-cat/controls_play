from . import BaseController


class Controller(BaseController):
  """
  A controller that outputs single-step spikes at intervals
  """
  def __init__(self):
    self.step = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.step += 1

    # Before ts 100: output 0
    if self.step < 100:
      return 0.0

    # From step 100 onwards: spike pattern
    # 100-119: 0, 120: 0.2, 121-140: 0, 141: 0.4, 142-161: 0, 162: 0.6, ...
    steps_after_100 = self.step - 100
    cycle = steps_after_100 // 21  # 20 zeros + 1 spike = 21 steps per cycle
    pos_in_cycle = steps_after_100 % 21

    # Position 20 in cycle is the spike
    if pos_in_cycle == 20:
      return 0.1 + cycle * 0.1
    else:
      return 0.0
