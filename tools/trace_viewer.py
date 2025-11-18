"""Interactive viewer for TinyPhysics trace logs.

Features:
- Load one or more trace CSVs (plus optional sidecar) produced by ``tinyphysics.py --log-trace``.
- Compare controller revisions via toggleable overlays.
- Playback controls (play/pause, step forward/back, speed slider).
- Cursor annotations showing step number, error, jerk, and clipping flags.
- Future-plan preview panel to inspect upcoming targets vs. realized motion.
- Bookmark interesting steps and export screenshots with notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

from trace_loader import TraceDataset, load_trace
from tinyphysics import CONTROL_START_IDX, DEL_T, MAX_ACC_DELTA, STEER_RANGE


class TraceViewer:
  def __init__(self, traces: Dict[str, TraceDataset], preview_trace: str):
    self.traces = traces
    self.preview_trace = preview_trace if preview_trace in traces else next(iter(traces))
    self.max_steps = max(len(t.dataframe) - 1 for t in traces.values())
    self.running = False
    self.bookmarks: List[int] = []

    self.fig, self.axes = plt.subplots(4, sharex=True, figsize=(13, 9))
    self.fig.subplots_adjust(bottom=0.25, right=0.75)
    self.cursor_lines = [ax.axvline(0, color='k', linestyle='--', alpha=0.6) for ax in self.axes]
    for ax in self.axes:
      ax.axvline(CONTROL_START_IDX, color='gray', linestyle=':', alpha=0.6, label='CONTROL_START_IDX')

    self.future_ax = self.fig.add_axes([0.76, 0.55, 0.2, 0.35])
    self.future_ax.set_title('Future plan preview')

    slider_ax = self.fig.add_axes([0.1, 0.12, 0.6, 0.03])
    self.step_slider = Slider(slider_ax, 'Step', 0, self.max_steps, valinit=0, valfmt='%0.0f')
    self.step_slider.on_changed(self._on_step_change)

    speed_ax = self.fig.add_axes([0.1, 0.07, 0.3, 0.03])
    self.speed_slider = Slider(speed_ax, 'Speed (steps/tick)', 1, 10, valinit=1, valstep=1)

    play_ax = self.fig.add_axes([0.72, 0.12, 0.08, 0.04])
    self.play_btn = Button(play_ax, 'Play/Pause')
    self.play_btn.on_clicked(self._toggle_play)

    next_ax = self.fig.add_axes([0.82, 0.12, 0.05, 0.04])
    self.next_btn = Button(next_ax, 'Next')
    self.next_btn.on_clicked(lambda _: self._bump_step(1))

    prev_ax = self.fig.add_axes([0.65, 0.12, 0.05, 0.04])
    self.prev_btn = Button(prev_ax, 'Prev')
    self.prev_btn.on_clicked(lambda _: self._bump_step(-1))

    bookmark_ax = self.fig.add_axes([0.65, 0.07, 0.1, 0.04])
    self.bookmark_btn = Button(bookmark_ax, 'Bookmark')
    self.bookmark_btn.on_clicked(self._bookmark_step)

    save_ax = self.fig.add_axes([0.82, 0.07, 0.1, 0.04])
    self.save_btn = Button(save_ax, 'Save shot')
    self.save_btn.on_clicked(self._save_screenshot)

    note_ax = self.fig.add_axes([0.1, 0.02, 0.3, 0.03])
    self.note_box = TextBox(note_ax, 'Note', initial='')

    toggle_ax = self.fig.add_axes([0.82, 0.55, 0.12, 0.2])
    self.trace_names = list(traces.keys())
    self.toggle = CheckButtons(toggle_ax, labels=self.trace_names, actives=[True] * len(traces))
    self.toggle.on_clicked(self._toggle_trace)

    self.stat_text = self.fig.text(0.1, 0.18, '', fontsize=10)

    self.lines = self._init_plots()
    self.timer = self.fig.canvas.new_timer(interval=80)
    self.timer.add_callback(self._on_timer)
    self.timer.start()

    self._on_step_change(0)

  def _init_plots(self):
    lines = {name: {} for name in self.trace_names}
    for name, dataset in self.traces.items():
      ax_lat, ax_action, ax_roll, ax_speed = self.axes
      (line_target,) = ax_lat.plot(dataset.dataframe['step'], dataset.dataframe['target_lataccel'], label=f"{name} target")
      (line_curr,) = ax_lat.plot(dataset.dataframe['step'], dataset.dataframe['current_lataccel'], label=f"{name} current")

      (line_action,) = ax_action.plot(dataset.dataframe['step'], dataset.dataframe['steer_command'], label=f"{name} steer")
      ax_action.axhline(STEER_RANGE[0], color='red', linestyle='--', alpha=0.4)
      ax_action.axhline(STEER_RANGE[1], color='red', linestyle='--', alpha=0.4)

      (line_roll,) = ax_roll.plot(dataset.dataframe['step'], dataset.dataframe['roll_lataccel'], label=f"{name} roll_lataccel")
      (line_speed,) = ax_speed.plot(dataset.dataframe['step'], dataset.dataframe['v_ego'], label=f"{name} v_ego")

      lines[name] = {
        'target': line_target,
        'current': line_curr,
        'action': line_action,
        'roll': line_roll,
        'speed': line_speed,
      }

    self.axes[0].set_ylabel('Lataccel (m/s^2)')
    self.axes[1].set_ylabel('Steer cmd')
    self.axes[2].set_ylabel('Roll lataccel')
    self.axes[3].set_ylabel('v_ego (m/s)')
    self.axes[3].set_xlabel('Step')
    for ax in self.axes:
      ax.legend(loc='upper right')

    return lines

  def _toggle_trace(self, label):
    visible = not self.lines[label]['target'].get_visible()
    for key, line in self.lines[label].items():
      line.set_visible(visible)
    for ax in self.axes:
      ax.legend(loc='upper right')
    self.fig.canvas.draw_idle()

  def _bump_step(self, delta: int):
    new_step = int(np.clip(self.step_slider.val + delta, 0, self.max_steps))
    self.step_slider.set_val(new_step)

  def _toggle_play(self, _event):
    self.running = not self.running

  def _on_timer(self):
    if not self.running:
      return
    delta = int(self.speed_slider.val)
    self._bump_step(delta)

  def _bookmark_step(self, _event):
    step = int(self.step_slider.val)
    if step not in self.bookmarks:
      self.bookmarks.append(step)
      self.axes[0].axvline(step, color='orange', linestyle=':', alpha=0.5)
      self.fig.canvas.draw_idle()

  def _save_screenshot(self, _event):
    step = int(self.step_slider.val)
    note = self.note_box.text.strip().replace(' ', '_')
    stem = f"trace_step{step}"
    if note:
      stem += f"_{note}"
    path = Path(f"{stem}.png")
    self.fig.savefig(path, dpi=150)
    print(f"Saved {path}")

  def _on_step_change(self, value):
    step = int(value)
    for line in self.cursor_lines:
      line.set_xdata([step, step])

    stats_lines = []
    for name, dataset in self.traces.items():
      df = dataset.dataframe
      if step >= len(df):
        continue
      stats_lines.append(self._format_stats(name, df, step))
    self.stat_text.set_text('\n'.join(stats_lines))

    self._update_future_preview(step)
    self.fig.canvas.draw_idle()

  def _format_stats(self, name: str, df, step: int) -> str:
    row = df.iloc[step]
    prev_row = df.iloc[step - 1] if step > 0 else row
    error = row['target_lataccel'] - row['current_lataccel']
    jerk = (row['current_lataccel'] - prev_row['current_lataccel']) / DEL_T if step > 0 else 0.0
    steer_clipped = row['steer_command'] <= STEER_RANGE[0] + 1e-6 or row['steer_command'] >= STEER_RANGE[1] - 1e-6
    accel_clipped = abs(row['current_lataccel'] - prev_row['current_lataccel']) >= MAX_ACC_DELTA - 1e-6 if step > 0 else False
    return (
      f"{name} | step {step} | err={error:.3f} jerk={jerk:.3f} "
      f"steer={row['steer_command']:.3f}{' (clipped)' if steer_clipped else ''} "
      f"lataccel={row['current_lataccel']:.3f}{' (Δ clipped)' if accel_clipped else ''}"
    )

  def _update_future_preview(self, step: int):
    dataset = self.traces[self.preview_trace]
    df = dataset.dataframe
    self.future_ax.clear()
    self.future_ax.set_title(f"Future plan | {self.preview_trace}")
    if step >= len(dataset.steps):
      return

    plan = dataset.steps[step].future_plan
    if plan:
      horizon = list(range(step + 1, step + 1 + plan.window))
      self.future_ax.plot(horizon[:len(plan.lataccel)], plan.lataccel, label='planned lataccel')
    realized_span = df.iloc[step + 1: step + 1 + (plan.window if plan else 0)]
    if not realized_span.empty:
      self.future_ax.plot(realized_span['step'], realized_span['current_lataccel'], label='actual lataccel')
      self.future_ax.plot(realized_span['step'], realized_span['target_lataccel'], label='target lataccel', linestyle='--')
    self.future_ax.legend(loc='upper right')


def parse_args():
  parser = argparse.ArgumentParser(description='Interactive TinyPhysics trace viewer')
  parser.add_argument('traces', nargs='+', help='Trace CSV files to load')
  parser.add_argument('--preview-trace', help='Which trace name to use for the future preview panel (defaults to first)')
  parser.add_argument('--expected-window', type=int, help='Validate a specific future plan window length')
  return parser.parse_args()


def main():
  args = parse_args()
  datasets: Dict[str, TraceDataset] = {}
  for trace_path in args.traces:
    path = Path(trace_path)
    datasets[path.stem] = load_trace(path, expected_window=args.expected_window)

  viewer = TraceViewer(datasets, preview_trace=args.preview_trace or next(iter(datasets)))
  plt.show()


if __name__ == '__main__':
  main()
