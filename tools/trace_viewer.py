import argparse
import itertools
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, SpanSelector, TextBox
import numpy as np

# Allow imports from repository root when the script is executed via `python tools/trace_viewer.py ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.append(str(REPO_ROOT))

from trace_loader import TraceRecord, load_trace

CONTROL_LINE_STYLE = {"color": "black", "linestyle": "--", "alpha": 0.4}


class TraceViewer:
  def __init__(self, traces: List[TraceRecord], screenshot_dir: Path, bookmark_output: Path):
    self.traces = traces
    self.screenshot_dir = screenshot_dir
    self.bookmark_output = bookmark_output
    self.bookmarks = []
    self.playing = False

    self.min_step = min(t.table["step"].min() for t in self.traces)
    self.max_step = min(t.table["step"].max() for t in self.traces)
    self.fig, self.axes = plt.subplots(4, sharex=True, figsize=(14, 11))
    plt.subplots_adjust(left=0.08, bottom=0.25)

    self.color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    self.trace_colors = {t.name: next(self.color_cycle) for t in self.traces}

    self._init_lines()
    self._init_controls()

    self.span_selector = SpanSelector(
      self.axes[0], self._on_span_select, "horizontal", useblit=True, props=dict(alpha=0.2, facecolor="tab:blue")
    )

    self._update_cursor(self.min_step)
    self._update_stats(self.min_step)

  def _init_lines(self) -> None:
    self.cursor_lines = []
    self.control_lines = []
    self.trace_lines = {"target": [], "current": [], "steer": [], "roll": [], "speed": []}

    for trace in self.traces:
      color = self.trace_colors[trace.name]
      table = trace.table
      self.trace_lines["target"].append(
        self.axes[0].plot(table["step"], table["target_lataccel"], label=f"{trace.name} target", color=color, linestyle="--")[0]
      )
      self.trace_lines["current"].append(
        self.axes[0].plot(table["step"], table["current_lataccel"], label=f"{trace.name} actual", color=color)[0]
      )
      self.trace_lines["steer"].append(
        self.axes[1].plot(table["step"], table["steer_command"], label=f"{trace.name} steer", color=color)[0]
      )
      self.trace_lines["roll"].append(
        self.axes[2].plot(table["step"], table["roll_lataccel"], label=f"{trace.name} roll", color=color, alpha=0.8)[0]
      )
      self.trace_lines["speed"].append(
        self.axes[3].plot(table["step"], table["v_ego"], label=f"{trace.name} v_ego", color=color, alpha=0.8)[0]
      )

    for ax, label in zip(self.axes, ["Lateral Acceleration", "Steering Command", "Roll Lateral Accel", "v_ego"]):
      ax.set_ylabel(label)
      self.control_lines.append(ax.axvline(self.traces[0].table["control_start_idx"].iloc[0], **CONTROL_LINE_STYLE))
      self.cursor_lines.append(ax.axvline(0, color="red", linestyle=":", alpha=0.7))
    self.axes[-1].set_xlabel("Step")

    self.axes[0].legend(loc="upper right")
    self.axes[1].legend(loc="upper right")
    self.axes[2].legend(loc="upper right")
    self.axes[3].legend(loc="upper right")

  def _init_controls(self) -> None:
    ax_slider = plt.axes([0.1, 0.14, 0.65, 0.03])
    self.step_slider = Slider(ax_slider, "Step", self.min_step, self.max_step, valinit=self.min_step, valfmt="%0.0f")
    self.step_slider.on_changed(lambda _: self._on_step_change())

    ax_speed = plt.axes([0.1, 0.09, 0.25, 0.03])
    self.speed_slider = Slider(ax_speed, "Speed", 0.1, 4.0, valinit=1.0, valfmt="%0.1fx")

    ax_play = plt.axes([0.4, 0.09, 0.08, 0.04])
    self.play_button = Button(ax_play, "Play/Pause")
    self.play_button.on_clicked(lambda _event: self._toggle_play())

    ax_prev = plt.axes([0.5, 0.09, 0.06, 0.04])
    self.prev_button = Button(ax_prev, "< Prev")
    self.prev_button.on_clicked(lambda _event: self._nudge_step(-1))

    ax_next = plt.axes([0.57, 0.09, 0.06, 0.04])
    self.next_button = Button(ax_next, "Next >")
    self.next_button.on_clicked(lambda _event: self._nudge_step(1))

    ax_note = plt.axes([0.1, 0.04, 0.25, 0.04])
    self.note_box = TextBox(ax_note, "Bookmark note")

    ax_bookmark = plt.axes([0.37, 0.04, 0.1, 0.04])
    self.bookmark_button = Button(ax_bookmark, "Bookmark")
    self.bookmark_button.on_clicked(lambda _event: self._bookmark_current())

    ax_export = plt.axes([0.48, 0.04, 0.12, 0.04])
    self.export_button = Button(ax_export, "Export Notes")
    self.export_button.on_clicked(lambda _event: self._export_bookmarks())

    ax_save = plt.axes([0.62, 0.04, 0.12, 0.04])
    self.save_button = Button(ax_save, "Save Frame")
    self.save_button.on_clicked(lambda _event: self._save_screenshot())

    ax_check = plt.axes([0.78, 0.04, 0.18, 0.12])
    labels = [trace.name for trace in self.traces]
    visibility = [True for _ in self.traces]
    self.check_buttons = CheckButtons(ax_check, labels=labels, actives=visibility)
    self.check_buttons.on_clicked(self._toggle_trace)

    self.stats_box = self.fig.text(0.82, 0.9, "", bbox=dict(facecolor="white", alpha=0.8))

    self.preview_ax = self.fig.add_axes([0.78, 0.58, 0.18, 0.25])
    self.preview_ax.set_title("Future Plan")

    self.timer = self.fig.canvas.new_timer(interval=int(self.traces[0].del_t * 1000))
    self.timer.add_callback(self._advance_frame)
    self.timer.start()

  def _toggle_trace(self, label: str) -> None:
    for key, lines in self.trace_lines.items():
      for line in lines:
        if line.get_label().startswith(label):
          line.set_visible(not line.get_visible())
    self.fig.canvas.draw_idle()

  def _on_span_select(self, x_min: float, x_max: float) -> None:
    self.axes[0].set_xlim(x_min, x_max)
    self.axes[1].set_xlim(x_min, x_max)
    self.axes[2].set_xlim(x_min, x_max)
    self.axes[3].set_xlim(x_min, x_max)
    self.fig.canvas.draw_idle()

  def _on_step_change(self) -> None:
    step = int(self.step_slider.val)
    self._update_cursor(step)
    self._update_stats(step)

  def _nudge_step(self, delta: int) -> None:
    next_step = np.clip(self.step_slider.val + delta, self.min_step, self.max_step)
    self.step_slider.set_val(next_step)

  def _toggle_play(self) -> None:
    self.playing = not self.playing

  def _step_to_position(self, trace: TraceRecord, step: int) -> int:
    return int(np.clip(np.searchsorted(trace.table["step"].values, step), 0, len(trace.table) - 1))

  def _advance_frame(self) -> None:
    if not self.playing:
      return
    next_step = self.step_slider.val + self.speed_slider.val
    if next_step >= self.max_step:
      self.playing = False
      next_step = self.max_step
    self.step_slider.set_val(next_step)

  def _update_cursor(self, step: int) -> None:
    for cursor in self.cursor_lines:
      cursor.set_xdata([step, step])
    self.fig.canvas.draw_idle()

  def _update_stats(self, step: int) -> None:
    lines = []
    visible_trace = None
    for trace in self.traces:
      target = np.interp(step, trace.table["step"], trace.table["target_lataccel"])
      current = np.interp(step, trace.table["step"], trace.table["current_lataccel"])
      steer = np.interp(step, trace.table["step"], trace.table["steer_command"])
      roll = np.interp(step, trace.table["step"], trace.table["roll_lataccel"])
      v_ego = np.interp(step, trace.table["step"], trace.table["v_ego"])
      jerk_series = trace.compute_jerk()
      idx = self._step_to_position(trace, step)
      jerk_val = jerk_series[min(idx, len(jerk_series) - 1)]
      lat_delta = 0.0
      if idx > 0:
        prev_val = trace.table["current_lataccel"].iloc[idx - 1]
        lat_delta = current - prev_val
      steer_clip = steer <= trace.steer_range[0] or steer >= trace.steer_range[1]
      lat_clip = abs(lat_delta) >= trace.max_acc_delta
      lines.append(
        f"{trace.name}: step {int(step)}, err={target - current:+.3f}, jerk={jerk_val:+.3f}, "
        f"steer={steer:+.2f}{' (clip)' if steer_clip else ''}, latΔ={lat_delta:+.2f}{' (clip)' if lat_clip else ''}, "
        f"v_ego={v_ego:.2f}, roll={roll:+.2f}"
      )
      if visible_trace is None and any(line.get_visible() for line in self.trace_lines["current"] if line.get_label().startswith(trace.name)):
        visible_trace = trace

    self.stats_box.set_text("\n".join(lines))
    if visible_trace:
      self._update_future_preview(visible_trace, int(step))

  def _update_future_preview(self, trace: TraceRecord, step: int) -> None:
    self.preview_ax.clear()
    self.preview_ax.set_title(f"Future Plan ({trace.name})")
    plan = trace.get_future_plan(step)
    if not plan:
      self.preview_ax.text(0.5, 0.5, "No plan", ha="center")
      self.fig.canvas.draw_idle()
      return
    horizon = list(range(len(plan.lataccel)))
    self.preview_ax.plot(horizon, plan.lataccel, label="planned lataccel")
    start_idx = self._step_to_position(trace, step)
    realized_window = trace.table["current_lataccel"].iloc[start_idx : start_idx + len(horizon)]
    if len(realized_window) > 0:
      self.preview_ax.plot(range(len(realized_window)), realized_window, label="actual")
    self.preview_ax.legend()
    self.fig.canvas.draw_idle()

  def _bookmark_current(self) -> None:
    step = int(self.step_slider.val)
    note = self.note_box.text
    for trace in self.traces:
      self.bookmarks.append({"trace": trace.name, "step": step, "note": note})
    print(f"Bookmarked step {step} with note '{note}' for all loaded traces")

  def _export_bookmarks(self) -> None:
    if not self.bookmarks:
      print("No bookmarks to export")
      return
    self.bookmark_output.parent.mkdir(parents=True, exist_ok=True)
    with self.bookmark_output.open("w") as fh:
      fh.write("trace,step,note\n")
      for bm in self.bookmarks:
        fh.write(f"{bm['trace']},{bm['step']},\"{bm['note']}\"\n")
    print(f"Exported {len(self.bookmarks)} bookmarks to {self.bookmark_output}")

  def _save_screenshot(self) -> None:
    self.screenshot_dir.mkdir(parents=True, exist_ok=True)
    step = int(self.step_slider.val)
    path = self.screenshot_dir / f"trace_step_{step}.png"
    self.fig.savefig(path)
    print(f"Saved screenshot to {path}")



def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Interactive viewer for TinyPhysicsSimulator traces")
  parser.add_argument("traces", nargs="+", type=Path, help="Trace CSV files to load")
  parser.add_argument("--future-path", action="append", default=None, help="Optional sidecar JSONL paths aligned with trace order")
  parser.add_argument("--screenshot-dir", type=Path, default=Path("trace_viewer_snaps"))
  parser.add_argument("--bookmarks-out", type=Path, default=Path("trace_bookmarks.csv"))
  return parser.parse_args()


def main():
  args = parse_args()
  future_paths: Optional[List[Optional[Path]]]
  if args.future_path:
    if len(args.future_path) != len(args.traces):
      raise SystemExit("Provide one --future-path for each trace or omit entirely")
    future_paths = [Path(p) if p else None for p in args.future_path]
  else:
    future_paths = [None] * len(args.traces)

  traces = [load_trace(path, future_sidecar=fp) for path, fp in zip(args.traces, future_paths)]
  viewer = TraceViewer(traces, screenshot_dir=args.screenshot_dir, bookmark_output=args.bookmarks_out)
  plt.show()


if __name__ == "__main__":
  main()
