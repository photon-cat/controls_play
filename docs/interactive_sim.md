# TinyPhysics Interactive Simulator

This tool turns `tinyphysics.onnx` into a “lat-accel engine” and surrounds it
with a simple 2.5‑D world so you can iterate on controllers with immediate
visual feedback.

At each 100 ms step we:

1. Look up the road curvature and cross‑slope (bank) from the active scenario.
2. Convert curvature into a target lateral acceleration (`v²κ`).
3. Feed road roll (`g * sin(bank)`), the commanded steer, and the current ego
   speed into TinyPhysics to obtain the measured lateral acceleration.
4. Integrate `latacc → yaw_rate → heading → (x, y)` using a bicycle‑style model.
5. Render the track, the future plan, and controller telemetry with PyVista.

This doc covers how to run the simulator, what to expect on screen, and how to
use the built-in tools to stress controllers.

## Getting Started

```bash
pip install -r requirements.txt
python interactive_sim.py --model_path ./models/tinyphysics.onnx
```

The simulator will list every controller found inside `controllers/` and start
with whichever one is first in the list.

## World Model Highlights

- **Road geometry & bank** – A scenario is a list of `RoadSegment`s
  (length/curvature/bank). The geometry module integrates this into a smooth
  centerline with lane boundaries and exposes samples via arc-length `s`.
- **Cross slope → roll** – For each sample we compute
  `road_roll = g * sin(bank)` and feed it to the controller state and to the
  TinyPhysics input so roll alone can kick the car sideways.
- **Future plan** – The next five seconds of curvature and bank are converted
  into future lat-accel/roll sequences (`FuturePlan`). They are rendered as
  “ghost” dots ahead of the car and also passed through to controllers.
- **Kinematics** – Yaw rate is simply `latacc / v`. Heading and XY are
  integrated with this yaw rate using the nominal ego speed.

## PyVista Views

### Track view

- Centerline in light gray, lane boundaries on either side, and the car as a
  small triangle. The camera scrolls smoothly with the car:
  `screen_x = (x - car_x) * scale + width * 0.4`.
- Future plan points (cyan/orange/green/purple depending on scenario) show the
  lookahead path the controllers see.

### HUD overlays

- Scenario / controller / mode status
- Speed, roll acceleration, yaw rate, and active fault toggles
- Logging status

### Rolling plots (toggle with `G`)

Three small oscilloscope-style plots (each 20 seconds wide):

1. Target lateral acceleration
2. Measured lateral acceleration
3. Applied steering command

## Controls & Hotkeys

| Key / Input             | Action |
| ----------------------- | ------ |
| `SPACE`                 | Toggle manual vs auto controller |
| `A/D`                   | Steer left/right in manual |
| `W`                     | Decay toward zero steer |
| `S`                     | Zero steer |
| `1`…`9`                 | Swap controllers on the fly |
| `F1`…`F4`               | Load scenarios (Straight Crown, S-Curve, Highway, Adversarial) |
| `-` / `=`               | Decrease / increase nominal speed |
| `R`                     | Reset current scenario (car pose + integrator history) |
| `T`                     | Toggle track rendering |
| `G`                     | Toggle graphs |
| `F`                     | Toggle future plan markers |
| `TAB`                   | Toggle HUD overlays |
| `L`                     | Toggle CSV logging (`logs/sim_<scenario>_<controller>.csv`) |
| Close window / `Q`      | Quit |

> Note: PyVista does not expose joystick/gamepad devices directly, so manual override currently relies on the keyboard-only shortcuts above.

### Fault injection (runtime):

| Key      | Fault |
| -------- | ----- |
| `F5`     | Tire failure (lat accel limited to ±0.3 g) |
| `F6`     | Add lateral force noise |
| `F7`     | Toggle 100 ms steering actuator delay |
| `F8`     | Clip steering authority to ±0.35 rad |

## Scenarios

1. **Straight Crown** – Long straight road with 4.5° of positive crown to test
   integrator windup and roll compensation.
2. **S Curve** – Medium-speed left/right transition for transient-response
   tuning.
3. **Highway Sweep** – 32 m/s large-radius curve with 5° bank to exercise
   feed-forward and high-speed stability.
4. **Adversarial** – Procedurally generated curvature and bank noise to
   torture-test controllers.

Each scenario provides its own color for future-plan dots and HUD status.

## Logging & Telemetry

- Hit `L` to start/stop logging. Files land in `logs/` with per-frame entries
  containing time, scenario, controller, target/measured latacc, roll, yaw
  rate, commanded & applied steering, and pose.
- Graphs provide immediate visual feedback. Toggle them with `G` if you want a
  cleaner view when screen-sharing.

## Controller API

The simulator imports any module in `controllers/` (except `__init__`) and
expects a `Controller` class implementing:

```python
def update(self, target_lataccel, current_lataccel, state: State, future_plan: FuturePlan) -> float:
    ...
```

If your controller implements `observe_applied_action(self, steer)` it will be
called after steering authority, fault injection, and manual overrides have
been applied so you can keep internal rate limits synchronized.

## Robustness Tools

Use the fault toggles during runtime to reproduce the nasty cases that break
controllers:

- Tire saturation (`F5`)
- Steering delay (`F7`)
- Lat-force noise (`F6`)
- Manual steering clamp (`F8`)

Combine these with the High-Crown or Adversarial scenarios for a controller
torture chamber.

---

Have ideas for additional scenes or disturbance models? Drop them into
`build_scenarios()` – the geometry + rendering stack adapts automatically.
