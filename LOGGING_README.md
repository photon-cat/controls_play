# Controller Run Logging and Visualization

This directory includes tools to log controller runs and visualize them later, similar to `--debug` mode but with persistent storage.

## Quick Start

### 1. Run a controller with logging

```bash
python tinyphysics_logging.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./tuning_data/tuning_scenario_01.csv \
  --controller pid \
  --log_dir logs
```

This will:
- Run the simulation
- Save all data to `logs/pid_test_scenario_TIMESTAMP.npz`
- Print the results and visualization command

### 2. Visualize the logged run

```bash
# Show interactive plot
python visualize_run.py --log_file logs/pid_test_scenario_20251205_195742.npz

# Save plot to file
python visualize_run.py --log_file logs/pid_test_scenario_20251205_195742.npz --save plots/my_run.png

# Just print summary (no plot)
python visualize_run.py --log_file logs/pid_test_scenario_20251205_195742.npz --no_plot
```

## How It Works

### Debug Mode (Original)

The `--debug` flag in `tinyphysics.py` creates **real-time** plots:

```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
                      --data_path ./data/00000.csv \
                      --controller pid \
                      --debug
```

**How it's generated** ([tinyphysics.py:193-209](tinyphysics.py#L193-L209)):
1. Creates matplotlib figure with 4 subplots
2. Every 10 steps during rollout, updates plots with:
   - Lateral acceleration (target vs actual)
   - Steering commands
   - Road roll lateral acceleration
   - Vehicle velocity
3. Uses `plt.ion()` for interactive mode and `plt.pause()` for updates
4. Shows final plot with `plt.show()`

**Limitations:**
- No persistent storage (data lost when window closes)
- Can't review runs later
- Can't compare multiple runs easily

### Logging System (New)

The new system saves all data to `.npz` files for later analysis.

**Logged Data:**
- `target_lataccel_history`: Desired lateral acceleration trajectory
- `current_lataccel_history`: Actual lateral acceleration achieved
- `action_history`: Steering commands from controller
- `state_history`: Vehicle states (roll_lataccel, v_ego, a_ego)
- `costs`: Dictionary with lataccel_cost, jerk_cost, total_cost
- `metadata`: Controller name, data path, timestamp, model path

**File Format:** NumPy `.npz` (compressed numpy arrays)

## Use Cases

### Compare Controllers

```bash
# Run multiple controllers on same scenario
python tinyphysics_logging.py --data_path ./data/test_scenario.csv --controller pid --log_dir logs
python tinyphysics_logging.py --data_path ./data/test_scenario.csv --controller zero --log_dir logs

# Visualize side by side
python visualize_run.py --log_file logs/pid_test_scenario_*.npz --save plots/pid.png
python visualize_run.py --log_file logs/zero_test_scenario_*.npz --save plots/zero.png
```

### Debug Controller Development

```bash
# Make changes to controller
# Run and log
python tinyphysics_logging.py --data_path ./data/test_scenario.csv --controller my_controller --log_dir logs

# Review plots multiple times without re-running simulation
python visualize_run.py --log_file logs/my_controller_test_scenario_*.npz
```

### Batch Testing

```bash
# Run on multiple scenarios
for data_file in data/00*.csv; do
    python tinyphysics_logging.py --model_path ./models/tinyphysics.onnx \
                                  --data_path $data_file \
                                  --controller pid \
                                  --log_dir logs/batch_test
done

# Review any problematic runs
python visualize_run.py --log_file logs/batch_test/pid_00042_*.npz
```

## Visualization Details

The `visualize_run.py` script generates **identical plots** to `--debug` mode:

1. **Lateral Acceleration**: Target vs Actual
   - Shows tracking performance
   - Vertical line at step 100 (control start)

2. **Steering Commands**: Controller output
   - Shows control effort
   - Horizontal line at 0

3. **Road Roll**: Lateral accel from road banking
   - Environmental factor
   - Affects required steering

4. **Velocity**: Vehicle speed
   - Context for understanding dynamics
   - Higher speed = more sensitive steering response

All plots include:
- Grid for easy reading
- Legend with labels
- Control start line (step 100)
- Cost information in title

## Tips

- **Log directory**: Automatically created if it doesn't exist
- **Filename format**: `{controller}_{scenario}_{timestamp}.npz`
- **File size**: ~100-200 KB per run (500 timesteps)
- **No logging**: Use `--no_log` flag to disable
- **Conda environment**: Make sure to use `conda run -n ai python` if needed

## Example Workflow

```bash
# 1. Create test scenario
python generate_test_scenario.py

# 2. Run your controller with logging
python tinyphysics_logging.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data/test_scenario.csv \
  --controller my_controller \
  --log_dir logs

# 3. Review the plots
python visualize_run.py --log_file logs/my_controller_test_scenario_*.npz

# 4. Save for documentation
python visualize_run.py \
  --log_file logs/my_controller_test_scenario_*.npz \
  --save reports/my_controller_performance.png
```
