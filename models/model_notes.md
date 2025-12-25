# Steer Model Training Notes

## Architecture
- **Type**: Transformer encoder
- **Hidden dim**: 128 (d_model)
- **Heads**: 4
- **Layers**: 2
- **Context**: 10 timesteps × 6 features
- **Input features**: [v_ego, a_ego, roll_lataccel, target_lataccel, steer_command, measured_lataccel]
- **Output**: steer_command prediction

## Models

### steer_model.pt
- First full training on all data
- No smoothness loss
- Basic noise injection

### steer_model_1.pt  
- 128 hidden, 512 batch size
- 20 epochs on full dataset
- No smoothness loss

### steer_model_noise_1.pt
- Added noise injection during training:
  - steer_command history: ±0.05 (simulates past prediction errors)
  - measured_lataccel: ±0.25 (simulates plant diverging from target)
- Helps with distribution shift at inference

### steer_model_smooth.pt
- noise=0.5, smooth=0.2
- Smoothness loss: penalizes change from previous steer
- `loss = MSE(pred, target) + 0.2 * MSE(pred, prev_steer)`

### steer_model_smooth_2.pt
- 10 epochs, noise=0.5, smooth=2.0
- Heavy smoothness penalty

### steer_model_scheduled.pt (training)
- **Scheduled sampling** with simulated dynamics
- Instead of random noise, simulates realistic error accumulation:
  ```
  steer_effect = 0.7 * steer + roll
  simulated_lat += 0.3 * (steer_effect - simulated_lat)
  ```
- Model sees realistic target vs actual mismatch during training
- noise=0.5, smooth=0.1, 20 epochs
- Should handle distribution shift better than pure noise injection

## Key Insights
1. **Distribution shift**: Model trained on ground truth sees different states at inference
2. **Smoothness**: Penalize steer changes to reduce oscillation
3. **Scheduled sampling**: Simulate realistic errors without running slow tinyphysics
4. **MPC warm start**: Neural model proposes, ONNX dynamics evaluates, CEM optimizes
