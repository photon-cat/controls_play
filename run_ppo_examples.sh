#!/bin/bash
# Quick PPO training examples

echo "===== PPO Training Examples ====="
echo ""

# Activate environment
source ~/.zshrc
conda activate ai

echo "1. Fast CartPole test (50 episodes, ~2 min)"
python3 test_ppo_cartpole.py --env cartpole --episodes 50

echo ""
echo "2. Full CartPole training (200 episodes, ~8 min)"
# python3 test_ppo_cartpole.py --env cartpole --episodes 200

echo ""
echo "3. Pendulum continuous control (50 episodes)"
# python3 test_ppo_cartpole.py --env pendulum --episodes 50

echo ""
echo "Done! Check results above."

