"""
Reward shaping options for steering control RL.

Current approach: reward = -(lat_error * 50 + jerk)
Problems: scale, delayed effects, no action smoothness

Better options:
"""

# Option 1: Normalize rewards to [-1, 1] range
def reward_normalized(lat_error, jerk):
    """
    Scale rewards to reasonable range for RL.
    Helps value network learn faster.
    """
    # Typical errors: lat_error ~0.01-1.0, jerk ~0.1-10
    lat_cost = lat_error * 50.0  # 0.5-50
    jerk_cost = jerk  # 0.1-10
    total_cost = lat_cost + jerk_cost
    
    # Normalize: cost of 100 = reward of -1
    reward = -total_cost / 100.0
    return reward


# Option 2: Add action smoothness penalty
def reward_with_action_smoothness(lat_error, jerk, action, prev_action):
    """
    Penalize rapid steering changes (not just lataccel changes).
    Encourages smoother control like PID.
    """
    lat_cost = lat_error * 50.0
    jerk_cost = jerk
    action_change = (action - prev_action) ** 2
    action_smoothness_cost = action_change * 5.0  # tune this weight
    
    total_cost = lat_cost + jerk_cost + action_smoothness_cost
    reward = -total_cost / 100.0
    return reward


# Option 3: Sparse terminal reward (episode-level)
def reward_sparse(done, episode_cost):
    """
    Only give reward at end of episode.
    Simple but harder to learn (credit assignment problem).
    """
    if done:
        # Good performance ~100, bad ~5000
        reward = -episode_cost / 100.0  # Scale to [-50, -1]
    else:
        reward = 0.0  # No intermediate feedback
    return reward


# Option 4: Hybrid - sparse + small dense
def reward_hybrid(lat_error, jerk, done, episode_cost):
    """
    Small step rewards + big terminal reward.
    Best of both worlds.
    """
    # Small dense reward for immediate feedback
    step_reward = -(lat_error * 50.0 + jerk) / 100.0
    
    # Big terminal reward for final cost
    if done:
        terminal_reward = -episode_cost / 10.0  # Larger magnitude
    else:
        terminal_reward = 0.0
    
    return step_reward + terminal_reward


# Option 5: Potential-based shaping (theoretically optimal)
def reward_potential_based(state, next_state, cost):
    """
    Add potential function that doesn't change optimal policy.
    Phi(s) could be distance to target trajectory.
    """
    gamma = 0.99
    
    # Potential: how far are we from target?
    phi_s = -abs(state['target_lat'] - state['current_lat']) * 10
    phi_next = -abs(next_state['target_lat'] - next_state['current_lat']) * 10
    
    # Base reward
    base_reward = -cost
    
    # Add potential difference
    shaping = gamma * phi_next - phi_s
    
    return base_reward + shaping


# Option 6: Curiosity/exploration bonus (for sparse rewards)
def reward_with_curiosity(base_reward, state_novelty):
    """
    Add bonus for exploring new states.
    Helps with sparse reward learning.
    """
    curiosity_bonus = state_novelty * 0.1  # small bonus
    return base_reward + curiosity_bonus


"""
RECOMMENDATION:
Start with Option 1 (normalized) or Option 4 (hybrid).

Option 1 is simplest fix - just rescale current approach.
Option 4 gives better credit assignment by emphasizing final cost.

For your case, I'd try:
- Step reward: -(lat_error * 50 + jerk) / 100  (normalized)
- Terminal reward: -total_cost / 10  (big signal at end)
- Action smoothness: Add penalty for (action - prev_action)^2

This way:
1. Agent gets immediate feedback (dense)
2. Agent knows final score matters more (terminal)
3. Agent learns smooth control (action penalty)
"""

