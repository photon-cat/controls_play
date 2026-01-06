"""
Action space constraints to prevent wild exploration in PPO steering control.

Problem: Agent initialized with PID behavior but explores too much and breaks.
Solution: Constrain action space and exploration.
"""

# Option 1: Reduce exploration noise (action std)
# Current: log_std = -1.0 → std = 0.37
# Too much exploration after BC initialization

class ConservativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # ... network ...
        self.log_std = nn.Parameter(torch.tensor(-2.0))  # std = 0.135 (less exploration)
        # Or even -3.0 → std = 0.05 (very conservative)


# Option 2: Decay exploration over time
class DecayingExplorationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.tensor(-1.0))
        self.min_log_std = -3.0  # Never go below this
    
    def decay_std(self, epoch, total_epochs):
        """Reduce exploration as training progresses"""
        # Linear decay: -1.0 → -3.0 over training
        new_log_std = -1.0 - 2.0 * (epoch / total_epochs)
        self.log_std.data = torch.tensor(max(new_log_std, self.min_log_std))


# Option 3: KL penalty to stay near BC policy
def ppo_update_with_kl_to_bc(policy, bc_policy, ...):
    """
    Add penalty for diverging from BC-initialized policy.
    Keeps agent close to PID behavior.
    """
    # Normal PPO loss
    policy_loss = ...
    
    # KL divergence to BC policy
    with torch.no_grad():
        bc_mean = bc_policy(states)
    current_mean = policy(states)
    
    # L2 distance (simpler than KL for Gaussian)
    bc_penalty = torch.mean((current_mean - bc_mean) ** 2)
    
    # Add to loss
    total_loss = policy_loss + 0.5 * bc_penalty  # tune weight
    return total_loss


# Option 4: Trust region - hard constraint on action change
class TrustRegionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_action = 0.0
        self.max_action_change = 0.3  # Max change per step
    
    def forward(self, x, deterministic=False):
        mean = self.net(x) * 2.0
        
        # Clip to trust region
        mean = torch.clamp(
            mean,
            self.prev_action - self.max_action_change,
            self.prev_action + self.max_action_change
        )
        
        # ... rest of forward ...
        return action


# Option 5: Action filter (exponential moving average)
class SmoothedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_action = 0.0
        self.smoothing = 0.7  # Higher = smoother
    
    def forward(self, x, deterministic=False):
        raw_action = self.net(x) * 2.0
        
        # Smooth with previous action
        smoothed_action = (self.smoothing * self.prev_action + 
                          (1 - self.smoothing) * raw_action)
        
        self.prev_action = smoothed_action.detach()
        return smoothed_action


# Option 6: Smaller PPO clip (more conservative updates)
def ppo_update_conservative(..., clip_eps=0.1):  # Instead of 0.2
    """
    Smaller clip → smaller policy changes per update
    Prevents catastrophic forgetting of BC initialization
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * advantages  # Tighter
    ...


# Option 7: Freeze policy partially (fine-tune only head)
def freeze_backbone():
    """
    Freeze lower layers, only train output head.
    Preserves learned features from BC.
    """
    for param in policy.net[:-2].parameters():  # Freeze all but last layer
        param.requires_grad = False


"""
RECOMMENDED COMBINATION:

1. Start with LOWER exploration:
   log_std = -2.0  # std = 0.135 instead of 0.37

2. Use SMALLER PPO clip:
   clip_eps = 0.1  # instead of 0.2

3. Add ACTION SMOOTHNESS to reward:
   reward = -(lat_error * 50 + jerk + 5 * action_change^2)

4. Optional: Decay std over epochs:
   Start -2.0 → end at -3.0

This keeps agent exploring but MUCH more conservatively around PID behavior.
"""


# PRACTICAL FIX - Update these parameters:
class SteeringPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # ...
        self.log_std = nn.Parameter(torch.tensor(-2.0))  # WAS -1.0
        #                                          ^^^^^ CHANGE THIS

# And in ppo_update():
def ppo_update(..., clip_eps=0.1):  # WAS 0.2
    #                        ^^^ CHANGE THIS
    ...

