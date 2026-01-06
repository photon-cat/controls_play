"""
Position-based PID controller.

Strategy:
1. Pre-compute target trajectory (waypoints) from target_lataccel for whole segment
2. Track actual position by integrating vehicle motion
3. Compute cross-track error (perpendicular distance from actual to target path)
4. Apply PID control on cross-track error to generate steer command

Cross-track error is positive when vehicle is to the right of target path.
"""

from . import BaseController
import numpy as np

ACC_G = 9.81
DEL_T = 0.1  # 10 Hz


def compute_waypoints_from_lataccel(v_ego_sequence, lataccel_sequence, roll_sequence):
    """
    Compute 2D waypoints from velocity and lateral acceleration.

    Returns: Nx2 array of [x, y] positions
    """
    n = len(v_ego_sequence)
    waypoints = np.zeros((n, 2))

    x, y, heading = 0.0, 0.0, 0.0

    for i in range(n):
        waypoints[i] = [x, y]

        v = v_ego_sequence[i]
        lat_a = lataccel_sequence[i]
        roll = roll_sequence[i]

        # Remove roll contribution
        roll_lataccel = np.sin(roll) * ACC_G
        turning_lataccel = lat_a - roll_lataccel

        # Compute yaw rate
        if abs(v) > 0.1:
            yaw_rate = turning_lataccel / v
        else:
            yaw_rate = 0.0

        # Update heading and position
        heading += yaw_rate * DEL_T
        x += v * np.cos(heading) * DEL_T
        y += v * np.sin(heading) * DEL_T

    return waypoints, heading


def find_closest_point_on_path(point, path):
    """
    Find the closest point on the target path to the given point.

    Returns:
        - closest_point: [x, y] on path
        - segment_idx: index of path segment
        - cross_track_error: signed distance (positive = right of path)
    """
    point = np.array(point)

    # Find closest segment
    min_dist = float('inf')
    closest_point = None
    closest_idx = 0

    for i in range(len(path) - 1):
        # Project point onto line segment
        p1 = path[i]
        p2 = path[i + 1]

        # Vector from p1 to p2
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)

        if segment_length < 1e-6:
            # Degenerate segment, use p1
            proj = p1
        else:
            # Vector from p1 to point
            to_point = point - p1

            # Project onto segment
            t = np.dot(to_point, segment) / (segment_length ** 2)
            t = np.clip(t, 0, 1)  # Clamp to segment

            proj = p1 + t * segment

        dist = np.linalg.norm(point - proj)
        if dist < min_dist:
            min_dist = dist
            closest_point = proj
            closest_idx = i

    # Compute signed cross-track error
    # Positive = right of path, negative = left of path
    if closest_idx < len(path) - 1:
        p1 = path[closest_idx]
        p2 = path[closest_idx + 1]

        # Path direction vector
        path_dir = p2 - p1
        path_dir_norm = np.linalg.norm(path_dir)

        if path_dir_norm > 1e-6:
            path_dir = path_dir / path_dir_norm

            # Vector from closest point to actual position
            offset = point - closest_point

            # Cross product to determine sign (right is positive)
            # In 2D: cross(path_dir, offset) = path_dir[0]*offset[1] - path_dir[1]*offset[0]
            cross = path_dir[0] * offset[1] - path_dir[1] * offset[0]
            cross_track_error = np.sign(cross) * min_dist
        else:
            cross_track_error = min_dist
    else:
        cross_track_error = min_dist

    return closest_point, closest_idx, cross_track_error


class Controller(BaseController):
    def __init__(self):
        # PID gains (much more conservative)
        self.kp = 0.08   # Proportional gain
        self.ki = 0.001  # Integral gain
        self.kd = 0.05   # Derivative gain

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        # Anti-windup limit
        self.integral_limit = 1.0

        # Lookahead distance for stability
        self.lookahead = 5  # waypoints ahead

        # Position tracking
        self.actual_x = 0.0
        self.actual_y = 0.0
        self.actual_heading = 0.0

        # Target trajectory (will be set on first update)
        self.target_waypoints = None
        self.segment_initialized = False

        # History for trajectory computation
        self.v_ego_history = []
        self.target_lataccel_history = []
        self.roll_history = []

        # Current timestep tracker
        self.current_timestep = 0

        print("Position-based PID controller initialized")

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.actual_x = 0.0
        self.actual_y = 0.0
        self.actual_heading = 0.0
        self.target_waypoints = None
        self.segment_initialized = False
        self.v_ego_history = []
        self.target_lataccel_history = []
        self.roll_history = []
        self.current_timestep = 0

    def initialize_segment(self, v_ego_seq, target_lataccel_seq, roll_seq):
        """
        Pre-compute target trajectory for the entire segment.
        Called once we have enough data.
        """
        print(f"Initializing segment with {len(v_ego_seq)} points...")
        self.target_waypoints, _ = compute_waypoints_from_lataccel(
            v_ego_seq, target_lataccel_seq, roll_seq
        )
        self.segment_initialized = True
        print(f"Target trajectory computed: {len(self.target_waypoints)} waypoints")

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        roll = state.roll_lataccel / ACC_G  # Convert back to roll angle
        roll = np.arcsin(np.clip(roll, -1, 1))

        # Store history for segment initialization
        self.v_ego_history.append(v_ego)
        self.target_lataccel_history.append(target_lataccel)
        self.roll_history.append(roll)

        # Initialize segment if we have future plan and haven't initialized yet
        if not self.segment_initialized and future_plan:
            if hasattr(future_plan, 'lataccel') and len(future_plan.lataccel) > 0:
                # Build full sequence from history + current + future
                full_v_ego = self.v_ego_history + [v_ego] + list(future_plan.v_ego)
                full_target = self.target_lataccel_history + [target_lataccel] + list(future_plan.lataccel)

                # Reconstruct roll from roll_lataccel
                future_roll = []
                for roll_lat in future_plan.roll_lataccel:
                    r = roll_lat / ACC_G
                    r = np.arcsin(np.clip(r, -1, 1))
                    future_roll.append(r)
                full_roll = self.roll_history + [roll] + future_roll

                self.initialize_segment(full_v_ego, full_target, full_roll)

        # Update actual position by integrating current motion
        roll_lataccel = state.roll_lataccel
        turning_lataccel = current_lataccel - roll_lataccel

        if abs(v_ego) > 0.1:
            yaw_rate = turning_lataccel / v_ego
        else:
            yaw_rate = 0.0

        self.actual_heading += yaw_rate * DEL_T
        self.actual_x += v_ego * np.cos(self.actual_heading) * DEL_T
        self.actual_y += v_ego * np.sin(self.actual_heading) * DEL_T

        # If segment not initialized yet, use simple proportional control
        if not self.segment_initialized:
            error = target_lataccel - current_lataccel
            steer = 0.3 * error
            self.current_timestep += 1
            return np.clip(steer, -2, 2)

        # Find cross-track error
        actual_pos = np.array([self.actual_x, self.actual_y])
        _, closest_idx, cross_track_error = find_closest_point_on_path(actual_pos, self.target_waypoints)

        # Use lookahead point for better stability
        if self.target_waypoints is not None and len(self.target_waypoints) > 0:
            lookahead_idx = min(closest_idx + self.lookahead, len(self.target_waypoints) - 1)
            target_point = self.target_waypoints[lookahead_idx]
        else:
            # Fallback
            error = target_lataccel - current_lataccel
            steer = 0.3 * error
            self.current_timestep += 1
            return np.clip(steer, -2, 2)

        # Compute heading error to lookahead point
        dx = target_point[0] - self.actual_x
        dy = target_point[1] - self.actual_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - self.actual_heading

        # Normalize heading error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Combined error: cross-track error + heading error
        error = cross_track_error + 0.5 * heading_error

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self.integral += error * DEL_T
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative
        d_term = self.kd * (error - self.prev_error) / DEL_T
        self.prev_error = error

        # Compute steer command
        # Positive error (right of path or heading right) -> negative steer (turn left)
        steer = -(p_term + i_term + d_term)

        # Clip to valid range
        steer = np.clip(steer, -2, 2)

        self.current_timestep += 1

        return steer
