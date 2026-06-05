"""Reward function for the TORCS environment.

Based on the working reference with:
  - look_ahead_bonus: reward for open space ahead (primary navigation signal)
  - speed_bonus: directed forward speed
  - center_bonus: reward for being near track center
  - edge_penalty: quadratic penalty near track edge
  - steer_smooth_penalty: soft limit on steering delta
  - backward_penalty: penalty for moving backwards
  - grace_bonus: flat bonus during grace period
"""

import math
from typing import Dict, List


def compute_reward(
    speed_x: float,
    angle: float,
    track_pos: float,
    damage: float,
    prev_damage: float,
    dist_raced: float,
    prev_dist_raced: float,
    track: List[float],
    current_steer: float,
    prev_steer: float,
    step_count: int,
    grace_steps: int,
    weights: Dict[str, float],
) -> float:
    w_speed = weights.get("speed", 1.0)
    w_progress = weights.get("progress", 2.0)
    w_center = weights.get("center", 0.5)
    w_look_ahead = weights.get("look_ahead", 0.3)
    w_edge = weights.get("edge", 10.0)
    w_edge_speed = weights.get("edge_speed", 0.18)
    w_steer_speed = weights.get("steer_speed", 0.5)
    w_backward = weights.get("backward", 5.0)
    w_steer_smooth = weights.get("steer_smooth", 0.3)
    w_steer_mag = weights.get("steer_mag", 0.5)
    w_idle = weights.get("idle", 1.0)
    w_damage = weights.get("damage", 0.0)
    grace_bonus = weights.get("grace_bonus", 1.0)
    edge_threshold = weights.get("edge_threshold", 0.7)

    # Look ahead: min of front track sensors (indices 7-11 = track straight ahead)
    front = track[7:12] if len(track) >= 12 else track
    front_distance = min(front) / 200.0 if front else 0.0

    # Forward speed (projected onto track axis)
    forward_speed = max(speed_x * math.cos(angle), 0.0)

    # Progress (capped at 2m to prevent gradient spikes on straights)
    progress = dist_raced - prev_dist_raced
    if progress < -5.0:
        progress = 0.0

    speed_bonus = forward_speed * w_speed
    progress_bonus = min(max(progress, 0.0), 2.0) * w_progress

    abs_tp = abs(track_pos)
    center_bonus = max(0.0, 1.0 - abs_tp / edge_threshold) * w_center
    look_ahead_bonus = front_distance * w_look_ahead

    if abs_tp > edge_threshold:
        edge_penalty = (abs_tp - edge_threshold) ** 2 * w_edge
    else:
        edge_penalty = 0.0

    edge_speed_penalty = (
        abs_tp * speed_x * max(speed_x - 30.0, 0.0) * w_edge_speed / 100.0
    )

    steer_delta = abs(current_steer - prev_steer)
    steer_penalty = steer_delta * w_steer_smooth + abs(current_steer) * w_steer_mag
    steer_speed_penalty = steer_delta * max(speed_x - 80.0, 0.0) * w_steer_speed

    backward_penalty = w_backward if progress < -0.5 else 0.0

    idle_penalty = w_idle if (speed_x < 2.0 and step_count > grace_steps) else 0.0

    damage_delta = max(0.0, damage - prev_damage)
    damage_penalty = damage_delta * w_damage

    reward = (
        speed_bonus
        + progress_bonus
        + center_bonus
        + look_ahead_bonus
        - edge_penalty
        - edge_speed_penalty
        - steer_penalty
        - steer_speed_penalty
        - backward_penalty
        - idle_penalty
        - damage_penalty
    )

    if step_count <= grace_steps:
        reward += grace_bonus

    return float(reward)
