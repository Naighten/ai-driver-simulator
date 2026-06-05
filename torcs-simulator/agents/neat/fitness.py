"""Fitness computation for NEAT genome evaluation.

The fitness is the cumulative reward over an episode, identical to the
reward function used for the SAC agent.
"""

from typing import List


def compute_fitness(rewards: List[float], dist_raced: float) -> float:
    """Compute the fitness score for a genome.

    Base fitness is the sum of per-step rewards. A small bonus is added
    for distance raced to encourage exploration in early generations.

    Args:
        rewards: list of per-step rewards for the episode.
        dist_raced: total distance raced (meters) in the episode.

    Returns:
        float: fitness score (higher is better).
    """
    base_fitness = sum(rewards)
    distance_bonus = dist_raced * 0.01
    return base_fitness + distance_bonus
