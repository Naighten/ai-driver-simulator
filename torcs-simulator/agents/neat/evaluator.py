"""Genome evaluator for NEAT.

Evaluates a single genome by:
  1. Creating a FeedForwardNetwork from the genome.
  2. Running it in a TORCS environment for episode_steps.
  3. Computing fitness from the accumulated rewards.
"""

import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import neat

from environment.torcs_env import TorcsEnv
from agents.neat.fitness import compute_fitness


def eval_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    env: TorcsEnv,
    max_steps: int = 3000,
) -> float:
    """Evaluate a single genome in a TORCS environment.

    Args:
        genome: NEAT genome to evaluate.
        config: NEAT configuration object.
        env: a TorcsEnv instance (already connected).
        max_steps: maximum steps for the evaluation episode.

    Returns:
        float: fitness score.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    obs, info = env.reset()
    rewards: List[float] = []
    final_dist = 0.0

    for step in range(max_steps):
        # NEAT network outputs 3 values in [-1, 1] (tanh)
        output = net.activate(obs)
        action = np.array(output, dtype=np.float32)

        # Map outputs to valid action ranges
        action[0] = np.clip(action[0], -1.0, 1.0)   # steer
        action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel: [-1,1] -> [0,1]
        action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake: [-1,1] -> [0,1]

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)
        final_dist = step_info.get("dist_raced", 0.0) or 0.0

        if terminated or truncated:
            break

        obs = next_obs

    fitness = compute_fitness(rewards, final_dist)
    return fitness
