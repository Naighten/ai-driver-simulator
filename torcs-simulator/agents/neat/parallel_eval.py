"""Parallel evaluator wrapper for NEAT.

Uses neat-python's built-in ParallelEvaluator with a pool of TORCS environment
instances managed by InstanceManager.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import neat
from neat.parallel import ParallelEvaluator

from agents.neat.evaluator import eval_genome


def create_parallel_evaluator(
    num_workers: int,
    neat_config: neat.Config,
    instance_manager,
    max_steps: int = 3000,
) -> ParallelEvaluator:
    """Create a ParallelEvaluator for NEAT training.

    Args:
        num_workers: number of parallel workers.
        neat_config: NEAT configuration object.
        instance_manager: InstanceManager with pre-created envs.
        max_steps: maximum steps per genome evaluation.

    Returns:
        Configured ParallelEvaluator instance.
    """

    def eval_function(genomes, neat_config_local):
        """Evaluate a list of genomes sequentially using envs from the pool."""
        for genome_id, genome in genomes:
            env_tuple = instance_manager.acquire_free()
            if env_tuple is None:
                # Fallback: use first env
                port = instance_manager._envs[list(instance_manager._envs.keys())[0]]
                env = instance_manager.get_env(port)
            else:
                port, env = env_tuple

            try:
                fitness = eval_genome(genome, neat_config_local, env, max_steps)
                genome.fitness = fitness
            finally:
                instance_manager.release(port)

    evaluator = ParallelEvaluator(
        num_workers=num_workers,
        eval_function=eval_function,
        timeout=None,
    )

    return evaluator
