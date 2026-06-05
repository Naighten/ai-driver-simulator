"""NEAT training script for TORCS.

Uses neat-python with parallel evaluation across multiple TORCS instances.

Usage:
    python -m agents.neat.train_neat --mode train
    python -m agents.neat.train_neat --mode resume --checkpoint <path>
"""

import argparse
import logging
import os
import signal
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import neat
import yaml

from runner.instance_manager import InstanceManager
from utils.logger import Logger
from utils.checkpoint import save_neat_checkpoint, load_neat_checkpoint
from agents.neat.evaluator import eval_genome

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML config files and merge into a single dict."""
    cfg = {}
    for fname in ["base.yaml", "reward.yaml", "neat.yaml"]:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f)
                if data:
                    cfg.update(data)
    return cfg


def create_neat_config(config_dir: str = "config") -> neat.Config:
    """Create a neat.Config from the configuration file and YAML overrides.

    Reads the base neat_config.txt and applies YAML overrides from neat.yaml.
    """
    config_path = os.path.join(config_dir, "neat_config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"NEAT config file not found: {config_path}")
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def train_neat(config: dict, checkpoint_path: Optional[str] = None) -> None:
    """Main NEAT training loop.

    Args:
        config: merged configuration dict.
        checkpoint_path: optional path to resume from checkpoint.
    """
    neat_cfg = config.get("neat", {})
    torcs_cfg = config.get("torcs", {})
    reward_weights = config.get("reward", {})
    ports = torcs_cfg.get("ports", [3001])
    log_dir = config.get("logging", {}).get("log_dir", "runs/neat")
    checkpoint_dir = config.get("checkpoint", {}).get("dir", "checkpoints")
    max_generations = neat_cfg.get("max_generations", 200)
    episode_steps = neat_cfg.get("episode_steps", 3000)

    # Create NEAT config
    neat_config = create_neat_config()

    # Override pop_size from YAML
    pop_size = neat_cfg.get("population_size", 48)
    neat_config.pop_size = pop_size

    # Create instance manager with envs on all ports
    instance_manager = InstanceManager(
        ports=ports,
        host="localhost",
        timeout=torcs_cfg.get("timeout", 1.0),
        max_steps=torcs_cfg.get("max_steps", 50000),
        grace_steps=torcs_cfg.get("grace_steps", 100),
        offtrack_limit=torcs_cfg.get("offtrack_limit", 1.5),
        offtrack_patience=torcs_cfg.get("offtrack_patience", 5),
        stuck_speed_threshold=torcs_cfg.get("stuck_speed_threshold", 2.0),
        stuck_steps=torcs_cfg.get("stuck_steps", 100),
        smoothing_alpha=config.get("action", {}).get("smoothing_alpha", 0.6),
        action_repeat=config.get("action", {}).get("repeat", 4),
        min_accel=config.get("action", {}).get("min_accel", 0.2),
        reward_weights=reward_weights,
    )

    # Check port availability
    available = instance_manager.check_ports()
    if not available:
        logger.error("No TORCS instances available. Exiting.")
        sys.exit(1)

    # Create population
    if checkpoint_path:
        pop_data = load_neat_checkpoint(checkpoint_path)
        population = neat.Population(
            neat_config,
            initial_state=pop_data["population"],
        )
        population.species = pop_data["species"]
        start_gen = pop_data["generation"] + 1
        logger.info(f"Resumed NEAT from generation {pop_data['generation']}")
    else:
        population = neat.Population(neat_config)
        start_gen = 0

    # Add reporters
    logger_writer = Logger(log_dir=log_dir, tag="neat")

    class NeatLoggerReporter(neat.reporting.BaseReporter):
        """Custom reporter that logs to TensorBoard and Python logger."""

        def __init__(self, writer: Logger):
            self.writer = writer
            self.generation = 0

        def start_generation(self, generation):
            self.generation = generation
            logger.info(f"Generation {generation} started")

        def end_generation(self, config, population, species_set):
            pass

        def post_evaluate(self, config, population, species, best_genome):
            gen = self.generation
            if best_genome is not None:
                metrics = {
                    "max_fitness": best_genome.fitness,
                    "mean_fitness": sum(g.fitness for g in population.values()) / len(population),
                    "species": len(species.species),
                    "nodes": best_genome.size()[0],
                    "connections": best_genome.size()[1],
                }
                self.writer.log_scalars(gen, metrics, prefix="neat")
                logger.info(
                    f"Gen {gen}: best={best_genome.fitness:.2f}, "
                    f"species={metrics['species']}, "
                    f"nodes={metrics['nodes']}, conns={metrics['connections']}"
                )

        def complete_extinction(self):
            logger.warning("Population extinct!")

        def found_solution(self, config, generation, best):
            logger.info(f"Solution found at generation {generation}!")

    neat_reporter = NeatLoggerReporter(logger_writer)
    population.add_reporter(neat_reporter)
    population.add_reporter(neat.StdOutReporter(True))

    # Fitness function for parallel evaluation
    def eval_genomes(genomes, neat_config_local):
        """Evaluate all genomes in a generation across available TORCS instances."""
        results = []
        for genome_id, genome in genomes:
            env_tuple = instance_manager.acquire_free()
            if env_tuple is None:
                # Fallback: reuse any env
                port = list(instance_manager._envs.keys())[0]
                env = instance_manager.get_env(port)
            else:
                port, env = env_tuple

            try:
                fitness = eval_genome(genome, neat_config_local, env, episode_steps)
                genome.fitness = fitness
                results.append((genome_id, fitness))
            except Exception as e:
                logger.error(f"Error evaluating genome {genome_id}: {e}")
                genome.fitness = 0.0
            finally:
                if env_tuple is not None:
                    instance_manager.release(port)

        # Calculate average fitness
        if results:
            avg_fitness = sum(f for _, f in results) / len(results)
            logger.info(f"Generation avg fitness: {avg_fitness:.2f}")

    # Signal handler
    stop_training = False

    def signal_handler(sig, frame):
        nonlocal stop_training
        logger.info(f"Received signal {sig}, stopping after current generation...")
        stop_training = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run evolution
        generations_to_run = max_generations - start_gen
        if generations_to_run > 0:
            best_genome = population.run(
                eval_genomes,
                generations_to_run,
            )

            # Save final checkpoint
            save_neat_checkpoint(
                checkpoint_dir,
                population,
                start_gen + generations_to_run - 1,
                best_genome,
                metadata={"config": config},
            )
            logger.info(
                f"Training complete. Best fitness: {best_genome.fitness:.2f}"
            )

    except Exception as e:
        logger.error(f"Training error: {e}")
    finally:
        # Save checkpoint on exit
        try:
            save_neat_checkpoint(
                checkpoint_dir,
                population,
                start_gen + generations_to_run - 1 if "generations_to_run" in dir() else start_gen,
                None,
                metadata={"config": config, "interrupted": stop_training},
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        instance_manager.close_all()
        logger_writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train NEAT agent for TORCS")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "resume"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    if args.mode == "resume":
        if not args.checkpoint:
            logger.error("Checkpoint path required for resume mode")
            sys.exit(1)
        train_neat(config, args.checkpoint)
    else:
        train_neat(config)


if __name__ == "__main__":
    main()
