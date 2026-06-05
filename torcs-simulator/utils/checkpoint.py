"""Checkpoint save/load utilities for SAC and NEAT agents.

Supports three trigger types:
  1. Valid lap completed
  2. Manual stop (Ctrl+C / SIGINT)
  3. Crash (SIGTERM / try/finally)

No periodic checkpointing.
"""

import os
import pickle
import torch
from typing import Dict, Any, Optional


def save_sac_checkpoint(
    path: str,
    actor_state: Dict[str, Any],
    critic_state: Dict[str, Any],
    target_critic_state: Dict[str, Any],
    actor_optimizer_state: Dict[str, Any],
    critic_optimizer_state: Dict[str, Any],
    alpha: float,
    log_alpha: Any,
    alpha_optimizer_state: Dict[str, Any],
    normalizer_state: Optional[Dict[str, Any]] = None,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a complete SAC training checkpoint.

    Args:
        path: directory path to save checkpoint.
        actor_state: Actor state_dict.
        critic_state: Critic (twin) state_dict.
        target_critic_state: target Critic state_dict.
        actor_optimizer_state: actor optimizer state_dict.
        critic_optimizer_state: critic optimizer state_dict.
        alpha: entropy coefficient.
        log_alpha: log of alpha (for autotune).
        alpha_optimizer_state: alpha optimizer state_dict.
        normalizer_state: optional RunningMeanStd state.
        step: current training step.
        metadata: optional dict with additional info.

    Returns:
        str: full path to the saved checkpoint file.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"sac_checkpoint_step_{step}.pt")

    checkpoint = {
        "actor_state": actor_state,
        "critic_state": critic_state,
        "target_critic_state": target_critic_state,
        "actor_optimizer_state": actor_optimizer_state,
        "critic_optimizer_state": critic_optimizer_state,
        "alpha": alpha,
        "log_alpha": log_alpha,
        "alpha_optimizer_state": alpha_optimizer_state,
        "normalizer_state": normalizer_state,
        "step": step,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, filepath)
    return filepath


def load_sac_checkpoint(
    path: str,
) -> Dict[str, Any]:
    """Load a SAC checkpoint from disk.

    Args:
        path: full path to the .pt checkpoint file.

    Returns:
        dict with all checkpoint entries.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def save_neat_checkpoint(
    path: str,
    population,
    generation: int,
    best_genome=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a NEAT population checkpoint using neat-python's Checkpointer format.

    Args:
        path: directory path to save checkpoint.
        population: neat.Population object.
        generation: current generation number.
        best_genome: optional best genome to save separately.
        metadata: optional dict with additional info.

    Returns:
        str: full path to the saved checkpoint file.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"neat_checkpoint_gen_{generation}")

    # Save population state via pickle
    pop_state = {
        "generation": generation,
        "population": population.population,
        "species": population.species,
        "config": population.config,
        "metadata": metadata or {},
    }
    with open(filepath + ".pkl", "wb") as f:
        pickle.dump(pop_state, f)

    # Save best genome separately
    if best_genome is not None:
        best_path = os.path.join(path, f"neat_best_genome_gen_{generation}.pkl")
        with open(best_path, "wb") as f:
            pickle.dump(best_genome, f)

    return filepath


def load_neat_checkpoint(path: str):
    """Load a NEAT population checkpoint from disk.

    Args:
        path: full path to the checkpoint file (without extension).

    Returns:
        dict with population state.
    """
    # Try .pkl first, then neat-python native format
    pkl_path = path + ".pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"NEAT checkpoint not found at {path}")


def find_latest_checkpoint(
    path: str, prefix: str = "sac_checkpoint_step_"
) -> Optional[str]:
    """Find the latest checkpoint file by step number in filename.

    Args:
        path: directory to search.
        prefix: filename prefix to filter by.

    Returns:
        str: full path to latest checkpoint, or None if not found.
    """
    if not os.path.isdir(path):
        return None

    candidates = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(".pt")]
    if not candidates:
        return None

    # Parse step numbers and pick the highest
    steps = []
    for f in candidates:
        step_str = f.replace(prefix, "").replace(".pt", "")
        try:
            steps.append((int(step_str), f))
        except ValueError:
            continue

    if not steps:
        return None

    steps.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(path, steps[0][1])
