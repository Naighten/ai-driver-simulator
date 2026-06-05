"""Evaluation script for trained SAC and NEAT agents.

Loads a checkpoint and runs the agent in the TORCS environment
without exploration noise, recording lap times and other metrics.

Usage:
    python eval.py --agent sac --checkpoint checkpoints/sac_checkpoint_step_10000.pt
    python eval.py --agent neat --checkpoint checkpoints/neat_best_genome_gen_50.pkl
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from environment.torcs_env import TorcsEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config") -> dict:
    cfg = {}
    for fname in ["base.yaml", "reward.yaml"]:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = yaml.safe_load(f)
                if data:
                    cfg.update(data)
    return cfg


def load_sac_for_eval(checkpoint_path: str) -> torch.nn.Module:
    """Load a SAC actor for evaluation (no training components)."""
    from agents.rl.actor import Actor

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = Actor(obs_dim=32, action_dim=3, hidden_dim=256, n_hidden=2)
    actor.load_state_dict(checkpoint["actor_state"])
    actor.eval()
    return actor


def load_neat_for_eval(checkpoint_path: str):
    """Load a NEAT genome and create a feedforward network for evaluation."""
    import pickle
    import neat

    config_path = os.path.join(
        os.path.dirname(__file__), "config", "neat_config.txt"
    )
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open(checkpoint_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    return net


def evaluate_sac(
    checkpoint_path: str, config: dict, episodes: int = 3, max_steps: int = 10000
) -> None:
    """Evaluate a trained SAC agent.

    Args:
        checkpoint_path: path to SAC checkpoint .pt file.
        config: merged config dictionary.
        episodes: number of evaluation episodes.
        max_steps: max steps per episode.
    """
    actor = load_sac_for_eval(checkpoint_path)

    torcs_cfg = config.get("torcs", {})
    reward_weights = config.get("reward", {})

    env = TorcsEnv(
        host="localhost",
        port=torcs_cfg.get("ports", [3001])[0],
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

    logger.info(f"Evaluating SAC agent from {checkpoint_path}")

    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        lap_times = []
        start_time = time.time()

        for step in range(max_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                h = actor.backbone(obs_tensor)
                mean = actor.mean_layer(h)
                # Deterministic: use mean, no sampling
                z = mean
                action = torch.tanh(z)
                action = action.squeeze(0).cpu().numpy()

            action[1] = max(0.0, min(1.0, action[1]))
            action[2] = max(0.0, min(1.0, action[2]))

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward

            # Track lap time if available
            last_lap = step_info.get("last_lap_time")
            if last_lap is not None and last_lap > 0:
                lap_times.append(last_lap)

            if terminated or truncated:
                reason = "off-track/Stuck" if terminated else "max_steps"
                logger.info(
                    f"Episode {ep + 1}: reward={episode_reward:.1f}, "
                    f"steps={step + 1}, reason={reason}"
                )
                if lap_times:
                    logger.info(f"  Lap times: {[f'{t:.2f}' for t in lap_times]}")
                break

            obs = next_obs

        elapsed = time.time() - start_time
        logger.info(
            f"Episode {ep + 1} finished in {elapsed:.1f}s, "
            f"total_reward={episode_reward:.1f}"
        )

    env.close()


def evaluate_neat(
    checkpoint_path: str, config: dict, episodes: int = 3, max_steps: int = 10000
) -> None:
    """Evaluate a trained NEAT agent.

    Args:
        checkpoint_path: path to NEAT genome .pkl file.
        config: merged config dictionary.
        episodes: number of evaluation episodes.
        max_steps: max steps per episode.
    """
    net = load_neat_for_eval(checkpoint_path)

    torcs_cfg = config.get("torcs", {})
    reward_weights = config.get("reward", {})

    env = TorcsEnv(
        host="localhost",
        port=torcs_cfg.get("ports", [3001])[0],
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

    logger.info(f"Evaluating NEAT agent from {checkpoint_path}")

    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        lap_times = []
        start_time = time.time()

        for step in range(max_steps):
            output = net.activate(obs.tolist())
            action = np.array(output, dtype=np.float32)
            action[0] = np.clip(action[0], -1.0, 1.0)
            action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
            action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward

            last_lap = step_info.get("last_lap_time")
            if last_lap is not None and last_lap > 0:
                lap_times.append(last_lap)

            if terminated or truncated:
                reason = "off-track/stuck" if terminated else "max_steps"
                logger.info(
                    f"Episode {ep + 1}: reward={episode_reward:.1f}, "
                    f"steps={step + 1}, reason={reason}"
                )
                if lap_times:
                    logger.info(f"  Lap times: {[f'{t:.2f}' for t in lap_times]}")
                break

            obs = next_obs

        elapsed = time.time() - start_time
        logger.info(
            f"Episode {ep + 1} finished in {elapsed:.1f}s, "
            f"total_reward={episode_reward:.1f}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TORCS agents")
    parser.add_argument(
        "--agent", type=str, required=True, choices=["sac", "neat"],
        help="Agent type to evaluate"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of evaluation episodes (default: 3)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Max steps per episode (default: 10000)"
    )
    args = parser.parse_args()

    config = load_config()

    if args.agent == "sac":
        evaluate_sac(args.checkpoint, config, args.episodes, args.max_steps)
    elif args.agent == "neat":
        evaluate_neat(args.checkpoint, config, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
